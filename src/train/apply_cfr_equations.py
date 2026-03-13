import argparse
import os
import sys
import json
import re
import math
import csv
import numpy as np
from scipy.signal.windows import hann
from scipy.fftpack import fft, fftshift
import scipy

# Allow importing optimization utilities from the repo
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY_CODE_DIR = os.path.join(REPO_ROOT, 'Python_code')
if PY_CODE_DIR not in sys.path:
    sys.path.insert(0, PY_CODE_DIR)
from optimization_utility import build_T_matrix, lasso_regression_osqp_fast

# -----------------------------------------------------------------------------
# This module adapts the SHARP pipeline to ESP32 CSI. The code implements the
# following equations from the paper (numbers as provided in your prompt):
#
# CFR model and hardware offsets
# - Eq. (CRF/CFR): H_m(n) = A_m(n) e^{j φ_m(n)} = Σ_p A_p(n) e^{-j 2π (f_c + m/T) τ_p(n)}
# - Measured CFR with offsets: \bar H_m(n) = H_m(n) e^{j φ_offs,m}
# - Offset decomposition: φ_offs,m = -2π m (τ_SFO + τ_PDD)/T + φ_CFO + φ_PPO + φ_PA
#
# Sparse delay-domain model and sanitization
# - Eq. (4): CFR vector h collects subcarriers at a time n
# - Eq. (5): h = T r, dictionary T encodes phase vs delay across subcarriers
# - Eq. (6): T_k row uses e^{-j 2π f_k t_p,tot}
# - Eq. (7): r bundles path complex gains (with common phase)
# - Eq. (8): Lasso: argmin ||h - T r||_2^2 + λ ||r||_1
# - Eq. (9)-(10)-(11): real-valued expansion for OSQP and reconstruction of r
# - Eq. (12)-(13): per-path contributions X_k = T_k^T ⊙ r (implicit in T*r)
# - Eq. (14)-(15): dominant-path referencing removes common phase offsets
#
# Doppler extraction and velocity mapping
# - Eq. (19): H_i windowed CFR matrix (frequency x time)
# - Eq. (20)-(21): Doppler power via temporal FFT (Hann window + FFT)
# - Eq. (22): v = u c / (f_c T_c N_D), mapping Doppler bin to radial velocity
# -----------------------------------------------------------------------------
# Key variable shapes used across functions:
#   H_raw   : (N_pkts, tones_raw) complex64 — measured CFR \bar H_m(n)
#   idx     : (K,) int — HT20 subcarrier indices (data tones retained)
#   H       : (N_pkts, K) complex64 — CFR vector h(n) [Eq. (4)]
#   freq_vec: (K,) float64 — sub-carrier frequencies f_k in Hz [Eq. (6)]
#   T_matrix: (K, P_0) complex128 — delay dictionary T [Eq. (6)]
#   r       : (P_0,) complex128 — sparse path coefficients [Eq. (7)]
#   H_sanit : (N_pkts, K) complex128 — sanitized CFR \hat H_k(n) [Eq. (15)]
#   S       : (N_win, N_D) float64 — Doppler power spectrum d_i(u) [Eq. (20)]
#   ts      : (N_pkts,) float64 — raw timestamps (for T_c estimation) [Eq. (22)]
#   delta_v : float — velocity bin spacing (m/s) computed from Eq. (22)
# -----------------------------------------------------------------------------


def parse_int_list_field(s):
    # Try JSON first
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [int(x) for x in val]
    except Exception:
        pass
    # Robust fallback: extract all integer tokens (handles noisy logs)
    nums = re.findall(r"[-+]?\d+", s)
    return [int(x) for x in nums]


def estimate_Tc_seconds(ts_arr):
    ts_arr = np.asarray(ts_arr, float)
    if ts_arr.size < 2:
        return np.nan
    d = np.diff(ts_arr)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return np.nan
    med = float(np.median(d))
    if 1e3 <= med < 1e7:
        return med / 1e6  # microseconds
    if 1 <= med < 1e3:
        return med / 1e3  # milliseconds
    return med  # seconds


def load_esp32_csv_lltf51_simple(path, data_col='data', ts_col='local_timestamp'):
    # keep positions p in ESP32 LLTF order (0..31, -32..-1), drop: p=0 (DC), p=1 (+1 invalid),
    # guards p=27..31 (+27..+31) and p=32..37 (-32..-27)
    keep_idx = np.array(list(range(2, 27)) + list(range(38, 64)))  # 51 bins

    def parse_list(s):
        s = s.strip()
        if not (s.startswith('[') and s.endswith(']')): return []
        return [int(x) for x in s[1:-1].split(',') if x.strip()]

    H, ts = [], []
    with open(path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            vals = parse_list(row.get(data_col, '') or '')
            if len(vals) < 128:  # need 64 I/Q pairs
                continue
            buf = np.asarray(vals[:128], dtype=np.int8).reshape(-1, 2)  # [imag, real]
            csi64 = buf[:,1].astype(np.float32) + 1j*buf[:,0].astype(np.float32)
            H.append(csi64[keep_idx])
            t = row.get(ts_col)
            ts.append(float(t) if t not in (None, '') else np.nan)

    H = np.stack(H, axis=0).astype(np.complex64) if H else np.zeros((0,51), np.complex64)
    ts = np.asarray(ts, dtype=np.float64)
    return H, ts


def lltf51_indices():
    """Return sub-carrier index vector corresponding to the 51 LLTF bins kept by
    `load_esp32_csv_lltf51_simple` (guards & DC removed, pilots retained).

    Output order matches the full HT20 canonical order used elsewhere
    (negative frequencies first, then positive):
    [-26, …, -1, 2, …, 26]  (length = 51)
    """
    neg = np.arange(-26, 0)
    pos = np.arange(2, 27)
    return np.concatenate([neg, pos])





def reorder_or_pad(H, target):
    """Re-order and/or pad CFR columns so they exactly match a requested
    sub-carrier index list (true pilot filtering).

    Parameters
    ----------
    H : ndarray, shape (N_pkts, tones_raw)
        Raw measured CFR (ESP32 order −28…−1, +1…+28).
    target : array-like[int] *or* int
        • If **array** → desired sub-carrier indices (e.g. the vector returned by
          `ht20_indices(drop_pilots=True)`).  The function keeps those columns in
          that exact order and discards the rest (proper pilot removal).
        • If **int** → legacy behaviour: keep/pad to that length (for backward
          compatibility).

    Returns
    -------
    ndarray, shape (N_pkts, len(target)) – CFR aligned with `target`.
    """
    # --- Legacy: user passed a scalar length ----------------------------------
    if np.isscalar(target):
        target_len = int(target)
        if H.shape[1] == target_len:
            return H
        if H.shape[1] > target_len:
            return H[:, -target_len:]
        pad = target_len - H.shape[1]
        return np.pad(H, ((0, 0), (0, pad)), mode='constant')

    # --- New: user passed an index vector -------------------------------------
    idx = np.asarray(target, dtype=int)

    # Reference full HT20 index ordering produced by ESP32 payload
    full_idx = np.concatenate([np.arange(-28, 0), np.arange(1, 29)])  # length 56
    full_len = full_idx.size  # 56

    # Trim or pad H so it contains exactly the 56 expected tones --------------
    if H.shape[1] > full_len:
        H56 = H[:, -full_len:]
    elif H.shape[1] < full_len:
        pad = full_len - H.shape[1]
        H56 = np.pad(H, ((0, 0), (0, pad)), mode='constant')
    else:
        H56 = H

    # Build column mapping: where does each desired idx live in the 56-tone list?
    try:
        col_map = [int(np.where(full_idx == k)[0][0]) for k in idx]
    except IndexError:
        raise ValueError("Some requested indices are outside the HT20 set [-28..-1,1..28].")

    return H56[:, col_map]


def build_freq_vector_from_indices(indices, delta_f=312_500.0):
    """
    Map HT20 subcarrier indices to baseband frequencies (Hz): f_k = idx_k * Δf.
    These f_k instantiate the dictionary rows T_k in Eq. (6).
    """
    return indices.astype(np.float64) * float(delta_f)


def sanitize_phase(H_kept, freq_vec_hz, delta_t=1e-7, t_min=-3e-7, t_max=5e-7,
                   subcarrier_stride=2):
    """
    Phase sanitization via sparse delay-domain reconstruction and dominant path.
    - Build dictionary T over a delay grid t in [t_min, t_max] with step delta_t:
      T(row=f_k, col=t_p) = exp(-j 2π f_k t_p) [Eq. (6)].
    - For each time n, solve the Lasso [Eq. (8)] for r(n) with OSQP:
        argmin ||h(n) - T r(n)||_2^2 + λ ||r(n)||_1
      using real-valued expansion [Eq. (9)-(10)] and reconstruct r [Eq. (11)].
    - Compute per-path contributions Tr = T ⊙ r (implicit) [Eq. (12)-(13)].
    - Identify dominant path p* and remove common phase by multiplying by conj of
      the dominant column, then sum across paths to obtain sanitized Ĥ [Eq. (14)-(15)].
    Returns Ĥ(n, k) across kept subcarriers k for each time n.
    """
    # -----------------------------------------------------------------------------
    # Detailed variable explanation (created/used inside this function)
    #   H_kept            : (N_pkts, K) complex64
    #                       Raw CFR rows after pilot filtering; one row per packet.
    #   freq_vec_hz       : (K,) float64
    #                       Base-band frequency of each kept sub-carrier (Hz).
    #   delta_t           : float
    #                       Step of delay grid (seconds) used when building the
    #                       dictionary T.
    #   t_min, t_max      : float
    #                       Minimum and maximum delay bounds (seconds).
    #   subcarrier_stride : int
    #                       Use every `subcarrier_stride`-th sub-carrier to speed
    #                       the sparse Lasso solver.
    #   T_matrix          : (K, P_0) complex128
    #                       Delay dictionary with entries e^{-j2π f_k t_p}.
    #   select_sc         : (⌈K/stride⌉,) int
    #                       Indices of sub-carriers actually passed to the solver.
    #   row_T, col_T      : int
    #                       Number of selected rows and delay-grid columns.
    #   m, n              : int
    #                       Dimensions of the real-valued optimisation variable
    #                       space (m rows for identity, n = 2·col_T for variables).
    #   Im, In, On, Onm   : sparse identity / zero matrices building the OSQP
    #                       quadratic program (see optimisation_utility).
    #   P, q              : Quadratic and linear cost terms for OSQP.
    #   A2, A3            : Inequality constraint blocks enforcing |r| ≤ t (L1 norm).
    #   ones_n, zeros_n   : Convenience all-ones / zeros vectors for bounds.
    #   r                 : (P_0,) complex128 – sparse delay-domain coefficients
    #                       returned by the Lasso solver for one packet.
    #   p_star            : int – index of the dominant path (largest |r_p|).
    #   Tr                : (K, P_0) complex128 – per-path contribution matrix.
    #   ref_col           : (K,1) complex128 – dominant path column used as phase ref.
    #   Trr               : (K, P_0) complex128 – phase-aligned path matrix.
    #   H_sanitized       : (N_pkts, K) complex128 – final offset-free CFR.
    # -----------------------------------------------------------------------------
    T_matrix, time_matrix = build_T_matrix(freq_vec_hz, delta_t, t_min, t_max)
    select_sc = np.arange(0, len(freq_vec_hz), subcarrier_stride)

    # Number of rows actually fed to the solver equals the length of the
    # selected sub-carrier index vector. Using len(select_sc) avoids the
    # floor division issue when K is not a multiple of the stride.
    row_T = len(select_sc)
    col_T = T_matrix.shape[1]
    m = 2 * row_T
    n = 2 * col_T
    Im = scipy.sparse.eye(m)
    In = scipy.sparse.eye(n)
    On = scipy.sparse.csc_matrix((n, n))
    Onm = scipy.sparse.csc_matrix((n, m))
    P = scipy.sparse.block_diag([On, Im, On], format='csc')
    q = np.zeros(2 * n + m)
    A2 = scipy.sparse.hstack([In, Onm, -In])
    A3 = scipy.sparse.hstack([In, Onm, In])
    ones_n = np.ones(n)
    zeros_n = np.zeros(n)

    H_sanitized = np.zeros_like(H_kept, dtype=np.complex128)

    # -------------------------------------------------------------------------
    # Iterate over each packet (time index n_idx) and sanitize its CFR vector
    # -------------------------------------------------------------------------
    for n_idx in range(H_kept.shape[0]):
        h = H_kept[n_idx, :]  # (K,) complex – raw CFR for current packet
        # Solve sparse delay-domain inverse problem (Eq 8) to estimate complex
        # path coefficients r for this packet. The OSQP wrapper returns complex
        # r(p) for each delay-grid column.
        r = lasso_regression_osqp_fast(
            h, T_matrix, select_sc, row_T, col_T,
            Im, Onm, P, q, A2, A3, ones_n, zeros_n, np.zeros(n + m)
        )
        # Index of dominant (strongest magnitude) path – used as phase reference
        p_star = int(np.argmax(np.abs(r)))
        # Tr: F x L (F=subcarriers, L=delay grid), each column is path contribution across subcarriers
        Tr = T_matrix * r  # (K,P_0) – contribution of every path across sub-carriers
        # Reference dominant path column across subcarriers, shape F x 1
        # Form the dominant-path column to use for common phase cancellation
        ref_col = (T_matrix[:, p_star] * r[p_star]).reshape(-1, 1)
        # Element-wise multiply each column by conj(reference) to remove common phase, keep shape F x L
        # Remove common phase by multiplying each path column by conj(reference)
        Trr = Tr * np.conj(ref_col)
        # Sum over paths (columns) to get sanitized subcarrier response, shape F
        # Collapse all path columns to get sanitized CFR for this packet
        H_sanitized[n_idx, :] = np.sum(Trr, axis=1)

    return H_sanitized


def compute_doppler(H_sanitized, window_N=31, hop=1, N_D=100):
    """
    Compute Doppler power spectrum from sanitized CFR Ĥ using temporal FFT:
    - Form windows H_i by taking window_N consecutive packets [Eq. (19)].
    - Apply Hann window over time and compute FFT over time for each subcarrier:
      \mathcal{F}{H_i}(k,u) with zero-padding N_D [Eq. (21)].
    - Compute power |.|^2 and sum over subcarriers to get d_i(u) [Eq. (20)].
    - Return S: stack of d_i(u) across windows. Later, map u to velocity [Eq. (22)].
    """
    profiles = []
    for start in range(0, H_sanitized.shape[0] - window_N, hop):
        seg = H_sanitized[start:start + window_N, :]
        seg = np.nan_to_num(seg)
        w = hann(window_N)[:, None]
        seg_w = seg * w
        D = fft(seg_w, n=N_D, axis=0)
        D = fftshift(D, axes=0)
        P = np.abs(D) ** 2
        d_profile = np.sum(P, axis=1)
        profiles.append(d_profile)
    if not profiles:
        return np.zeros((0, N_D))
    S = np.asarray(profiles)
    S_max = np.max(S, axis=1, keepdims=True) + 1e-12
    S_norm = S / S_max
    return S_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=os.path.join(REPO_ROOT, 'input_files', 'empty.csv'))
    parser.add_argument('--outdir', default=os.path.join(REPO_ROOT, 'input_files', 'doppler_out'))
    parser.add_argument('--data_col', default='data')
    parser.add_argument('--ts_col', default='timestamp')
    parser.add_argument('--first_word_col', default='first_word')
    parser.add_argument('--tones_hint', type=int, default=56)
    parser.add_argument('--drop_pilots', action='store_true')
    parser.add_argument('--delta_f', type=float, default=312500.0)
    parser.add_argument('--delta_t', type=float, default=1e-7)
    parser.add_argument('--t_min', type=float, default=-3e-7)
    parser.add_argument('--t_max', type=float, default=5e-7)
    parser.add_argument('--stride_sc', type=int, default=2)
    parser.add_argument('--window', type=int, default=31)
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--fft', type=int, default=100)
    parser.add_argument('--fc', type=float, default=2.412e9)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    H_raw, ts = load_esp32_csv(
        args.input,
        data_col=args.data_col,
        ts_col=args.ts_col,
        first_word_col=args.first_word_col,
        tones_hint=args.tones_hint,
        drop_first_word=True,
    )
    if H_raw.shape[0] == 0:
        print('No CSI rows parsed from', args.input)
        return

    idx = ht20_indices(drop_pilots=args.drop_pilots)
    H = reorder_or_pad(H_raw, idx)

    freq_vec = build_freq_vector_from_indices(idx, delta_f=args.delta_f)

    H_sanit = sanitize_phase(
        H, freq_vec,
        delta_t=args.delta_t,
        t_min=args.t_min,
        t_max=args.t_max,
        subcarrier_stride=args.stride_sc,
    )

    S = compute_doppler(H_sanit, window_N=args.window, hop=args.hop, N_D=args.fft)

    # Estimate Tc and velocity bin size (Eq. 22)
    Tc = estimate_Tc_seconds(ts)
    c = 3e8
    delta_v = (c / (args.fc * Tc * args.fft)) if np.isfinite(Tc) and Tc > 0 else np.nan

    out_npz = os.path.join(args.outdir, 'doppler_spectrum.npz')
    np.savez_compressed(out_npz, S=S, idx=idx, fc=args.fc, delta_f=args.delta_f, Tc=Tc, delta_v=delta_v)
    print('Saved Doppler spectrum to', out_npz)
    if np.isfinite(Tc):
        print(f'Estimated Tc [s]: {Tc}   velocity bin [m/s]: {delta_v}')
    else:
        print('Timestamp not available; Tc and delta_v not saved.')

    # Velocity bin size (for plotting elsewhere):
    # v = u * c / (fc * T_c * N_D) – T_c should be computed from timestamps outside this script if available.


if __name__ == '__main__':
    main()
