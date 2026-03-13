import os
import argparse
import numpy as np
# This test mirrors the SHARP pipeline and now processes the full capture by default.
# Equation references (see apply_cfr_equations.py for detailed comments):
# - Load CFR/CSI (measured \bar H): CFR model and offsets
# - Build indices/frequencies: Eq. (6)
# - Phase sanitization via sparse model h = T r and Lasso: Eq. (5), (8), (9)-(11), (14)-(15)
# - Doppler (Hann+FFT over time, sum across subcarriers): Eq. (19)-(21)
# - Velocity mapping Δv (bin spacing): Eq. (22)
#
# Variable shapes across the pipeline:
#   H_raw: (N_pkts, tones_raw) complex64 — measured CFR \bar H_m(n)
#   idx: (K,) int — selected subcarrier indices (data tones kept)
#   H: (N_pkts, K) complex64 — CFR vector h(n) [Eq. (4)]
#   freq_vec: (K,) float64 — sub-carrier baseband frequencies f_k (Hz) [Eq. (6)]
#   H_sanit: (N_pkts, K) complex128 — sanitized CFR \hat H_k(n) [Eq. (15)]
#   S: (N_win, N_D) float64 — Doppler power spectrum d_i(u) [Eq. (20)]
#   ts: (N_pkts,) float — raw packet timestamps used to estimate T_c
#   delta_v: float — velocity bin size (m/s) derived from Eq. (22)
#
# Every major step below translates one or more theoretical equations into code:
#   1) load_esp32_csv → Eq. (\bar H): reads measured CFR including offsets.
#   2) ht20_indices + reorder_or_pad → Eq. (4) forms h(n) of length K.
#   3) build_freq_vector_from_indices → Eq. (6) builds f_k for dictionary T.
#   4) sanitize_phase → Eqs. (5), (8)-(15) removes hardware offsets.
#   5) compute_doppler → Eqs. (19)-(21) produces Doppler spectrum.
#   6) estimate_Tc_seconds + delta_v calc → Eq. (22) maps Doppler bins to velocity.
from apply_cfr_equations import (
    load_esp32_csv_lltf51_simple as load_esp32_csv,
    lltf51_indices as ht20_indices,
    reorder_or_pad,
    build_freq_vector_from_indices,
    sanitize_phase,
    compute_doppler,
    estimate_Tc_seconds,
    REPO_ROOT,
)
from plot_doppler import plot_doppler

INPUT = os.path.join(REPO_ROOT, 'input_files', 'empty.csv')

def finite_diffs(ts):
    """
    Estimate sampling period T_c from timestamps (auto-detect µs/ms/seconds).
    Used in Eq. (22): Δv = c / (f_c T_c N_D)
    """
    ts = np.asarray(ts, float)
    d = np.diff(ts)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return np.nan
    med = float(np.median(d))
    # Heuristics for units:
    # - If med in [1e3, 1e7], assume microseconds, convert to seconds
    # - If med in [1, 1e3), assume milliseconds, convert to seconds
    # - Else assume already seconds
    if 1e3 <= med < 1e7:
        return med / 1e6
    if 1 <= med < 1e3:
        return med / 1e3
    return med

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=INPUT)
    parser.add_argument('--drop_pilots', action='store_true')
    parser.add_argument('--sanitize', action='store_true', help='Run Lasso-based phase sanitization (slow on full capture)')
    parser.add_argument('--window', type=int, default=51)
    parser.add_argument('--hop', type=int, default=5)
    parser.add_argument('--fft', type=int, default=256)
    parser.add_argument('--fc', type=float, default=2.412e9)
    parser.add_argument('--outdir', default=os.path.join(REPO_ROOT, 'input_files', 'doppler_out'))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load measured CFR/CSI \bar H(t, k) from ESP32 CSV (CFR model & offsets)
    H_raw, ts = load_esp32_csv(args.input)
    print('Loaded H_raw:', H_raw.shape, 'timestamps:', ts.shape)

    # 2) Select HT20 data tones (drop pilots) and align shapes → h in Eq. (4)
    idx = ht20_indices()
    H = reorder_or_pad(H_raw, idx)
    print('Reindexed/padded H:', H.shape, 'target tones (51):', idx.size)

    # 3) Build subcarrier frequency vector f_k for dictionary T (Eq. 6)
    freq_vec = build_freq_vector_from_indices(idx)
    print('freq_vec Hz:', freq_vec.shape, 'range:', float(freq_vec.min()), float(freq_vec.max()))

    # 4) Phase sanitization (Eq. 5, 8, 9-11, 14-15) [optional on full capture]
    if args.sanitize:
        H_sanit = sanitize_phase(H, freq_vec)
    else:
        H_sanit = H
    print('Sanitized H (or raw if not sanitized):', H_sanit.shape, 'dtype:', H_sanit.dtype)

    # 5) Doppler spectrum via Hann+FFT over time, sum across subcarriers (Eq. 19-21)
    S = compute_doppler(H_sanit, window_N=args.window, hop=args.hop, N_D=args.fft)
    print('Doppler S:', S.shape, 'min/max:', float(S.min()) if S.size else np.nan, float(S.max()) if S.size else np.nan)

    # 6) Velocity bin Δv using Eq. (22); show raw median delta before unit normalization
    raw_med = float(np.median(np.diff(ts))) if ts.size > 1 else np.nan
    Tc = estimate_Tc_seconds(ts)
    if np.isfinite(Tc):
        c = 3e8
        fc = args.fc
        N_D = args.fft
        delta_v = c / (fc * Tc * N_D)
        print('Estimated Tc [s]:', Tc, 'velocity bin [m/s]:', delta_v, '(raw median delta:', raw_med, ')')
    else:
        print('No valid timestamps found to estimate Tc; velocity mapping skipped.')

    # 7) Save NPZ and plot Doppler with velocity axis
    out_npz = os.path.join(args.outdir, 'doppler_spectrum_test.npz')
    np.savez_compressed(out_npz, S=S, idx=idx, fc=fc, Tc=Tc, delta_v=delta_v)
    print('Saved NPZ to', out_npz)

    try:
        out_png = plot_doppler(out_npz)
        print('Saved Doppler plot to', out_png)
    except Exception as e:
        print('Plotting failed (install matplotlib?):', e)
