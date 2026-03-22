#!/usr/bin/env python3
"""
CSI viewer that calls collect_csi as a subprocess and visualizes amplitudes.
- Runs collect_csi.py as a subprocess to collect data
- Plots amplitudes of 52 subcarriers with real Hz axis
- Includes snapshot button for saving current plot
"""

import argparse
import sys
import time
import threading
import signal
import math
import subprocess
import numpy as np
from collections import deque, Counter
import os
import pickle

# PCA imports removed

# Ensure local utils.py (and ../train/utils.py as fallback) is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_this_dir, os.path.join(_this_dir, '..', 'train')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from utils import CSI_SUBCARRIER_MASK

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# Global variables
stop_event = threading.Event()
HEATMAP_COLS = 120          # Time-axis resolution of the heatmap image
SLIDING_WINDOW_SEC = 10.0   # Show last N seconds of data
MAX_LINE_PTS = 120          # Max points per subcarrier line
_data_buffer = deque(maxlen=15000)
_timestamps = deque(maxlen=15000)
_data_seq = 0              # monotonic counter — incremented on every append
_snapshot_counter = 0
_collection_done = False
_csi_process = None  # subprocess.Popen reference for cleanup
_occupation_model = None   # loaded occupation model dict
_occupation_window = 300   # occupation window size (overridden by model config)
_occupation_var_window = 100  # rolling variance window (overridden by model config)
_occupation_model_path = None  # path to the .pkl file for saving retrained models

# Reference sample state removed

# Outlier / impulse detection state
_outlier_z_threshold = 3.5     # z-score threshold for Hampel-style detection
_outlier_window = 30           # local median window (packets)
_impulse_rate_history = deque(maxlen=600)  # (timestamp, impulses_per_sec) pairs
_impulse_rate_ts = deque(maxlen=600)

# Preprocessing toggle state (for ablation view)
_preproc_flags = {
    'dc_removal': False,       # per-window DC (mean) removal
    'lowpass': False,           # low-pass filter (~5 Hz cutoff)
    'relative_csi': False,     # subcarrier-difference features
}

# Rolling mean buffer for temporal stability (longer horizon)
STABILITY_WINDOW_SEC = 60.0    # temporal stability lookback
STABILITY_ROLLING_WIN = 50     # rolling mean window in packets
STABILITY_HEATMAP_COLS = 200   # time bins for stability heatmap

# Subcarrier groups for sensitivity analysis
SC_GROUPS = {
    'low':  list(range(0, 17)),   # subcarriers 0-16
    'mid':  list(range(17, 35)),  # subcarriers 17-34
    'high': list(range(35, 52)),  # subcarriers 35-51
}
SC_GROUP_COLORS = {'low': '#42a5f5', 'mid': '#66bb6a', 'high': '#ef5350'}

# PCA state removed


def _update_pca_removed(data):
    """PCA functionality removed."""
    return None


def _rolling_variance(mag, var_window):
    """Compute rolling variance over a sliding window per subcarrier."""
    if var_window <= 1:
        return np.zeros_like(mag)
    n = mag.shape[0]
    cs = np.cumsum(mag, axis=0)
    cs2 = np.cumsum(mag ** 2, axis=0)
    cs = np.vstack([np.zeros((1, mag.shape[1])), cs])
    cs2 = np.vstack([np.zeros((1, mag.shape[1])), cs2])
    hi = np.arange(1, n + 1)
    lo = np.clip(hi - var_window, 0, None)
    counts = (hi - lo).reshape(-1, 1)
    means = (cs[hi] - cs[lo]) / counts
    mean_sq = (cs2[hi] - cs2[lo]) / counts
    return np.clip(mean_sq - means ** 2, 0, None)


def _rolling_mean(mag, win):
    """Compute rolling mean over a sliding window per subcarrier."""
    if win <= 1 or mag.shape[0] < 2:
        return mag.copy()
    n = mag.shape[0]
    cs = np.cumsum(mag, axis=0)
    cs = np.vstack([np.zeros((1, mag.shape[1])), cs])
    hi = np.arange(1, n + 1)
    lo = np.clip(hi - win, 0, None)
    counts = (hi - lo).reshape(-1, 1)
    return (cs[hi] - cs[lo]) / counts


def _detect_outliers(data_win, z_thresh=3.5, local_win=30):
    """Vectorized Hampel-style outlier detection per subcarrier.

    Uses block-wise median instead of per-sample rolling median for O(n)
    performance. Returns boolean mask and per-packet outlier count.
    """
    n, c = data_win.shape
    if n < local_win:
        return np.zeros_like(data_win, dtype=bool), np.zeros(n, dtype=int)
    # Block-wise approach: split into non-overlapping blocks, compute
    # median/MAD per block, then broadcast back — O(n) instead of O(n²)
    n_blocks = max(1, n // local_win)
    block_size = n // n_blocks
    trimmed = data_win[:n_blocks * block_size].reshape(n_blocks, block_size, c)
    block_med = np.median(trimmed, axis=1)        # (n_blocks, c)
    block_mad = np.median(np.abs(trimmed - block_med[:, None, :]), axis=1) * 1.4826
    block_mad = np.where(block_mad < 1e-6, 1.0, block_mad)
    # Assign each sample to its block
    block_idx = np.clip(np.arange(n) // block_size, 0, n_blocks - 1)
    med_expanded = block_med[block_idx]           # (n, c)
    mad_expanded = block_mad[block_idx]           # (n, c)
    z_scores = np.abs(data_win - med_expanded) / mad_expanded
    outlier_mask = z_scores > z_thresh
    per_pkt_count = outlier_mask.sum(axis=1)
    return outlier_mask, per_pkt_count


def _apply_preproc(data, flags):
    """Apply preprocessing toggles to data array in-place (returns copy).

    flags: dict with keys 'dc_removal', 'lowpass', 'relative_csi'.
    """
    out = data.copy()
    if flags.get('dc_removal'):
        out = out - out.mean(axis=0, keepdims=True)
    if flags.get('relative_csi') and out.shape[1] > 1:
        out = np.diff(out, axis=1)
    if flags.get('lowpass'):
        try:
            from scipy.ndimage import uniform_filter1d
            out = uniform_filter1d(out, size=7, axis=0)
        except ImportError:
            # Fallback: simple moving average
            kernel = 7
            if out.shape[0] >= kernel:
                cs = np.cumsum(out, axis=0)
                cs = np.vstack([np.zeros((1, out.shape[1])), cs])
                out = (cs[kernel:] - cs[:-kernel]) / kernel
    return out


def _save_reference_removed(name, data, ts, meta=None):
    """Reference functionality removed."""
    pass


def _load_reference_removed(fpath):
    """Reference functionality removed."""
    return None


def _resample_to_sr(raw_data, raw_ts, target_sr):
    """Bin-average resample raw CSI data to target_sr Hz.

    Mirrors the CSI_Loader._resample_equal_intervals logic so that
    calibration data and live inference data match the training pipeline.

    Parameters
    ----------
    raw_data : np.ndarray, shape (N, C)
    raw_ts : np.ndarray, shape (N,) — timestamps in seconds
    target_sr : int — target sample rate in Hz

    Returns
    -------
    np.ndarray, shape (M, C) — resampled data at target_sr Hz
    """
    if len(raw_data) < 2:
        return raw_data.copy()
    t0 = raw_ts[0]
    n_out = int((raw_ts[-1] - t0) * target_sr) + 1
    if n_out < 2:
        return raw_data.copy()
    ncols = raw_data.shape[1]
    target_t = t0 + np.arange(n_out) / target_sr
    dt = 1.0 / target_sr
    edges = np.concatenate([target_t - dt / 2, [target_t[-1] + dt / 2]])
    bin_assign = np.clip(np.searchsorted(edges, raw_ts) - 1, 0, n_out - 1)
    resampled = np.zeros((n_out, ncols), dtype=np.float64)
    np.add.at(resampled, bin_assign, raw_data)
    bin_counts = np.bincount(bin_assign, minlength=n_out).astype(np.float64)
    populated = bin_counts > 0
    resampled[populated] /= bin_counts[populated, None]
    for b in np.where(~populated)[0]:
        if b > 0:
            resampled[b] = resampled[b - 1]
    return resampled


def _retrain_occupation_inprocess(cal_buffer, var_window, window, labels=None):
    """Retrain occupation model from calibration buffer (in-process).

    Parameters
    ----------
    cal_buffer : list of (label_str, mag_array) tuples
        Each mag_array is already resampled to training SR.
    var_window : int — rolling variance window
    window : int — segment window size
    labels : list[str] or None — ordered class names; inferred if None

    Returns
    -------
    dict — model dict compatible with _occupation_model format, or None on failure
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("[cal] ERROR: scikit-learn not installed, cannot retrain")
        return None

    if not cal_buffer:
        print("[cal] No calibration data collected")
        return None

    # Discover labels
    if labels is None:
        labels = sorted(set(lbl for lbl, _ in cal_buffer))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    all_X, all_y = [], []
    for lbl, mag in cal_buffer:
        if lbl not in label_to_idx:
            continue
        # Apply rolling variance
        if var_window > 1:
            mag_v = _rolling_variance(mag, var_window)
        else:
            mag_v = mag
        # Non-overlapping windows
        n_win = mag_v.shape[0] // window
        for wi in range(n_win):
            chunk = mag_v[wi * window : (wi + 1) * window, :]
            all_X.append(chunk.ravel())
            all_y.append(label_to_idx[lbl])

    if len(all_X) < 2:
        print(f"[cal] Not enough windows ({len(all_X)}), need at least 2")
        return None

    X = np.array(all_X, dtype=np.float64)
    y = np.array(all_y, dtype=np.int64)

    # Check we have at least 2 classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        present = [labels[i] for i in unique_classes]
        print(f"[cal] Only 1 class present ({present}), need at least 2 to train")
        return None

    print(f"[cal] Training on {X.shape[0]} windows, {X.shape[1]} features, "
          f"{len(unique_classes)} classes")
    for lbl in labels:
        idx = label_to_idx[lbl]
        print(f"[cal]   {lbl}: {(y == idx).sum()} windows")

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    # Quick self-accuracy
    acc = float((clf.predict(X) == y).mean())
    print(f"[cal] Train accuracy: {acc:.4f}")

    model_dict = {
        'classifier': clf,
        'labels': labels,
        'var_window': var_window,
        'window': window,
        'sr': _cal_resample_sr,
        'n_subcarriers': 52,
        'n_features': X.shape[1],
        'feat_mean': X.mean(axis=0),
        'feat_std': np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0)),
        'accuracy': acc,
        'calibration': True,
    }
    return model_dict


_conf_threshold = 0.40  # min softmax prob to accept prediction
_smooth_window = 5      # majority-vote over last N predictions


# Video recording state (PIL-based, no cv2 dependency)
_video_frames = []       # list of PIL.Image frames
_video_recording = False
_video_filename = None

# CSI parameters (adjust based on your setup)
NUM_SUBCARRIERS = 52
SUBCARRIER_SPACING = 312.5e3  # 312.5 kHz spacing for WiFi
CENTER_FREQ = 2.437e9  # 2.437 GHz (WiFi channel 6)
FFT_SIZE = 64  # Typical for ESP32 CSI

def get_subcarrier_frequencies():
    """Calculate actual frequencies for each subcarrier."""
    # For WiFi, subcarriers are numbered from -28 to +28, with DC at 0
    # We typically use 52 subcarriers (excluding some edge subcarriers)
    subcarrier_indices = np.arange(-26, 26)  # -26 to +25 = 52 subcarriers
    frequencies = CENTER_FREQ + subcarrier_indices * SUBCARRIER_SPACING
    return frequencies

def parse_csi_line(line):
    """Parse a single CSI line from collect_csi output.
    
    Computes amplitude for all 64 subcarriers, then applies
    CSI_SUBCARRIER_MASK to select the 52 valid LLTF subcarriers.
    """
    try:
        parts = line.split(',', 14)
        if len(parts) < 15:
            return None
            
        rssi = int(parts[3])
        data_str = parts[14]
        
        if not data_str:
            return None
            
        nums = data_str.strip('"[] ').split(',')
        amps = []
        
        for j in range(0, len(nums) - 1, 2):
            try:
                im_v = float(nums[j])      # even index = imaginary (ESP32 format)
                re_v = float(nums[j + 1])   # odd index  = real
                amp = math.sqrt(re_v * re_v + im_v * im_v)
                amps.append(amp)
            except (ValueError, IndexError):
                continue
        
        # We need all 64 subcarriers to apply the mask
        if len(amps) == 64:
            all_amps = np.array(amps)
            return all_amps[CSI_SUBCARRIER_MASK]   # -> 52 valid subcarriers
    except Exception:
        pass
    return None

def collect_csi_subprocess(rx_port, tx_port, baud, duration):
    """Run collect_csi.py as subprocess and tail its CSV in real-time."""
    global _data_buffer, _timestamps, _data_seq
    
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    base_name = f"temp_csi_{unique_id}"
    actual_csv_path = f"{base_name}_1.csv"
    
    # Build command
    cmd = [
        sys.executable, 'collect_csi.py',
        '--rx-port', rx_port,
        '--duration', str(duration),
        '--times', '1',
        '--out', base_name
    ]
    
    if tx_port:
        cmd.extend(['--tx-port', tx_port])
    
    print(f"[info] Starting collect_csi subprocess: {' '.join(cmd)}")
    
    try:
        global _csi_process
        # Start subprocess (don't capture stdout so its logs print directly)
        process = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
        )
        _csi_process = process
        
        start_time = time.time()
        
        # Wait for the CSV file to appear
        while not os.path.exists(actual_csv_path) and not stop_event.is_set():
            if process.poll() is not None:  # subprocess exited early
                break
            time.sleep(0.05)
        
        if not os.path.exists(actual_csv_path):
            print(f"[error] CSV file never appeared: {actual_csv_path}")
            return
        
        print(f"[info] Tailing CSV in real-time: {actual_csv_path}")
        line_count = 0
        header_skipped = False
        
        # Tail the file while the subprocess is still writing
        with open(actual_csv_path, 'r') as f:
            while not stop_event.is_set():
                line = f.readline()
                if line:
                    line = line.strip()
                    if not header_skipped:
                        header_skipped = True
                        continue  # skip CSV header
                    if not line:
                        continue
                    
                    amps = parse_csi_line(line)
                    if amps is not None:
                        current_time = time.time() - start_time
                        _data_buffer.append(amps)
                        _timestamps.append(current_time)
                        _data_seq += 1
                        line_count += 1
                else:
                    # No new data yet — check if subprocess is still alive
                    if process.poll() is not None:
                        # Drain any remaining lines
                        for remaining in f:
                            remaining = remaining.strip()
                            if not remaining:
                                continue
                            amps = parse_csi_line(remaining)
                            if amps is not None:
                                current_time = time.time() - start_time
                                _data_buffer.append(amps)
                                _timestamps.append(current_time)
                                _data_seq += 1
                                line_count += 1
                        break
                    time.sleep(0.01)  # brief sleep before retrying
        
        return_code = process.wait()
        print(f"\n[info] Subprocess exited (rc={return_code}), parsed {line_count} CSI lines")
        
        # Clean up temporary file
        try:
            os.remove(actual_csv_path)
            print(f"[info] Cleaned up {actual_csv_path}")
        except OSError:
            pass
        
    except Exception as e:
        print(f"[error] Failed to run collect_csi: {e}")
        import traceback
        traceback.print_exc()
        stop_event.set()

# Fixed amplitude range
AMP_MIN = 0
AMP_MAX = 50


# Publication font sizes
FONT_TITLE   = 16
FONT_LABEL   = 13
FONT_TICK    = 11
FONT_LEGEND  = 11
FONT_STATS   = 12
FONT_BTN     = 9
FONT_SUPTITLE = 18

# Active view: 'timeseries' or 'analytics'
_active_view = 'timeseries'


_saved_ax_positions = {}  # {id(ax): Bbox} — original positions for off-screen trick

def _set_axes_visible(axes_list, visible):
    """Show/hide a list of axes, moving button axes off-screen when hidden
    to prevent matplotlib's grab_mouse conflict on overlapping invisible axes."""
    from matplotlib.transforms import Bbox
    for ax in axes_list:
        ax.set_visible(visible)
        ax_id = id(ax)
        if visible:
            # Restore original position if we moved it off-screen
            if ax_id in _saved_ax_positions:
                ax.set_position(_saved_ax_positions.pop(ax_id))
        else:
            # Save position and move off-screen to prevent mouse grabs
            if ax_id not in _saved_ax_positions:
                _saved_ax_positions[ax_id] = ax.get_position()
            ax.set_position(Bbox([[9, 9], [9.01, 9.01]]))


def _style_ax(ax):
    """Apply publication spine styling to an axis."""
    for spine in ax.spines.values():
        spine.set_color('#455a64')
        spine.set_linewidth(0.8)


def create_visualization():
    """Create three switchable view groups with publication styling."""
    global _snapshot_counter

    frequencies = get_subcarrier_frequencies()
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('ThothCraft  \u2014  CSI Real-Time Monitor',
                 fontsize=FONT_SUPTITLE, fontweight='bold', color='#00e5ff')

    # Shared region for all views
    L, R, T, B = 0.06, 0.94, 0.90, 0.10

    # Stats bar (always visible, absolute position)
    ax_info = fig.add_axes([L, 0.04, R - L, 0.035])
    ax_info.axis('off')
    stats_text = ax_info.text(
        0.5, 0.5, '', transform=ax_info.transAxes,
        ha='center', va='center', fontsize=FONT_STATS, color='#b0bec5',
        family='monospace',
    )

    # =================================================================
    # VIEW 1: Time Series
    # =================================================================
    gs1 = fig.add_gridspec(3, 2, height_ratios=[3.0, 3.0, 2.5],
                           width_ratios=[1, 0.018],
                           left=L, right=R, top=T, bottom=B,
                           hspace=0.38, wspace=0.03)
    ax_heat  = fig.add_subplot(gs1[0, 0])
    ax_cbar1 = fig.add_subplot(gs1[0, 1])
    ax_lines = fig.add_subplot(gs1[1, 0])
    ax_mean  = fig.add_subplot(gs1[2, 0])

    blank = np.zeros((NUM_SUBCARRIERS, HEATMAP_COLS))
    im = ax_heat.imshow(blank, aspect='auto', origin='lower', cmap='inferno',
                        extent=[0, SLIDING_WINDOW_SEC, 0, NUM_SUBCARRIERS],
                        interpolation='bilinear', vmin=AMP_MIN, vmax=AMP_MAX)
    ax_heat.set_ylabel('Subcarrier Index', fontsize=FONT_LABEL, color='#b0bec5')
    ax_heat.set_title('Amplitude Heatmap (0\u201350)', fontsize=FONT_TITLE,
                      color='#e0e0e0', fontweight='semibold', pad=8)
    ax_heat.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    ax_heat.set_xlim(0, SLIDING_WINDOW_SEC)
    _style_ax(ax_heat)
    cbar1 = fig.colorbar(im, cax=ax_cbar1)
    cbar1.set_label('Amplitude', fontsize=FONT_LABEL - 2, color='#b0bec5')
    cbar1.ax.tick_params(colors='#90a4ae', labelsize=FONT_TICK - 2)

    sc_lines = []
    sc_colors = plt.cm.turbo(np.linspace(0.05, 0.95, NUM_SUBCARRIERS))
    for i in range(NUM_SUBCARRIERS):
        ln, = ax_lines.plot([], [], color=sc_colors[i], linewidth=0.6, alpha=0.7)
        sc_lines.append(ln)
    ax_lines.set_ylabel('Amplitude', fontsize=FONT_LABEL, color='#b0bec5')
    ax_lines.set_ylim(AMP_MIN, AMP_MAX)
    ax_lines.set_xlim(0, SLIDING_WINDOW_SEC)
    ax_lines.set_title('52 Subcarrier Lines', fontsize=FONT_TITLE,
                       color='#e0e0e0', fontweight='semibold', pad=8)
    ax_lines.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    ax_lines.grid(True, alpha=0.12, color='#455a64', linewidth=0.5)
    _style_ax(ax_lines)

    mean_line, = ax_mean.plot([], [], color='#00e5ff', linewidth=2.0, label='Mean')
    std_hi_line, = ax_mean.plot([], [], color='#00e5ff', linewidth=0.7, alpha=0.35)
    std_lo_line, = ax_mean.plot([], [], color='#00e5ff', linewidth=0.7, alpha=0.35)
    ax_mean.set_ylabel('Mean Amplitude', fontsize=FONT_LABEL, color='#b0bec5')
    ax_mean.set_xlabel('Time (s)', fontsize=FONT_LABEL, color='#b0bec5')
    ax_mean.set_xlim(0, SLIDING_WINDOW_SEC)
    ax_mean.set_ylim(AMP_MIN, AMP_MAX)
    ax_mean.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    ax_mean.grid(True, alpha=0.12, color='#455a64', linewidth=0.5)
    ax_mean.legend(loc='upper right', fontsize=FONT_LEGEND, framealpha=0.3)
    ax_mean.set_title('Mean \u00b1 Std Amplitude', fontsize=FONT_TITLE,
                      color='#e0e0e0', fontweight='semibold', pad=8)
    _style_ax(ax_mean)

    v1_axes = [ax_heat, ax_cbar1, ax_lines, ax_mean]

    # =================================================================
    # VIEW 2: Analytics (simplified)
    # =================================================================
    gs2 = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.5],
                           width_ratios=[1.1, 0.9],
                           left=L, right=R, top=T, bottom=B,
                           hspace=0.40, wspace=0.18)
    ax_timeline = fig.add_subplot(gs2[0, :])   # activity timeline (full width)
    ax_stats = fig.add_subplot(gs2[1, 0])     # basic statistics
    ax_ctrl = fig.add_subplot(gs2[1, 1])      # control panel

    # ── Activity timeline (simplified) ──
    ax_timeline.set_xlim(0, SLIDING_WINDOW_SEC)
    ax_timeline.set_ylim(-0.5, 0.5)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel('Time (s)', fontsize=FONT_LABEL - 1, color='#78909c')
    ax_timeline.set_title('Signal Activity Timeline', fontsize=FONT_TITLE,
                          color='#e0e0e0', fontweight='semibold', pad=6)
    ax_timeline.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    ax_timeline.set_facecolor('#0d1117')
    _style_ax(ax_timeline)
    # Scatter will be updated each frame; create an empty one
    timeline_scatter = ax_timeline.scatter([], [], s=60, c='#00e5ff', marker='o',
                                           edgecolors='none', alpha=0.85, zorder=5)

    # ── Basic statistics ──
    ax_stats.set_title('Signal Statistics', fontsize=FONT_TITLE,
                       color='#e0e0e0', fontweight='semibold', pad=8)
    ax_stats.set_xlabel('Metric', fontsize=FONT_LABEL, color='#b0bec5')
    ax_stats.set_ylabel('Value', fontsize=FONT_LABEL, color='#b0bec5')
    ax_stats.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    ax_stats.grid(True, alpha=0.12, color='#455a64', linewidth=0.5)
    _style_ax(ax_stats)
    stats_bars = ax_stats.bar([], [], color=['#00e5ff', '#66bb6a', '#ff6b6b'], alpha=0.8)
    stats_info_text = ax_stats.text(0.02, 0.95, '', transform=ax_stats.transAxes,
                                   fontsize=FONT_TICK - 1, color='#80cbc4', va='top', family='monospace',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', alpha=0.7))

    # ── Control panel ──
    ax_ctrl.axis('off')
    ax_ctrl.set_facecolor('#0d1117')
    for sp in ax_ctrl.spines.values():
        sp.set_color('#1e2a38'); sp.set_linewidth(1)

    ax_ctrl.text(0.02, 0.92, 'Analytics Controls:', transform=ax_ctrl.transAxes,
                  fontsize=FONT_TICK + 1, color='#58a6ff', va='top', fontweight='bold')

    analytics_status_text = ax_ctrl.text(
        0.5, 0.5, 'Analytics ready...', transform=ax_ctrl.transAxes,
        fontsize=FONT_TICK + 2, color='#66bb6a', va='center', ha='center',
        fontweight='bold', family='monospace')

    v2_axes = [ax_timeline, ax_stats, ax_ctrl]

    # =================================================================
    # VIEW 3: Signal Quality
    # =================================================================
    gs3 = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1, 1],
                           left=L, right=R, top=T, bottom=B,
                           hspace=0.38, wspace=0.22)
    ax_profile = fig.add_subplot(gs3[0, 0])
    ax_hist    = fig.add_subplot(gs3[0, 1])
    ax_varheat = fig.add_subplot(gs3[1, :])

    profile_bars = ax_profile.bar(
        range(NUM_SUBCARRIERS), [0]*NUM_SUBCARRIERS,
        color=sc_colors, edgecolor='none', width=0.8)
    ax_profile.set_xlim(-0.5, NUM_SUBCARRIERS - 0.5)
    ax_profile.set_ylim(AMP_MIN, AMP_MAX)
    ax_profile.set_xlabel('Subcarrier Index', fontsize=FONT_LABEL, color='#b0bec5')
    ax_profile.set_ylabel('Mean Amplitude', fontsize=FONT_LABEL, color='#b0bec5')
    ax_profile.set_title('Per-Subcarrier Amplitude Profile', fontsize=FONT_TITLE,
                          color='#e0e0e0', fontweight='semibold', pad=8)
    ax_profile.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    ax_profile.grid(True, axis='y', alpha=0.12, color='#455a64', linewidth=0.5)
    _style_ax(ax_profile)

    hist_n, hist_bins, hist_patches = ax_hist.hist(
        [0], bins=40, range=(AMP_MIN, AMP_MAX),
        color='#00e5ff', alpha=0.7, edgecolor='#263238', linewidth=0.5)
    ax_hist.set_xlabel('Amplitude', fontsize=FONT_LABEL, color='#b0bec5')
    ax_hist.set_ylabel('Count', fontsize=FONT_LABEL, color='#b0bec5')
    ax_hist.set_title('Amplitude Distribution', fontsize=FONT_TITLE,
                       color='#e0e0e0', fontweight='semibold', pad=8)
    ax_hist.set_xlim(AMP_MIN, AMP_MAX)
    ax_hist.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    ax_hist.grid(True, axis='y', alpha=0.12, color='#455a64', linewidth=0.5)
    _style_ax(ax_hist)
    hist_stats_text = ax_hist.text(0.97, 0.95, '', transform=ax_hist.transAxes,
                                    fontsize=FONT_TICK, color='#80cbc4',
                                    va='top', ha='right', family='monospace')

    var_blank = np.zeros((NUM_SUBCARRIERS, HEATMAP_COLS))
    var_im = ax_varheat.imshow(var_blank, aspect='auto', origin='lower',
                                cmap='magma', interpolation='bilinear',
                                vmin=0, vmax=25)
    ax_varheat.set_ylabel('Subcarrier Index', fontsize=FONT_LABEL, color='#b0bec5')
    ax_varheat.set_xlabel('Time (s)', fontsize=FONT_LABEL, color='#b0bec5')
    ax_varheat.set_title('Rolling Variance Heatmap (motion detector)',
                          fontsize=FONT_TITLE, color='#e0e0e0',
                          fontweight='semibold', pad=8)
    ax_varheat.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    _style_ax(ax_varheat)

    v3_axes = [ax_profile, ax_hist, ax_varheat]

    # =================================================================
    # VIEW 4: Diagnostics
    # =================================================================
    gs4 = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 0.8],
                           width_ratios=[1, 1],
                           left=L, right=R, top=T, bottom=B,
                           hspace=0.42, wspace=0.22)
    ax_temporal  = fig.add_subplot(gs4[0, 0])   # Temporal stability heatmap
    ax_scgroups  = fig.add_subplot(gs4[0, 1])   # Subcarrier group sensitivity
    ax_outliers  = fig.add_subplot(gs4[1, 0])   # Outlier impulse rate
    ax_diag_ctrl = fig.add_subplot(gs4[1, 1])   # Diagnostics control panel

    # ── Temporal stability heatmap (rolling mean per subcarrier) ──
    stab_blank = np.zeros((NUM_SUBCARRIERS, STABILITY_HEATMAP_COLS))
    stab_im = ax_temporal.imshow(stab_blank, aspect='auto', origin='lower',
                                  cmap='viridis', interpolation='bilinear',
                                  vmin=AMP_MIN, vmax=AMP_MAX)
    ax_temporal.set_ylabel('Subcarrier', fontsize=FONT_LABEL, color='#b0bec5')
    ax_temporal.set_xlabel('Time (s)', fontsize=FONT_LABEL - 1, color='#78909c')
    ax_temporal.set_title('Temporal Stability (Rolling Mean)',
                           fontsize=FONT_TITLE, color='#e0e0e0',
                           fontweight='semibold', pad=8)
    ax_temporal.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    _style_ax(ax_temporal)
    stab_info_text = ax_temporal.text(
        0.02, 0.95, '', transform=ax_temporal.transAxes,
        fontsize=FONT_TICK - 1, color='#80cbc4', va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', alpha=0.7))

    # ── Subcarrier group sensitivity (variance per group over time) ──
    sc_group_lines = {}
    for gname, gcolor in SC_GROUP_COLORS.items():
        ln, = ax_scgroups.plot([], [], color=gcolor, linewidth=1.8,
                                label=gname, alpha=0.85)
        sc_group_lines[gname] = ln
    ax_scgroups.set_ylabel('Group Variance', fontsize=FONT_LABEL, color='#b0bec5')
    ax_scgroups.set_xlabel('Time (s)', fontsize=FONT_LABEL - 1, color='#78909c')
    ax_scgroups.set_title('Subcarrier Group Sensitivity',
                           fontsize=FONT_TITLE, color='#e0e0e0',
                           fontweight='semibold', pad=8)
    ax_scgroups.set_ylim(0, 30)
    ax_scgroups.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    ax_scgroups.grid(True, alpha=0.12, color='#455a64', linewidth=0.5)
    ax_scgroups.legend(loc='upper right', fontsize=FONT_LEGEND - 1, framealpha=0.4)
    _style_ax(ax_scgroups)
    scg_info_text = ax_scgroups.text(
        0.02, 0.95, '', transform=ax_scgroups.transAxes,
        fontsize=FONT_TICK - 1, color='#ffab40', va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', alpha=0.7))

    # ── Outlier / impulse rate timeline ──
    outlier_line, = ax_outliers.plot([], [], color='#ef5350', linewidth=1.5,
                                      label='Impulses/s')
    outlier_fill = None  # will be updated dynamically
    ax_outliers.set_ylabel('Impulses / sec', fontsize=FONT_LABEL, color='#b0bec5')
    ax_outliers.set_xlabel('Time (s)', fontsize=FONT_LABEL - 1, color='#78909c')
    ax_outliers.set_title(f'Outlier Detection (z>{_outlier_z_threshold:.1f})',
                           fontsize=FONT_TITLE, color='#e0e0e0',
                           fontweight='semibold', pad=8)
    ax_outliers.set_ylim(0, 20)
    ax_outliers.tick_params(colors='#90a4ae', labelsize=FONT_TICK)
    ax_outliers.grid(True, alpha=0.12, color='#455a64', linewidth=0.5)
    _style_ax(ax_outliers)
    outlier_stats_text = ax_outliers.text(
        0.98, 0.95, '', transform=ax_outliers.transAxes,
        fontsize=FONT_TICK, color='#ef9a9a', va='top', ha='right',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', alpha=0.7))


    # ── Diagnostics control panel ──
    ax_diag_ctrl.axis('off')
    ax_diag_ctrl.set_facecolor('#0d1117')
    for sp in ax_diag_ctrl.spines.values():
        sp.set_color('#1e2a38'); sp.set_linewidth(1)

    # Preprocessing toggle controls
    ax_diag_ctrl.text(0.01, 0.92, 'Preprocess:', transform=ax_diag_ctrl.transAxes,
                      fontsize=FONT_TICK + 1, color='#ffab40', va='top',
                      fontweight='bold')

    dbtn_w = 0.07 * 0.8
    dbtn_h = 0.38 * 0.8
    dbtn_y = 0.08

    pp_names = ['DC Remove', 'Low-Pass', 'Relative']
    pp_keys  = ['dc_removal', 'lowpass', 'relative_csi']
    pp_btn_axes = []
    pp_btns = []
    for pi, (pp_name, pp_key) in enumerate(zip(pp_names, pp_keys)):
        px = 0.01 + pi * (dbtn_w * 1.25 + 0.003)
        ax_pp = fig.add_axes([px, dbtn_y, dbtn_w * 1.2, dbtn_h])
        b = Button(ax_pp, pp_name, color='#1a2332', hovercolor='#2a4060')
        b.label.set_color('#586069')
        b.label.set_fontsize(FONT_BTN)
        pp_btn_axes.append(ax_pp)
        pp_btns.append(b)

    # Diagnostics status text
    diag_status_text = ax_diag_ctrl.text(
        0.75, 0.5, '', transform=ax_diag_ctrl.transAxes,
        fontsize=FONT_TICK + 1, color='#66bb6a', va='center', ha='center',
        fontweight='bold', family='monospace')

    v4_axes = [ax_temporal, ax_scgroups, ax_outliers, ax_diag_ctrl, *pp_btn_axes]

    # Start with only V1 visible
    _set_axes_visible(v2_axes, False)
    _set_axes_visible(v3_axes, False)
    _set_axes_visible(v4_axes, False)

    # =================================================================
    # Buttons & Controls — styled toolbar
    # =================================================================
    _tab_color   = '#1a2332'
    _tab_active  = '#263d52'
    _tab_hover   = '#2a4060'
    _ctrl_color  = '#1e2a38'
    _ctrl_hover  = '#2a3a4e'
    tab_style  = dict(color=_tab_color, hovercolor=_tab_hover)
    ctrl_style = dict(color=_ctrl_color, hovercolor=_ctrl_hover)

    # Toolbar background strip
    ax_toolbar_bg = fig.add_axes([0.0, 0.0, 1.0, 0.038])
    ax_toolbar_bg.set_facecolor('#0d1117')
    ax_toolbar_bg.set_xticks([]); ax_toolbar_bg.set_yticks([])
    for sp in ax_toolbar_bg.spines.values():
        sp.set_visible(False)

    # ── View tabs (left group) ──
    _tab_w, _tab_h, _tab_y = 0.075, 0.026, 0.005
    _tab_gap = 0.002

    ax_b1 = fig.add_axes([0.015, _tab_y, _tab_w, _tab_h])
    btn_v1 = Button(ax_b1, '▸ Signals', color=_tab_active, hovercolor=_tab_hover)
    btn_v1.label.set_color('#ffab40'); btn_v1.label.set_fontsize(FONT_BTN)
    btn_v1.label.set_fontweight('bold')

    ax_b2 = fig.add_axes([0.015 + _tab_w + _tab_gap, _tab_y, _tab_w, _tab_h])
    btn_v2 = Button(ax_b2, '▸ Analytics', **tab_style)
    btn_v2.label.set_color('#58a6ff'); btn_v2.label.set_fontsize(FONT_BTN)

    ax_b3 = fig.add_axes([0.015 + 2*(_tab_w + _tab_gap), _tab_y, _tab_w, _tab_h])
    btn_v3 = Button(ax_b3, '▸ Quality', **tab_style)
    btn_v3.label.set_color('#b388ff'); btn_v3.label.set_fontsize(FONT_BTN)

    ax_b4 = fig.add_axes([0.015 + 3*(_tab_w + _tab_gap), _tab_y, _tab_w + 0.01, _tab_h])
    btn_v4 = Button(ax_b4, '▸ Diagnostics', **tab_style)
    btn_v4.label.set_color('#ce93d8'); btn_v4.label.set_fontsize(FONT_BTN)

    # Separator
    fig.text(0.34, 0.018, '│', fontsize=12, color='#30363d', va='center')

    # ── Actions (middle group) ──
    _act_x0 = 0.355
    ax_bs = fig.add_axes([_act_x0, _tab_y, 0.065, _tab_h])
    btn_snap = Button(ax_bs, '[*] Snap', **ctrl_style)
    btn_snap.label.set_color('#8b949e'); btn_snap.label.set_fontsize(FONT_BTN)

    ax_br = fig.add_axes([_act_x0 + 0.068, _tab_y, 0.055, _tab_h])
    btn_rec_toggle = Button(ax_br, '● Rec', **ctrl_style)
    btn_rec_toggle.label.set_color('#4caf50'); btn_rec_toggle.label.set_fontsize(FONT_BTN)

    rec_text = fig.text(_act_x0 + 0.13, 0.018, '', fontsize=FONT_BTN,
                        color='#f44336', fontweight='bold', family='monospace')

    # Separator
    fig.text(0.505, 0.018, '│', fontsize=12, color='#30363d', va='center')

    # ── Window length (right-middle group) ──
    fig.text(0.52, 0.018, 'Window:', fontsize=7, color='#586069', va='center')
    _win_options = [5.0, 10.0, 20.0, 30.0, 60.0]
    _win_btn_axes = []
    _win_btns = []
    for wi, wval in enumerate(_win_options):
        ax_w = fig.add_axes([0.56 + wi * 0.038, _tab_y, 0.035, _tab_h])
        lbl = f'{int(wval)}s'
        b = Button(ax_w, lbl, **ctrl_style)
        b.label.set_fontsize(FONT_BTN - 1)
        b.label.set_color('#ffab40' if wval == SLIDING_WINDOW_SEC else '#8b949e')
        _win_btn_axes.append(ax_w)
        _win_btns.append(b)

    # Separator
    fig.text(0.755, 0.018, '│', fontsize=12, color='#30363d', va='center')

    # Label state for calibration data collection
    _label_state = {
        'labels': [],
        'current_label': None,
        'current_color': None,
    }

    # =================================================================
    # View toggle callbacks
    # =================================================================
    all_view_groups = [v1_axes, v2_axes, v3_axes, v4_axes]
    view_names = ['timeseries', 'analytics', 'sigquality', 'diagnostics']
    _tab_btns = [btn_v1, btn_v2, btn_v3, btn_v4]
    _tab_btn_axes = [ax_b1, ax_b2, ax_b3, ax_b4]
    _tab_colors = ['#ffab40', '#58a6ff', '#b388ff', '#ce93d8']

    def _switch_view(idx):
        def _cb(event):
            global _active_view
            _active_view = view_names[idx]
            for i, grp in enumerate(all_view_groups):
                _set_axes_visible(grp, i == idx)
            # Style active/inactive tabs
            for i, (btn, bax) in enumerate(zip(_tab_btns, _tab_btn_axes)):
                if i == idx:
                    bax.set_facecolor(_tab_active)
                    btn.label.set_fontweight('bold')
                    btn.label.set_color(_tab_colors[i])
                else:
                    bax.set_facecolor(_tab_color)
                    btn.label.set_fontweight('normal')
                    btn.label.set_color('#586069')
            ax_info.set_visible(True)
            ax_toolbar_bg.set_visible(True)
            fig.canvas.draw_idle()
        return _cb

    btn_v1.on_clicked(_switch_view(0))
    btn_v2.on_clicked(_switch_view(1))
    btn_v3.on_clicked(_switch_view(2))
    btn_v4.on_clicked(_switch_view(3))
    # Set initial inactive styling for tabs 2, 3, 4
    btn_v2.label.set_color('#586069')
    btn_v3.label.set_color('#586069')
    btn_v4.label.set_color('#586069')

    # Window length callbacks
    def _make_win_cb(wval, wi):
        def _cb(event):
            global SLIDING_WINDOW_SEC
            SLIDING_WINDOW_SEC = wval
            for j, b in enumerate(_win_btns):
                b.label.set_color('#ffab40' if j == wi else '#8b949e')
            fig.canvas.draw_idle()
        return _cb

    for wi, wval in enumerate(_win_options):
        _win_btns[wi].on_clicked(_make_win_cb(wval, wi))

    # =================================================================
    # File dialog + recording
    # =================================================================
    def _ask_filename(title, default, ext):
        import tkinter.simpledialog as sd
        root = fig.canvas.get_tk_widget().winfo_toplevel()
        name = sd.askstring(title, f'Enter filename (without .{ext}):',
                            initialvalue=default, parent=root)
        if name:
            name = name.strip()
            if not name.endswith(f'.{ext}'):
                name = f'{name}.{ext}'
            return name
        return None

    def save_snapshot(event):
        global _snapshot_counter
        _snapshot_counter += 1
        default = f'csi_screenshot_{_snapshot_counter:03d}'
        filename = _ask_filename('Save Screenshot', default, 'png')
        if filename:
            fig.savefig(filename, dpi=150, facecolor=fig.get_facecolor())
            print(f"[screenshot] Saved to {filename}")
        else:
            _snapshot_counter -= 1

    def toggle_recording(event):
        global _video_frames, _video_recording, _video_filename
        if _video_recording:
            # Stop recording — save as animated GIF
            _video_recording = False
            n_frames = len(_video_frames)
            default = f'csi_recording_{int(time.time())}'
            filename = _ask_filename('Save Recording', default, 'gif')
            if filename and n_frames > 0:
                try:
                    _video_frames[0].save(
                        filename, save_all=True,
                        append_images=_video_frames[1:],
                        duration=100, loop=0, optimize=False)
                    print(f"[video] Saved {n_frames} frames to {filename}")
                except Exception as e:
                    print(f"[video] Save error: {e}")
            elif n_frames == 0:
                print("[video] No frames captured")
            _video_frames = []
            _video_filename = None
            rec_text.set_text('')
            btn_rec_toggle.label.set_text('\u25cf Rec')
            btn_rec_toggle.label.set_color('#4caf50')
        else:
            # Start recording — clear frame buffer
            _video_frames = []
            _video_recording = True
            rec_text.set_text('\u25cf REC')
            btn_rec_toggle.label.set_text('\u25a0 Stop')
            btn_rec_toggle.label.set_color('#f44336')
            print(f"[video] Recording started (PIL frames)")

    btn_snap.on_clicked(save_snapshot)
    btn_rec_toggle.on_clicked(toggle_recording)

    # =================================================================
    # Diagnostics callbacks (preproc only)
    # =================================================================
    def _make_pp_toggle(pp_key, pp_idx):
        def _cb(event):
            _preproc_flags[pp_key] = not _preproc_flags[pp_key]
            on = _preproc_flags[pp_key]
            pp_btns[pp_idx].label.set_color('#66bb6a' if on else '#586069')
            pp_btn_axes[pp_idx].set_facecolor('#1e3a2e' if on else '#1a2332')
            state_str = ', '.join(k for k, v in _preproc_flags.items() if v) or 'none'
            diag_status_text.set_text(f'Preproc: {state_str}')
            diag_status_text.set_color('#ffab40')
            print(f"[preproc] {pp_key} = {on}")
            fig.canvas.draw_idle()
        return _cb

    for pi, pp_key in enumerate(pp_keys):
        pp_btns[pi].on_clicked(_make_pp_toggle(pp_key, pi))

    artists = {
        'im': im, 'sc_lines': sc_lines,
        'mean_line': mean_line, 'std_hi': std_hi_line, 'std_lo': std_lo_line,
        'stats_text': stats_text, 'rec_text': rec_text,
        'profile_bars': profile_bars,
        'hist_patches': hist_patches, 'hist_stats_text': hist_stats_text,
        'var_im': var_im,
        # Analytics artists
        'stats_bars': stats_bars,
        'stats_info_text': stats_info_text,
        'analytics_status_text': analytics_status_text,
        # Activity timeline
        'timeline_scatter': timeline_scatter,
        'ax_timeline': ax_timeline,
        # Diagnostics (V4)
        'stab_im': stab_im,
        'stab_info_text': stab_info_text,
        'sc_group_lines': sc_group_lines,
        'scg_info_text': scg_info_text,
        'outlier_line': outlier_line,
        'outlier_stats_text': outlier_stats_text,
        'diag_status_text': diag_status_text,
        'diag_frame_skip': [0],
    }
    fig._csi_buttons = (btn_v1, btn_v2, btn_v3, btn_v4, btn_snap, btn_rec_toggle,
                        ax_toolbar_bg, *_win_btns, *pp_btns)
    views = {
        'ax_ts': [ax_heat, ax_lines, ax_mean],
        'ax_timeline': ax_timeline,
        'ax_stats': ax_stats,
        'ax_profile': ax_profile, 'ax_hist': ax_hist, 'ax_varheat': ax_varheat,
        'ax_temporal': ax_temporal, 'ax_scgroups': ax_scgroups,
        'ax_outliers': ax_outliers,
    }
    return fig, artists, views, frequencies


def _downsample(arr, max_pts):
    """Stride-downsample rows to at most max_pts."""
    if len(arr) <= max_pts:
        return arr
    step = max(1, len(arr) // max_pts)
    return arr[::step]


def update_once(fig, artists, views, last_seq, fps_times):
    """Single update tick. Returns (new_last_seq, did_draw)."""
    seq = _data_seq
    if len(_data_buffer) < 2 or seq == last_seq:
        return last_seq, False

    fps_times.append(time.time())

    data_array = np.array(_data_buffer)
    timestamps_array = np.array(_timestamps)
    # Guard against race: collection thread may append between the two snapshots
    n = min(len(data_array), len(timestamps_array))
    data_array = data_array[:n]
    timestamps_array = timestamps_array[:n]

    t_now = timestamps_array[-1]
    win_sec = SLIDING_WINDOW_SEC
    t_start = max(0.0, t_now - win_sec)
    t_end = t_start + win_sec
    win_mask = timestamps_array >= t_start
    data_win = data_array[win_mask]
    ts_win = timestamps_array[win_mask]

    if len(data_win) < 2:
        return last_seq, False

    ncols = min(NUM_SUBCARRIERS, data_win.shape[1])

    # ---- PCA computation removed ----
    pca_model = None

    step = max(1, len(ts_win) // 250)
    ts_m = ts_win[::step]
    chunk = data_win[::step, :ncols]
    mean_v = chunk.mean(axis=1)
    std_v = chunk.std(axis=1)

    # Shared binning (used by V1 and V3)
    if _active_view in ('timeseries', 'sigquality'):
        bin_edges = np.linspace(t_start, t_end, HEATMAP_COLS + 1)
        bin_idx = np.clip(np.digitize(ts_win, bin_edges) - 1, 0, HEATMAP_COLS - 1)
        counts = np.bincount(bin_idx, minlength=HEATMAP_COLS).astype(np.float64)

    # ---- V1: Time Series ----
    if _active_view == 'timeseries':
        heatmap_t = np.zeros((HEATMAP_COLS, ncols), dtype=np.float64)
        np.add.at(heatmap_t, bin_idx, data_win[:, :ncols])
        pop = counts > 0
        heatmap_t[pop] /= counts[pop, None]
        for b in np.where(~pop)[0]:
            if b > 0:
                heatmap_t[b] = heatmap_t[b - 1]
        heatmap = np.clip(heatmap_t.T, AMP_MIN, AMP_MAX)
        artists['im'].set_data(heatmap)
        artists['im'].set_extent([t_start, t_end, 0, NUM_SUBCARRIERS])

        ts_ds = _downsample(ts_win, MAX_LINE_PTS)
        data_ds = _downsample(data_win, MAX_LINE_PTS)
        for i in range(ncols):
            artists['sc_lines'][i].set_data(ts_ds, np.clip(data_ds[:, i], AMP_MIN, AMP_MAX))

        artists['mean_line'].set_data(ts_m, mean_v)
        artists['std_hi'].set_data(ts_m, np.clip(mean_v + std_v, AMP_MIN, AMP_MAX))
        artists['std_lo'].set_data(ts_m, np.clip(mean_v - std_v, AMP_MIN, AMP_MAX))

        for ax in views['ax_ts']:
            ax.set_xlim(t_start, t_end)

    # ---- V2: Analytics + Activity Timeline ----
    if _active_view == 'analytics':
        # Update activity timeline
        if len(ts_win) > 0:
            offsets = np.column_stack([ts_win, np.zeros(len(ts_win))])
            artists['timeline_scatter'].set_offsets(offsets)
            artists['ax_timeline'].set_xlim(t_start, t_end)
        
        # Update basic statistics
        if len(data_win) > 0:
            # Calculate basic statistics
            mean_amp = float(data_win.mean())
            std_amp = float(data_win.std())
            max_amp = float(data_win.max())
            min_amp = float(data_win.min())
            
            # Update bar chart
            stats_values = [mean_amp, std_amp, max_amp - min_amp]
            stats_labels = ['Mean', 'Std', 'Range']
            
            for i, (bar, val) in enumerate(zip(artists['stats_bars'], stats_values)):
                bar.set_height(val)
                bar.set_x(i)
                bar.set_width(0.6)
            
            views['ax_stats'].set_xlim(-0.5, len(stats_values) - 0.5)
            views['ax_stats'].set_ylim(0, max(stats_values) * 1.2)
            
            artists['stats_info_text'].set_text(
                f'Mean: {mean_amp:.1f}\nStd: {std_amp:.1f}\nRange: {max_amp - min_amp:.1f}'
            )
            
            artists['analytics_status_text'].set_text('Analytics Active')
            artists['analytics_status_text'].set_color('#66bb6a')

    # ---- V3: Signal Quality ----
    if _active_view == 'sigquality':
        sc_means = data_win[:, :ncols].mean(axis=0)
        for bar, val in zip(artists['profile_bars'], sc_means):
            bar.set_height(np.clip(val, AMP_MIN, AMP_MAX))

        all_amps = data_win[:, :ncols].ravel()
        all_amps_clipped = np.clip(all_amps, AMP_MIN, AMP_MAX)
        hist_vals, _ = np.histogram(all_amps_clipped, bins=40, range=(AMP_MIN, AMP_MAX))
        for patch, h in zip(artists['hist_patches'], hist_vals):
            patch.set_height(h)
        views['ax_hist'].set_ylim(0, 5000)  # Fixed histogram height
        artists['hist_stats_text'].set_text(
            f'mean={all_amps.mean():.1f}\nstd={all_amps.std():.1f}\n'
            f'med={np.median(all_amps):.1f}')

        var_heat = np.zeros((HEATMAP_COLS, ncols), dtype=np.float64)
        sum_sq = np.zeros((HEATMAP_COLS, ncols), dtype=np.float64)
        np.add.at(var_heat, bin_idx, data_win[:, :ncols])
        np.add.at(sum_sq, bin_idx, data_win[:, :ncols] ** 2)
        pop_v = counts > 1
        mean_bins = np.zeros_like(var_heat)
        mean_bins[pop_v] = var_heat[pop_v] / counts[pop_v, None]
        var_bins = np.zeros_like(var_heat)
        var_bins[pop_v] = sum_sq[pop_v] / counts[pop_v, None] - mean_bins[pop_v] ** 2
        var_bins = np.clip(var_bins, 0, None)
        artists['var_im'].set_data(var_bins.T)
        artists['var_im'].set_extent([t_start, t_end, 0, NUM_SUBCARRIERS])
        views['ax_varheat'].set_xlim(t_start, t_end)

    # ---- V4: Diagnostics ----
    if _active_view == 'diagnostics':
        diag_skip = artists['diag_frame_skip']
        diag_skip[0] += 1
        if diag_skip[0] >= 3:  # update every 3rd frame to save CPU
            diag_skip[0] = 0

            # Optionally apply preprocessing
            any_pp = any(_preproc_flags.values())
            diag_data = _apply_preproc(data_win[:, :ncols], _preproc_flags) if any_pp else data_win[:, :ncols]
            diag_ncols = diag_data.shape[1]

            # ── Temporal stability heatmap (rolling mean) ──
            stab_lookback = max(STABILITY_WINDOW_SEC, win_sec)
            stab_t_start = max(0.0, t_now - stab_lookback)
            stab_mask = timestamps_array >= stab_t_start
            stab_data_raw = data_array[stab_mask][:, :ncols]
            stab_data = _apply_preproc(stab_data_raw, _preproc_flags) if any_pp else stab_data_raw
            stab_ts = timestamps_array[stab_mask]
            stab_ncols = stab_data.shape[1]

            if len(stab_data) > STABILITY_ROLLING_WIN:
                rm = _rolling_mean(stab_data, STABILITY_ROLLING_WIN)
                stab_bins = np.linspace(stab_t_start, t_now, STABILITY_HEATMAP_COLS + 1)
                stab_bidx = np.clip(np.digitize(stab_ts, stab_bins) - 1, 0, STABILITY_HEATMAP_COLS - 1)
                stab_heat = np.zeros((STABILITY_HEATMAP_COLS, stab_ncols), dtype=np.float64)
                np.add.at(stab_heat, stab_bidx, rm)
                stab_cnt = np.bincount(stab_bidx, minlength=STABILITY_HEATMAP_COLS).astype(np.float64)
                sp = stab_cnt > 0
                stab_heat[sp] /= stab_cnt[sp, None]
                for b in np.where(~sp)[0]:
                    if b > 0:
                        stab_heat[b] = stab_heat[b - 1]
                artists['stab_im'].set_data(stab_heat.T)
                artists['stab_im'].set_extent([stab_t_start, t_now, 0, stab_ncols])
                views['ax_temporal'].set_xlim(stab_t_start, t_now)
                rm_var = rm.var(axis=0).mean()
                artists['stab_info_text'].set_text(
                    f'Win={STABILITY_ROLLING_WIN}pkt  Drift={rm_var:.2f}')

            # ── Subcarrier group sensitivity (vectorized) ──
            if len(diag_data) > 20:
                grp_bins = 60
                grp_edges = np.linspace(t_start, t_end, grp_bins + 1)
                grp_bidx = np.clip(np.digitize(ts_win, grp_edges) - 1, 0, grp_bins - 1)
                grp_ts_centers = (grp_edges[:-1] + grp_edges[1:]) / 2

                max_var = 0
                for gname, gidx_list in SC_GROUPS.items():
                    valid_idx = [i for i in gidx_list if i < diag_ncols]
                    if not valid_idx:
                        continue
                    grp_sub = diag_data[:, valid_idx]
                    # Vectorized per-bin variance using add.at
                    grp_sum = np.zeros(grp_bins, dtype=np.float64)
                    grp_sum2 = np.zeros(grp_bins, dtype=np.float64)
                    grp_cnt = np.bincount(grp_bidx, minlength=grp_bins).astype(np.float64)
                    flat_mean = grp_sub.mean(axis=1)  # mean across subcarriers per pkt
                    np.add.at(grp_sum, grp_bidx, flat_mean)
                    np.add.at(grp_sum2, grp_bidx, flat_mean ** 2)
                    pop_g = grp_cnt > 1
                    grp_var_arr = np.zeros(grp_bins, dtype=np.float64)
                    grp_var_arr[pop_g] = (grp_sum2[pop_g] / grp_cnt[pop_g]
                                          - (grp_sum[pop_g] / grp_cnt[pop_g]) ** 2)
                    grp_var_arr = np.clip(grp_var_arr, 0, None)
                    # Forward-fill empty bins
                    for bi in range(1, grp_bins):
                        if grp_cnt[bi] == 0 and grp_cnt[bi - 1] > 0:
                            grp_var_arr[bi] = grp_var_arr[bi - 1]
                    artists['sc_group_lines'][gname].set_data(grp_ts_centers, grp_var_arr)
                    max_var = max(max_var, float(grp_var_arr.max()))

                views['ax_scgroups'].set_xlim(t_start, t_end)
                views['ax_scgroups'].set_ylim(0, 50)  # Fixed variance range

                group_vars = {}
                for gname, gidx_list in SC_GROUPS.items():
                    valid_idx = [i for i in gidx_list if i < diag_ncols]
                    if valid_idx:
                        group_vars[gname] = float(diag_data[:, valid_idx].var())
                if group_vars:
                    most_sens = max(group_vars, key=group_vars.get)
                    uniformity = min(group_vars.values()) / max(max(group_vars.values()), 1e-9)
                    artists['scg_info_text'].set_text(
                        f'Most sensitive: {most_sens}\n'
                        f'Uniformity: {uniformity:.2f}')

            # ── Outlier / impulse detection (vectorized) ──
            if len(diag_data) > _outlier_window * 2:
                check_step = max(1, len(diag_data) // 500)
                check_data = diag_data[::check_step]
                check_ts = ts_win[::check_step]
                _, pkt_counts = _detect_outliers(check_data, _outlier_z_threshold,
                                                  min(_outlier_window, max(5, len(check_data) // 5)))
                if len(check_ts) > 1:
                    dur = check_ts[-1] - check_ts[0]
                    if dur > 0.5:
                        n_sec_bins = max(1, int(dur))
                        sec_edges = np.linspace(check_ts[0], check_ts[-1], n_sec_bins + 1)
                        sec_bidx = np.clip(np.digitize(check_ts, sec_edges) - 1, 0, n_sec_bins - 1)
                        # Vectorized impulse binning via bincount
                        imp_per_sec = np.bincount(sec_bidx, weights=pkt_counts.astype(np.float64),
                                                   minlength=n_sec_bins).astype(np.float64)
                        sec_centers = (sec_edges[:-1] + sec_edges[1:]) / 2
                        artists['outlier_line'].set_data(sec_centers, imp_per_sec)
                        views['ax_outliers'].set_xlim(t_start, t_end)
                        views['ax_outliers'].set_ylim(0, 20)  # Fixed outlier rate range

                        total_imp = int(pkt_counts.sum())
                        mean_rate = float(imp_per_sec.mean())
                        artists['outlier_stats_text'].set_text(
                            f'Total: {total_imp}\n'
                            f'Rate: {mean_rate:.1f}/s\n'
                            f'z>{_outlier_z_threshold}')


    # ---- Stats text ----
    plot_fps = 0.0
    if len(fps_times) >= 2:
        dt = fps_times[-1] - fps_times[0]
        if dt > 0:
            plot_fps = (len(fps_times) - 1) / dt
    csi_rate = len(data_win) / max(t_now - t_start, 0.01)
    cur_mean = float(mean_v[-1]) if len(mean_v) else 0
    cur_std  = float(std_v[-1])  if len(std_v)  else 0
    status = 'LIVE' if not _collection_done else 'DONE'
    vtag = {'timeseries': 'TS', 'analytics': 'AN', 'sigquality': 'SQ',
            'diagnostics': 'DX'}.get(_active_view, '??')
    cal_tag = ''
    
    # PCA status tag removed
    pca_tag = ''
    
    pp_tag = ''
    if any(_preproc_flags.values()):
        pp_tag = '  PP:' + '+'.join(k[:3] for k, v in _preproc_flags.items() if v)
    ref_tag = ''
    artists['stats_text'].set_text(
        f'[{status}:{vtag}]  Samples: {seq:,}  |  '
        f'CSI: {csi_rate:.0f} pkt/s  |  Plot: {plot_fps:.1f} fps  |  '
        f'Mean: {cur_mean:.1f}  |  Std: {cur_std:.1f}  |  '
        f'Win: {win_sec:.0f}s  |  Time: {t_now:.1f}s{cal_tag}{pca_tag}{pp_tag}{ref_tag}')

    # Video recording frame capture
    if _video_recording:
        try:
            from PIL import Image
            buf = fig.canvas.buffer_rgba()
            frame = Image.frombytes('RGBA', fig.canvas.get_width_height(), bytes(buf))
            _video_frames.append(frame.convert('RGB'))
        except Exception:
            pass

    fig.canvas.draw_idle()
    return seq, True


def main():
    global _occupation_model, _occupation_window, _occupation_var_window
    global _occupation_model_path

    parser = argparse.ArgumentParser(description="CSI viewer with analytics")
    parser.add_argument("--rx-port", required=True)
    parser.add_argument("--tx-port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--occupation-model", type=str, default=None,
                        help="Path to occupation model (.pkl). Deprecated - simple analytics used instead.")
    parser.add_argument("--conf-threshold", type=float, default=0.40,
                        help="Min probability to accept a prediction (0-1, default: 0.40)")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Temporal smoothing: majority-vote over last N predictions (default: 5)")

    args = parser.parse_args()

    # Load occupation model if provided
    if args.occupation_model:
        _occupation_model_path = args.occupation_model
        if os.path.exists(args.occupation_model):
            with open(args.occupation_model, 'rb') as f:
                _occupation_model = pickle.load(f)
            _occupation_window = _occupation_model.get('window', 300)
            _occupation_var_window = _occupation_model.get('var_window', 100)
            _cal_resample_sr = _occupation_model.get('sr', 150)
            print(f"[model] Loaded: {args.occupation_model}")
            print(f"[model] Labels: {_occupation_model['labels']}")
            print(f"[model] Window: {_occupation_window}, Var window: {_occupation_var_window}, "
                  f"Features: {_occupation_model['n_features']}")
            print(f"[model] Accuracy: {_occupation_model.get('accuracy', 0):.4f}")
        else:
            _occupation_model_path = args.occupation_model
            print(f"[model] File not found: {args.occupation_model} "
                  f"(will save here after calibration)")

    _conf_threshold = args.conf_threshold
    _smooth_window = max(1, args.smooth_window)
    print(f"[config] Confidence: {_conf_threshold:.0%}, Smoothing: {_smooth_window} frames")

    print(f"[info] CSI Viewer starting...")
    print(f"[info] Receiver: {args.rx_port} @ {args.baud}")
    if args.tx_port:
        print(f"[info] Sender: {args.tx_port} @ {args.baud}")
    print(f"[info] Duration: {args.duration}s  |  Subcarriers: {NUM_SUBCARRIERS}")

    fig, artists, views, frequencies = create_visualization()

    def _collect_wrapper(*a):
        global _collection_done
        try:
            collect_csi_subprocess(*a)
        finally:
            _collection_done = True
            print("[info] Collection finished \u2014 close window to exit.")

    threading.Thread(
        target=_collect_wrapper,
        args=(args.rx_port, args.tx_port, args.baud, args.duration),
        daemon=True,
    ).start()

    print("[info] Starting real-time visualization...")

    plt.ion()
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    last_seq = 0
    fps_times = deque(maxlen=30)
    target_dt = 0.10

    try:
        while plt.fignum_exists(fig.number):
            t0 = time.time()
            last_seq, _ = update_once(fig, artists, views, last_seq, fps_times)
            elapsed = time.time() - t0
            sleep_t = max(0.01, target_dt - elapsed)
            try:
                tk = fig.canvas.get_tk_widget()
                end_t = time.time() + sleep_t
                while time.time() < end_t:
                    tk.update()
                    time.sleep(0.005)
            except Exception:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        print("\n[info] Interrupted")
    finally:
        if _video_recording and _video_frames:
            try:
                from PIL import Image
                auto_name = _video_filename or f'csi_rec_autosave_{int(time.time())}.gif'
                if not auto_name.endswith('.gif'):
                    auto_name = auto_name.rsplit('.', 1)[0] + '.gif'
                _video_frames[0].save(
                    auto_name, save_all=True,
                    append_images=_video_frames[1:],
                    duration=100, loop=0, optimize=False)
                print(f"[video] Auto-saved {len(_video_frames)} frames to {auto_name}")
            except Exception as e:
                print(f"[video] Auto-save failed: {e}")
        stop_event.set()
        # Kill the collect_csi subprocess if still running
        if _csi_process is not None:
            try:
                _csi_process.terminate()
                _csi_process.wait(timeout=3)
                print("[info] Subprocess terminated.")
            except Exception:
                try:
                    _csi_process.kill()
                    print("[info] Subprocess killed.")
                except Exception:
                    pass

    print(f"[info] Viewer ended. {len(_data_buffer)} samples, "
          f"{_snapshot_counter} snapshots")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: (stop_event.set(), print("\n[info] Stopping...")))
    main()
