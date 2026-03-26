#!/usr/bin/env python3
"""
CSI viewer that calls collect_csi as a subprocess and visualizes amplitudes.
- Runs collect_csi.py as a subprocess to collect data
- Plots amplitudes of 52 subcarriers with real Hz axis
- Two tabs: Signals and Quality
- Per-graph PDF export buttons for publication-ready figures
"""

import argparse
import sys
import time
import threading
import signal
import math
import subprocess
import numpy as np
from collections import deque
import os
import pickle

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
HEATMAP_COLS = 80
SLIDING_WINDOW_SEC = 10.0
MAX_LINE_PTS = 80
_data_buffer = deque(maxlen=15000)
_timestamps = deque(maxlen=15000)
_data_seq = 0
_snapshot_counter = 0
_collection_done = False
_csi_process = None
_occupation_model = None
_occupation_window = 300
_occupation_var_window = 100
_occupation_model_path = None

# Subcarrier groups for sensitivity analysis
SC_GROUPS = {
    'low':  list(range(0, 17)),
    'mid':  list(range(17, 35)),
    'high': list(range(35, 52)),
}
SC_GROUP_COLORS = {'low': '#42a5f5', 'mid': '#66bb6a', 'high': '#ef5350'}

# CSI parameters
NUM_SUBCARRIERS = 52
SUBCARRIER_SPACING = 312.5e3
CENTER_FREQ = 2.437e9
FFT_SIZE = 64

# Fixed amplitude range
AMP_MIN = 0
AMP_MAX = 50

# Publication font sizes
FONT_TITLE    = 14
FONT_LABEL    = 12
FONT_TICK     = 10
FONT_LEGEND   = 10
FONT_STATS    = 11
FONT_BTN      = 8
FONT_SUPTITLE = 16

# Active view
_active_view = 'timeseries'
_saved_ax_positions = {}


def get_subcarrier_frequencies():
    """Calculate actual frequencies for each subcarrier."""
    subcarrier_indices = np.arange(-26, 26)
    return CENTER_FREQ + subcarrier_indices * SUBCARRIER_SPACING


def parse_csi_line(line):
    """Parse a single CSI line from collect_csi output."""
    try:
        parts = line.split(',', 14)
        if len(parts) < 15:
            return None
        data_str = parts[14]
        if not data_str:
            return None
        nums = data_str.strip('"[] ').split(',')
        amps = []
        for j in range(0, len(nums) - 1, 2):
            try:
                im_v = float(nums[j])
                re_v = float(nums[j + 1])
                amps.append(math.sqrt(re_v * re_v + im_v * im_v))
            except (ValueError, IndexError):
                continue
        if len(amps) == 64:
            return np.array(amps)[CSI_SUBCARRIER_MASK]
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
        process = subprocess.Popen(cmd, stdout=None, stderr=None)
        _csi_process = process
        start_time = time.time()

        while not os.path.exists(actual_csv_path) and not stop_event.is_set():
            if process.poll() is not None:
                break
            time.sleep(0.05)

        if not os.path.exists(actual_csv_path):
            print(f"[error] CSV file never appeared: {actual_csv_path}")
            return

        print(f"[info] Tailing CSV in real-time: {actual_csv_path}")
        line_count = 0
        header_skipped = False

        with open(actual_csv_path, 'r') as f:
            while not stop_event.is_set():
                line = f.readline()
                if line:
                    line = line.strip()
                    if not header_skipped:
                        header_skipped = True
                        continue
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
                    if process.poll() is not None:
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
                    time.sleep(0.01)

        return_code = process.wait()
        print(f"\n[info] Subprocess exited (rc={return_code}), parsed {line_count} CSI lines")

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


def _set_axes_visible(axes_list, visible):
    """Show/hide a list of axes, moving button axes off-screen when hidden."""
    from matplotlib.transforms import Bbox
    for ax in axes_list:
        ax.set_visible(visible)
        ax_id = id(ax)
        if visible:
            if ax_id in _saved_ax_positions:
                ax.set_position(_saved_ax_positions.pop(ax_id))
        else:
            if ax_id not in _saved_ax_positions:
                _saved_ax_positions[ax_id] = ax.get_position()
            ax.set_position(Bbox([[9, 9], [9.01, 9.01]]))


def _style_ax(ax):
    """Apply publication styling to an axis."""
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#546e7a')
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(direction='out', length=4, width=0.8,
                   colors='#90a4ae', labelsize=FONT_TICK)


def _export_single_ax_pdf(fig, source_ax, title):
    """Export a single axes to a standalone publication-ready PDF."""
    global _snapshot_counter
    _snapshot_counter += 1

    import tkinter.simpledialog as sd
    try:
        root = fig.canvas.get_tk_widget().winfo_toplevel()
    except Exception:
        root = None
    default = f'csi_{title.lower().replace(" ", "_")}_{_snapshot_counter:03d}'
    name = sd.askstring('Export PDF', 'Filename (without .pdf):',
                        initialvalue=default, parent=root)
    if not name:
        _snapshot_counter -= 1
        return
    name = name.strip()
    if not name.endswith('.pdf'):
        name += '.pdf'

    pub_fig, pub_ax = plt.subplots(figsize=(7, 4))
    pub_fig.patch.set_facecolor('white')

    for line in source_ax.get_lines():
        xd, yd = line.get_data()
        pub_ax.plot(xd, yd, color=line.get_color(),
                    linewidth=max(line.get_linewidth(), 1.0),
                    alpha=line.get_alpha() or 1.0,
                    label=line.get_label() if not line.get_label().startswith('_') else None,
                    linestyle=line.get_linestyle())

    for im_artist in source_ax.get_images():
        data = im_artist.get_array()
        extent = im_artist.get_extent()
        pub_ax.imshow(data, aspect='auto', origin='lower',
                      cmap=im_artist.get_cmap(), extent=extent,
                      interpolation='bilinear',
                      vmin=im_artist.get_clim()[0], vmax=im_artist.get_clim()[1])

    for container in source_ax.containers:
        if hasattr(container, 'patches'):
            for patch in container:
                from matplotlib.patches import Rectangle
                r = Rectangle(
                    (patch.get_x(), patch.get_y()),
                    patch.get_width(), patch.get_height(),
                    facecolor=patch.get_facecolor(),
                    edgecolor='#263238', linewidth=0.5, alpha=0.85)
                pub_ax.add_patch(r)

    for coll in source_ax.collections:
        offsets = coll.get_offsets()
        if len(offsets) > 0:
            pub_ax.scatter(offsets[:, 0], offsets[:, 1],
                           s=coll.get_sizes(), c='#0288d1',
                           edgecolors='none', alpha=0.8)

    pub_ax.set_xlim(source_ax.get_xlim())
    pub_ax.set_ylim(source_ax.get_ylim())
    pub_ax.set_xlabel(source_ax.get_xlabel(), fontsize=12, fontfamily='serif')
    pub_ax.set_ylabel(source_ax.get_ylabel(), fontsize=12, fontfamily='serif')
    pub_ax.set_title(title, fontsize=14, fontweight='semibold', fontfamily='serif', pad=10)
    pub_ax.tick_params(labelsize=10, direction='out')
    for sp in ['top', 'right']:
        pub_ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        pub_ax.spines[sp].set_linewidth(0.8)
        pub_ax.spines[sp].set_color('#333333')
    if pub_ax.get_legend_handles_labels()[1]:
        pub_ax.legend(fontsize=10, framealpha=0.8, edgecolor='#cccccc')
    pub_ax.grid(True, alpha=0.15, linewidth=0.5, color='#888888')

    pub_fig.tight_layout()
    pub_fig.savefig(name, format='pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(pub_fig)
    print(f"[pdf] Exported: {name}")


def _make_pdf_btn(fig, ax_source, title, x, y, w, h):
    """Create a small PDF export button next to a graph axes."""
    ax_btn = fig.add_axes([x, y, w, h])
    btn = Button(ax_btn, 'PDF', color='#1a2332', hovercolor='#2a4060')
    btn.label.set_color('#80cbc4')
    btn.label.set_fontsize(FONT_BTN)
    btn.on_clicked(lambda event: _export_single_ax_pdf(fig, ax_source, title))
    return ax_btn, btn


def create_visualization():
    """Create two-tab layout: Signals and Quality, with per-graph PDF export."""
    frequencies = get_subcarrier_frequencies()
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('ThothCraft  \u2014  CSI Real-Time Monitor',
                 fontsize=FONT_SUPTITLE, fontweight='bold', color='#00e5ff')

    L, R, T, B = 0.06, 0.92, 0.90, 0.08

    # Stats bar (always visible)
    ax_info = fig.add_axes([L, 0.035, R - L, 0.030])
    ax_info.axis('off')
    stats_text = ax_info.text(
        0.5, 0.5, '', transform=ax_info.transAxes,
        ha='center', va='center', fontsize=FONT_STATS, color='#b0bec5',
        family='monospace',
    )

    # PDF button positioning
    _pdf_x = R + 0.01
    _pdf_w = 0.035
    _pdf_h = 0.022
    _all_pdf_btn_axes = []
    _all_pdf_btns = []

    def _add_pdf(ax_source, title, row_y):
        ax_b, b = _make_pdf_btn(fig, ax_source, title, _pdf_x, row_y, _pdf_w, _pdf_h)
        _all_pdf_btn_axes.append(ax_b)
        _all_pdf_btns.append(b)
        return ax_b

    # =================================================================
    # TAB 1: Signals (heatmap + subcarrier lines + mean/std)
    # =================================================================
    gs1 = fig.add_gridspec(3, 2, height_ratios=[3.0, 3.0, 2.5],
                           width_ratios=[1, 0.018],
                           left=L, right=R, top=T, bottom=B,
                           hspace=0.42, wspace=0.03)
    ax_heat  = fig.add_subplot(gs1[0, 0])
    ax_cbar1 = fig.add_subplot(gs1[0, 1])
    ax_lines = fig.add_subplot(gs1[1, 0])
    ax_mean  = fig.add_subplot(gs1[2, 0])

    sc_colors = plt.cm.turbo(np.linspace(0.05, 0.95, NUM_SUBCARRIERS))

    blank = np.zeros((NUM_SUBCARRIERS, HEATMAP_COLS))
    im = ax_heat.imshow(blank, aspect='auto', origin='lower', cmap='inferno',
                        extent=[0, SLIDING_WINDOW_SEC, 0, NUM_SUBCARRIERS],
                        interpolation='bilinear', vmin=AMP_MIN, vmax=AMP_MAX)
    ax_heat.set_ylabel('Subcarrier Index', fontsize=FONT_LABEL, color='#b0bec5')
    ax_heat.set_title('Amplitude Heatmap', fontsize=FONT_TITLE,
                      color='#e0e0e0', fontweight='semibold', pad=6)
    ax_heat.set_xlim(0, SLIDING_WINDOW_SEC)
    _style_ax(ax_heat)
    cbar1 = fig.colorbar(im, cax=ax_cbar1)
    cbar1.set_label('Amplitude', fontsize=FONT_LABEL - 2, color='#b0bec5')
    cbar1.ax.tick_params(colors='#90a4ae', labelsize=FONT_TICK - 2)

    SC_LINE_STEP = 4  # draw every 4th subcarrier (13 lines instead of 52)
    sc_line_indices = list(range(0, NUM_SUBCARRIERS, SC_LINE_STEP))
    sc_lines = []
    for i in sc_line_indices:
        ln, = ax_lines.plot([], [], color=sc_colors[i], linewidth=0.6, alpha=0.70)
        sc_lines.append(ln)
    ax_lines.set_ylabel('Amplitude', fontsize=FONT_LABEL, color='#b0bec5')
    ax_lines.set_ylim(AMP_MIN, AMP_MAX)
    ax_lines.set_xlim(0, SLIDING_WINDOW_SEC)
    ax_lines.set_title('Subcarrier Traces (every 4th)', fontsize=FONT_TITLE,
                       color='#e0e0e0', fontweight='semibold', pad=6)
    ax_lines.grid(True, alpha=0.10, color='#455a64', linewidth=0.4)
    _style_ax(ax_lines)

    mean_line, = ax_mean.plot([], [], color='#00e5ff', linewidth=1.8, label='Mean')
    std_hi_line, = ax_mean.plot([], [], color='#00e5ff', linewidth=0.6, alpha=0.30)
    std_lo_line, = ax_mean.plot([], [], color='#00e5ff', linewidth=0.6, alpha=0.30)
    ax_mean.set_ylabel('Mean Amplitude', fontsize=FONT_LABEL, color='#b0bec5')
    ax_mean.set_xlabel('Time (s)', fontsize=FONT_LABEL, color='#b0bec5')
    ax_mean.set_xlim(0, SLIDING_WINDOW_SEC)
    ax_mean.set_ylim(AMP_MIN, AMP_MAX)
    ax_mean.grid(True, alpha=0.10, color='#455a64', linewidth=0.4)
    ax_mean.legend(loc='upper right', fontsize=FONT_LEGEND, framealpha=0.3)
    ax_mean.set_title('Mean \u00b1 Std Amplitude', fontsize=FONT_TITLE,
                      color='#e0e0e0', fontweight='semibold', pad=6)
    _style_ax(ax_mean)

    v1_pdf1 = _add_pdf(ax_heat, 'Amplitude Heatmap', 0.78)
    v1_pdf2 = _add_pdf(ax_lines, 'Subcarrier Traces', 0.50)
    v1_pdf3 = _add_pdf(ax_mean, 'Mean Std Amplitude', 0.22)

    v1_axes = [ax_heat, ax_cbar1, ax_lines, ax_mean, v1_pdf1, v1_pdf2, v1_pdf3]

    # =================================================================
    # TAB 2: Quality (histogram + variance heatmap + group sensitivity)
    # =================================================================
    gs2 = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.2, 1.0],
                           left=L, right=R, top=T, bottom=B,
                           hspace=0.42)
    ax_hist    = fig.add_subplot(gs2[0, 0])
    ax_varheat = fig.add_subplot(gs2[1, 0])
    ax_scgroups = fig.add_subplot(gs2[2, 0])

    # Amplitude distribution histogram
    hist_n, hist_bins, hist_patches = ax_hist.hist(
        [0], bins=40, range=(AMP_MIN, AMP_MAX),
        color='#00e5ff', alpha=0.7, edgecolor='#263238', linewidth=0.4)
    ax_hist.set_xlabel('Amplitude', fontsize=FONT_LABEL, color='#b0bec5')
    ax_hist.set_ylabel('Count', fontsize=FONT_LABEL, color='#b0bec5')
    ax_hist.set_title('Amplitude Distribution', fontsize=FONT_TITLE,
                       color='#e0e0e0', fontweight='semibold', pad=6)
    ax_hist.set_xlim(AMP_MIN, AMP_MAX)
    ax_hist.grid(True, axis='y', alpha=0.10, color='#455a64', linewidth=0.4)
    _style_ax(ax_hist)
    hist_stats_text = ax_hist.text(0.97, 0.95, '', transform=ax_hist.transAxes,
                                    fontsize=FONT_TICK, color='#80cbc4',
                                    va='top', ha='right', family='monospace')

    # Rolling variance heatmap
    var_blank = np.zeros((NUM_SUBCARRIERS, HEATMAP_COLS))
    var_im = ax_varheat.imshow(var_blank, aspect='auto', origin='lower',
                                cmap='magma', interpolation='bilinear',
                                vmin=0, vmax=25)
    ax_varheat.set_ylabel('Subcarrier Index', fontsize=FONT_LABEL, color='#b0bec5')
    ax_varheat.set_xlabel('Time (s)', fontsize=FONT_LABEL, color='#b0bec5')
    ax_varheat.set_title('Rolling Variance Heatmap',
                          fontsize=FONT_TITLE, color='#e0e0e0',
                          fontweight='semibold', pad=6)
    _style_ax(ax_varheat)

    # Subcarrier group sensitivity
    sc_group_lines = {}
    for gname, gcolor in SC_GROUP_COLORS.items():
        ln, = ax_scgroups.plot([], [], color=gcolor, linewidth=1.6,
                                label=gname, alpha=0.85)
        sc_group_lines[gname] = ln
    ax_scgroups.set_ylabel('Group Variance', fontsize=FONT_LABEL, color='#b0bec5')
    ax_scgroups.set_xlabel('Time (s)', fontsize=FONT_LABEL, color='#b0bec5')
    ax_scgroups.set_title('Subcarrier Group Sensitivity',
                           fontsize=FONT_TITLE, color='#e0e0e0',
                           fontweight='semibold', pad=6)
    ax_scgroups.set_ylim(0, 30)
    ax_scgroups.grid(True, alpha=0.10, color='#455a64', linewidth=0.4)
    ax_scgroups.legend(loc='upper right', fontsize=FONT_LEGEND, framealpha=0.4)
    _style_ax(ax_scgroups)
    scg_info_text = ax_scgroups.text(
        0.02, 0.95, '', transform=ax_scgroups.transAxes,
        fontsize=FONT_TICK - 1, color='#ffab40', va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', alpha=0.7))

    v2_pdf1 = _add_pdf(ax_hist, 'Amplitude Distribution', 0.78)
    v2_pdf2 = _add_pdf(ax_varheat, 'Variance Heatmap', 0.50)
    v2_pdf3 = _add_pdf(ax_scgroups, 'Group Sensitivity', 0.22)

    v2_axes = [ax_hist, ax_varheat, ax_scgroups, v2_pdf1, v2_pdf2, v2_pdf3]

    # Start with only V1 visible
    _set_axes_visible(v2_axes, False)

    # =================================================================
    # Toolbar — two tabs + window length
    # =================================================================
    _tab_color   = '#1a2332'
    _tab_active  = '#263d52'
    _tab_hover   = '#2a4060'
    _ctrl_color  = '#1e2a38'
    _ctrl_hover  = '#2a3a4e'
    ctrl_style = dict(color=_ctrl_color, hovercolor=_ctrl_hover)

    ax_toolbar_bg = fig.add_axes([0.0, 0.0, 1.0, 0.032])
    ax_toolbar_bg.set_facecolor('#0d1117')
    ax_toolbar_bg.set_xticks([]); ax_toolbar_bg.set_yticks([])
    for sp in ax_toolbar_bg.spines.values():
        sp.set_visible(False)

    _tab_w, _tab_h, _tab_y = 0.08, 0.022, 0.005
    _tab_gap = 0.004

    ax_b1 = fig.add_axes([0.015, _tab_y, _tab_w, _tab_h])
    btn_v1 = Button(ax_b1, 'Signals', color=_tab_active, hovercolor=_tab_hover)
    btn_v1.label.set_color('#ffab40'); btn_v1.label.set_fontsize(FONT_BTN)
    btn_v1.label.set_fontweight('bold')

    ax_b2 = fig.add_axes([0.015 + _tab_w + _tab_gap, _tab_y, _tab_w, _tab_h])
    btn_v2 = Button(ax_b2, 'Quality', color=_tab_color, hovercolor=_tab_hover)
    btn_v2.label.set_color('#586069'); btn_v2.label.set_fontsize(FONT_BTN)

    fig.text(0.20, 0.016, '\u2502', fontsize=10, color='#30363d', va='center')

    # Window length buttons
    fig.text(0.215, 0.016, 'Window:', fontsize=7, color='#586069', va='center')
    _win_options = [5.0, 10.0, 20.0, 30.0, 60.0]
    _win_btn_axes = []
    _win_btns = []
    for wi, wval in enumerate(_win_options):
        ax_w = fig.add_axes([0.26 + wi * 0.038, _tab_y, 0.035, _tab_h])
        b = Button(ax_w, f'{int(wval)}s', **ctrl_style)
        b.label.set_fontsize(FONT_BTN - 1)
        b.label.set_color('#ffab40' if wval == SLIDING_WINDOW_SEC else '#8b949e')
        _win_btn_axes.append(ax_w)
        _win_btns.append(b)

    # =================================================================
    # Callbacks
    # =================================================================
    all_view_groups = [v1_axes, v2_axes]
    view_names = ['timeseries', 'quality']
    _tab_btns = [btn_v1, btn_v2]
    _tab_btn_axes = [ax_b1, ax_b2]
    _tab_colors = ['#ffab40', '#b388ff']

    def _switch_view(idx):
        def _cb(event):
            global _active_view
            _active_view = view_names[idx]
            for i, grp in enumerate(all_view_groups):
                _set_axes_visible(grp, i == idx)
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

    def _make_win_cb(wval, wi):
        def _cb(event):
            global SLIDING_WINDOW_SEC
            SLIDING_WINDOW_SEC = wval
            for j, b in enumerate(_win_btns):
                b.label.set_color('#ffab40' if j == wi else '#8b949e')
            # Update all axes x-limits to new fixed [0, wval]
            for ax in [ax_heat, ax_lines, ax_mean]:
                ax.set_xlim(0, wval)
            ax_varheat.set_xlim(0, wval)
            ax_scgroups.set_xlim(0, wval)
            fig.canvas.draw_idle()
        return _cb

    for wi, wval in enumerate(_win_options):
        _win_btns[wi].on_clicked(_make_win_cb(wval, wi))

    # =================================================================
    # Pack return values
    # =================================================================
    artists = {
        'im': im, 'sc_lines': sc_lines, 'sc_line_indices': sc_line_indices,
        'mean_line': mean_line, 'std_hi': std_hi_line, 'std_lo': std_lo_line,
        'stats_text': stats_text,
        'hist_patches': hist_patches, 'hist_stats_text': hist_stats_text,
        'var_im': var_im,
        'sc_group_lines': sc_group_lines,
        'scg_info_text': scg_info_text,
    }
    fig._csi_buttons = (btn_v1, btn_v2, ax_toolbar_bg,
                        *_win_btns, *_all_pdf_btns)
    views = {
        'ax_ts': [ax_heat, ax_lines, ax_mean],
        'ax_hist': ax_hist, 'ax_varheat': ax_varheat,
        'ax_scgroups': ax_scgroups,
    }
    return fig, artists, views, frequencies


def _downsample(arr, max_pts):
    """Stride-downsample rows to at most max_pts."""
    if len(arr) <= max_pts:
        return arr
    step = max(1, len(arr) // max_pts)
    return arr[::step]


def update_once(fig, artists, views, last_seq, fps_times):
    """Single update tick — x-axis fixed [0, win_sec], bin-averaged smooth lines."""
    seq = _data_seq
    if len(_data_buffer) < 2 or seq == last_seq:
        return last_seq, False

    fps_times.append(time.time())

    win_sec = SLIDING_WINDOW_SEC
    buf_list = list(_data_buffer)
    ts_list = list(_timestamps)
    n = min(len(buf_list), len(ts_list))
    if n < 2:
        return last_seq, False

    t_now = ts_list[n - 1]
    t_start = max(0.0, t_now - win_sec)

    # Binary search for window start
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_list[mid] < t_start:
            lo = mid + 1
        else:
            hi = mid

    data_win = np.array(buf_list[lo:n])
    ts_abs = np.array(ts_list[lo:n])

    if len(data_win) < 2:
        return last_seq, False

    ncols = min(NUM_SUBCARRIERS, data_win.shape[1])

    # Remap timestamps to [0, win_sec] — x-axis stays fixed
    ts_win = ts_abs - t_start

    # Shared binning in [0, win_sec] — used for heatmaps and smooth lines
    nbins = HEATMAP_COLS
    bin_edges = np.linspace(0, win_sec, nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    bin_idx = np.clip(np.digitize(ts_win, bin_edges) - 1, 0, nbins - 1)
    counts = np.bincount(bin_idx, minlength=nbins).astype(np.float64)
    pop = counts > 0

    # Bin-averaged mean and std across all subcarriers (smooth curves)
    mean_sum = np.zeros(nbins, dtype=np.float64)
    mean_sq_sum = np.zeros(nbins, dtype=np.float64)
    row_means = data_win[:, :ncols].mean(axis=1)
    np.add.at(mean_sum, bin_idx, row_means)
    np.add.at(mean_sq_sum, bin_idx, row_means ** 2)
    bin_mean = np.zeros(nbins)
    bin_std = np.zeros(nbins)
    bin_mean[pop] = mean_sum[pop] / counts[pop]
    bin_std[pop] = np.sqrt(np.clip(
        mean_sq_sum[pop] / counts[pop] - bin_mean[pop] ** 2, 0, None))
    # Forward-fill empty bins for continuity
    for b in np.where(~pop)[0]:
        if b > 0:
            bin_mean[b] = bin_mean[b - 1]
            bin_std[b] = bin_std[b - 1]

    # ---- TAB 1: Signals ----
    if _active_view == 'timeseries':
        # Heatmap (bin-averaged per subcarrier)
        heatmap_t = np.zeros((nbins, ncols), dtype=np.float64)
        np.add.at(heatmap_t, bin_idx, data_win[:, :ncols])
        heatmap_t[pop] /= counts[pop, None]
        for b in np.where(~pop)[0]:
            if b > 0:
                heatmap_t[b] = heatmap_t[b - 1]
        artists['im'].set_data(np.clip(heatmap_t.T, AMP_MIN, AMP_MAX))
        artists['im'].set_extent([0, win_sec, 0, NUM_SUBCARRIERS])

        # Subcarrier lines — bin-averaged, smooth
        sc_line_indices = artists['sc_line_indices']
        for li, sci in enumerate(sc_line_indices):
            if sci < ncols:
                sc_bin = np.zeros(nbins, dtype=np.float64)
                np.add.at(sc_bin, bin_idx, data_win[:, sci])
                sc_bin[pop] /= counts[pop]
                for b in np.where(~pop)[0]:
                    if b > 0:
                        sc_bin[b] = sc_bin[b - 1]
                artists['sc_lines'][li].set_data(
                    bin_centers, np.clip(sc_bin, AMP_MIN, AMP_MAX))

        # Mean +/- std lines (already bin-averaged above)
        artists['mean_line'].set_data(bin_centers, bin_mean)
        artists['std_hi'].set_data(bin_centers,
                                   np.clip(bin_mean + bin_std, AMP_MIN, AMP_MAX))
        artists['std_lo'].set_data(bin_centers,
                                   np.clip(bin_mean - bin_std, AMP_MIN, AMP_MAX))

    # ---- TAB 2: Quality ----
    elif _active_view == 'quality':
        # Amplitude distribution
        sub_step = max(1, len(data_win) // 1000)
        all_amps = data_win[::sub_step, :ncols].ravel()
        all_amps_clipped = np.clip(all_amps, AMP_MIN, AMP_MAX)
        hist_vals, _ = np.histogram(all_amps_clipped, bins=40, range=(AMP_MIN, AMP_MAX))
        for patch, h in zip(artists['hist_patches'], hist_vals):
            patch.set_height(h)
        views['ax_hist'].set_ylim(0, max(int(hist_vals.max() * 1.2), 100))
        artists['hist_stats_text'].set_text(
            f'mean={all_amps.mean():.1f}  std={all_amps.std():.1f}  '
            f'med={np.median(all_amps):.1f}')

        # Rolling variance heatmap
        var_heat = np.zeros((nbins, ncols), dtype=np.float64)
        sum_sq = np.zeros((nbins, ncols), dtype=np.float64)
        np.add.at(var_heat, bin_idx, data_win[:, :ncols])
        np.add.at(sum_sq, bin_idx, data_win[:, :ncols] ** 2)
        pop_v = counts > 1
        mean_bins = np.zeros_like(var_heat)
        mean_bins[pop_v] = var_heat[pop_v] / counts[pop_v, None]
        var_bins = np.zeros_like(var_heat)
        var_bins[pop_v] = sum_sq[pop_v] / counts[pop_v, None] - mean_bins[pop_v] ** 2
        var_bins = np.clip(var_bins, 0, None)
        artists['var_im'].set_data(var_bins.T)
        artists['var_im'].set_extent([0, win_sec, 0, NUM_SUBCARRIERS])
        views['ax_varheat'].set_xlim(0, win_sec)

        # Subcarrier group sensitivity (bin-averaged in [0, win_sec])
        if len(data_win) > 20:
            grp_bins = 40
            grp_edges = np.linspace(0, win_sec, grp_bins + 1)
            grp_bidx = np.clip(np.digitize(ts_win, grp_edges) - 1, 0, grp_bins - 1)
            grp_centers = (grp_edges[:-1] + grp_edges[1:]) * 0.5
            grp_cnt = np.bincount(grp_bidx, minlength=grp_bins).astype(np.float64)

            for gname, gidx_list in SC_GROUPS.items():
                valid_idx = [i for i in gidx_list if i < ncols]
                if not valid_idx:
                    continue
                flat_mean = data_win[:, valid_idx].mean(axis=1)
                grp_sum = np.zeros(grp_bins, dtype=np.float64)
                grp_sum2 = np.zeros(grp_bins, dtype=np.float64)
                np.add.at(grp_sum, grp_bidx, flat_mean)
                np.add.at(grp_sum2, grp_bidx, flat_mean ** 2)
                pop_g = grp_cnt > 1
                grp_var = np.zeros(grp_bins, dtype=np.float64)
                grp_var[pop_g] = (grp_sum2[pop_g] / grp_cnt[pop_g]
                                  - (grp_sum[pop_g] / grp_cnt[pop_g]) ** 2)
                grp_var = np.clip(grp_var, 0, None)
                for bi in range(1, grp_bins):
                    if grp_cnt[bi] == 0 and grp_cnt[bi - 1] > 0:
                        grp_var[bi] = grp_var[bi - 1]
                artists['sc_group_lines'][gname].set_data(grp_centers, grp_var)

            views['ax_scgroups'].set_xlim(0, win_sec)
            views['ax_scgroups'].set_ylim(0, 50)

            group_vars = {}
            for gname, gidx_list in SC_GROUPS.items():
                valid_idx = [i for i in gidx_list if i < ncols]
                if valid_idx:
                    group_vars[gname] = float(data_win[:, valid_idx].var())
            if group_vars:
                most_sens = max(group_vars, key=group_vars.get)
                uniformity = min(group_vars.values()) / max(max(group_vars.values()), 1e-9)
                artists['scg_info_text'].set_text(
                    f'Most sensitive: {most_sens}  Uniformity: {uniformity:.2f}')

    # ---- Stats text ----
    plot_fps = 0.0
    if len(fps_times) >= 2:
        dt = fps_times[-1] - fps_times[0]
        if dt > 0:
            plot_fps = (len(fps_times) - 1) / dt
    csi_rate = len(data_win) / max(win_sec, 0.01)
    cur_mean = float(bin_mean[pop][-1]) if pop.any() else 0
    cur_std  = float(bin_std[pop][-1])  if pop.any() else 0
    status = 'LIVE' if not _collection_done else 'DONE'
    vtag = 'SIG' if _active_view == 'timeseries' else 'QTY'
    artists['stats_text'].set_text(
        f'[{status}:{vtag}]  Samples: {seq:,}  |  '
        f'CSI: {csi_rate:.0f} pkt/s  |  Plot: {plot_fps:.1f} fps  |  '
        f'Mean: {cur_mean:.1f}  Std: {cur_std:.1f}  |  '
        f'Win: {win_sec:.0f}s  |  T: {t_now:.1f}s')

    fig.canvas.draw_idle()
    return seq, True


def main():
    global _occupation_model, _occupation_window, _occupation_var_window
    global _occupation_model_path

    parser = argparse.ArgumentParser(description="CSI viewer")
    parser.add_argument("--rx-port", required=True)
    parser.add_argument("--tx-port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--occupation-model", type=str, default=None,
                        help="Path to occupation model (.pkl)")

    args = parser.parse_args()

    if args.occupation_model:
        _occupation_model_path = args.occupation_model
        if os.path.exists(args.occupation_model):
            with open(args.occupation_model, 'rb') as f:
                _occupation_model = pickle.load(f)
            _occupation_window = _occupation_model.get('window', 300)
            _occupation_var_window = _occupation_model.get('var_window', 100)
            print(f"[model] Loaded: {args.occupation_model}")
        else:
            print(f"[model] File not found: {args.occupation_model}")

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
    target_dt = 0.033  # ~30 fps target for smooth real-time streaming

    try:
        tk_widget = fig.canvas.get_tk_widget()
    except Exception:
        tk_widget = None

    try:
        while plt.fignum_exists(fig.number):
            t0 = time.time()
            last_seq, _ = update_once(fig, artists, views, last_seq, fps_times)
            elapsed = time.time() - t0
            sleep_t = max(0.002, target_dt - elapsed)
            if tk_widget is not None:
                end_t = time.time() + sleep_t
                while time.time() < end_t:
                    tk_widget.update()
                    time.sleep(0.001)
            else:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        print("\n[info] Interrupted")
    finally:
        stop_event.set()
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
          f"{_snapshot_counter} PDF exports")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: (stop_event.set(), print("\n[info] Stopping...")))
    main()
