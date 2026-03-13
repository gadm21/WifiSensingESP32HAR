#!/usr/bin/env python3
"""
PCA training on CSI data from multiple environments.

Loads CSI files from 4 dataset folders, labels each by environment,
fits a PCA model on windowed magnitude vectors, plots 2D PCA scatter,
and saves the fitted PCA (mean, components) for live-stream inference.

Usage:
    python pca_train.py                     # default paths
    python pca_train.py --data-root ../../wifi_sensing_data
    python pca_train.py --window 200 --sr 150
"""

import argparse
import os
import sys
import glob
import time
import numpy as np
import pickle
from collections import Counter

# Add train/ to path for CSI_Loader and utils
_train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train')
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from utils import CSI_Loader, FeatureSelector, CSI_SUBCARRIER_MASK, TrainingDataset

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
# Maps folder name -> environment ('home' or 'office')
DATASET_FOLDERS = {
    'home_har_data':            'home',
    'home_occupation_data':     'home',
    'office_har_data':          'office',
    'office_localization_data': 'office',
}

# Location labels
LOCATIONS = ['home', 'office']

# Activity labels (extracted from filename stem)
ACTIVITIES = ['empty', 'drink', 'eat', 'sleep', 'smoke', 'watch', 'work']
VALID_ACTIVITIES = set(ACTIVITIES)

# Combined label = location_activity  (used for plotting)
LABELS = [f'{loc}_{act}' for loc in LOCATIONS for act in ACTIVITIES]

# Colors: location-based hue, activity shifts brightness
LOCATION_COLORS = {'home': '#2979ff', 'office': '#ffea00'}
LOCATION_MARKERS = {'home': 'o', 'office': '^'}

# Legacy combined colors for the scatter plot
LABEL_COLORS = {
    'home_empty':   '#2979ff', 'home_drink':  '#42a5f5', 'home_eat':     '#64b5f6',
    'home_sleep':   '#90caf9', 'home_smoke':  '#1565c0', 'home_watch':   '#0d47a1',
    'home_work':    '#1e88e5',
    'office_empty': '#00e5ff', 'office_drink': '#26c6da', 'office_eat':  '#ffea00',
    'office_sleep': '#b2ebf2', 'office_smoke': '#ffc107', 'office_watch': '#ff6d00',
    'office_work':  '#ff9800',
}

LABEL_MARKERS = {
    'home_empty': 'o', 'home_drink': 'o', 'home_eat': 'o',
    'home_sleep': 'o', 'home_smoke': 'o', 'home_watch': 'o', 'home_work': 'o',
    'office_empty': '^', 'office_drink': '^', 'office_eat': '^',
    'office_sleep': '^', 'office_smoke': '^', 'office_watch': '^', 'office_work': '^',
}


def _classify_file(env, filename):
    """Extract location and activity from environment + filename.

    Returns (location, activity, combined_label).
    location: 'home' or 'office'
    activity: e.g. 'empty', 'eat', 'work', 'five_two', etc.
    combined: e.g. 'home_eat', 'office_empty'
    """
    base = os.path.basename(filename).lower().replace('.csv', '')
    # Strip trailing numeric suffixes: eat_1 -> eat, five_two_2 -> five_two
    # Strategy: try known activity names longest-first
    known = sorted(ACTIVITIES, key=len, reverse=True)
    activity = None
    for act in known:
        if base == act or base.startswith(act + '_') or base.startswith(act):
            # Verify the remainder (after act) is empty or _<digits>
            remainder = base[len(act):]
            if remainder == '' or (remainder[0] in ('_', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9') and
                                   remainder.lstrip('_').replace('_', '').isdigit()):
                activity = act
                break
            # Also match e.g. 'drink2' -> 'drink'
            if remainder.isdigit():
                activity = act
                break
    if activity is None:
        activity = base.rstrip('0123456789_')  # fallback
        if not activity:
            activity = 'unknown'
    location = env  # 'home' or 'office'
    combined = f'{location}_{activity}'
    return location, activity, combined


def find_csv_files(folder):
    """Recursively find all .csv files in a folder."""
    csvs = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith('.csv'):
                csvs.append(os.path.join(root, f))
    return sorted(csvs)


# Use the canonical implementation from utils to avoid duplication
_rolling_variance = TrainingDataset._rolling_variance


def load_all_datasets(data_root, guaranteed_sr=100, window=100, verbose=False,
                      var_window=20):
    """Load CSI from all dataset folders, window into fixed-length segments.

    Parameters
    ----------
    var_window : int
        Rolling variance window size.  For each sample, the variance of the
        preceding *var_window* samples is computed.  0 or 1 = no transform.

    Returns
    -------
    X : np.ndarray, shape (N_windows, 52 * window)
        Flattened windowed magnitude vectors.
    locations : list[str]
        Location label ('home' or 'office') for each window.
    activities : list[str]
        Activity label ('empty', 'eat', 'work', etc.) for each window.
    combined : list[str]
        Combined label ('home_eat', 'office_empty', etc.) for each window.
    file_info : list[str]
        Source file for each window.
    """
    loader = CSI_Loader(verbose=verbose)
    loader.guaranteed_sr = guaranteed_sr
    selector = FeatureSelector(mask=CSI_SUBCARRIER_MASK, verbose=False)

    all_windows = []
    all_locations = []   # 'home' or 'office'
    all_activities = []  # 'empty', 'eat', 'work', etc.
    all_combined = []    # 'home_eat', 'office_empty', etc.
    all_files = []

    for folder_name, env in DATASET_FOLDERS.items():
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            print(f"[warn] Folder not found, skipping: {folder_path}")
            continue

        csv_files = find_csv_files(folder_path)
        print(f"\n[{env}] Found {len(csv_files)} CSV files in {folder_name}/")

        for csv_path in csv_files:
            fname = os.path.relpath(csv_path, data_root)
            location, activity, combined = _classify_file(env, csv_path)

            # Filter: exclude files whose activity is not in the valid set
            if activity not in VALID_ACTIVITIES:
                print(f"  EXCLUDE {fname}  location={location}  activity={activity}  "
                      f"(not in {sorted(VALID_ACTIVITIES)})")
                continue

            try:
                data = loader.process(csv_path)
                data = selector.process(data)
                mag = data['mag']  # (N_samples, 52)

                # Rolling variance transform
                if var_window > 1:
                    mag = _rolling_variance(mag, var_window)

                if mag.shape[0] < window:
                    print(f"  SKIP    {fname}  location={location}  activity={activity}  "
                          f"only {mag.shape[0]} samples (need {window})")
                    continue

                # Slide non-overlapping windows
                n_windows = mag.shape[0] // window
                for wi in range(n_windows):
                    chunk = mag[wi * window : (wi + 1) * window, :]  # (window, 52)
                    all_windows.append(chunk.ravel())  # flatten to (52*window,)
                    all_locations.append(location)
                    all_activities.append(activity)
                    all_combined.append(combined)
                    all_files.append(fname)

                print(f"  OK      {fname}  location={location}  activity={activity}  "
                      f"{mag.shape[0]} samples -> {n_windows} windows")

            except Exception as e:
                print(f"  ERROR   {fname}: {e}")
                continue

    if not all_windows:
        print("\n[error] No data loaded!")
        sys.exit(1)

    X = np.array(all_windows, dtype=np.float64)
    print(f"\n[summary] Total windows: {X.shape[0]}, Feature dim: {X.shape[1]}")
    # Print counts per combined label
    combined_counts = Counter(all_combined)
    for lbl in sorted(combined_counts):
        print(f"  {lbl}: {combined_counts[lbl]} windows")

    return X, all_locations, all_activities, all_combined, all_files


def fit_pca(X, n_components=2):
    """Fit PCA via SVD. Returns projected data, mean, components, explained variance."""
    mean = X.mean(axis=0)
    X_centered = X - mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]  # (n_components, n_features)
    projected = X_centered @ components.T  # (N, n_components)

    var_explained = (S ** 2) / (S ** 2).sum() * 100

    return projected, mean, components, var_explained


def plot_pca_3d(projected, labels, var_explained, save_path=None):
    """Plot a publication-quality 3D PCA scatter colored by label.

    Applies percentile-based outlier clipping so extreme points don't
    squash the main cluster into a corner.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.colors as mcolors

    # ── highly-distinct colour palette ───────────────────────────────
    _PALETTE = {
        'home_empty':   '#2196f3',   # blue
        'home_drink':   '#e91e63',   # pink
        'home_eat':     '#4caf50',   # green
        'home_sleep':   '#9c27b0',   # purple
        'home_smoke':   '#ff5722',   # deep orange
        'home_watch':   '#00bcd4',   # cyan
        'home_work':    '#8bc34a',   # light green
        'office_empty': '#ffc107',   # amber
        'office_drink': '#ff9800',   # orange
        'office_eat':   '#cddc39',   # lime
        'office_sleep': '#607d8b',   # blue-grey
        'office_smoke': '#795548',   # brown
        'office_watch': '#f44336',   # red
        'office_work':  '#ffeb3b',   # yellow
    }

    # ── theme ────────────────────────────────────────────────────────
    BG      = '#0d1117'
    PANE    = '#161b22'
    GRID_C  = '#30363d'
    TEXT_C  = '#c9d1d9'
    ACCENT  = '#58a6ff'

    labels_arr = np.array(labels)
    unique_labels = sorted(set(labels))
    n_comp = min(projected.shape[1], 3)

    # ── percentile clipping (remove extreme outliers) ────────────────
    lo_pct, hi_pct = 1.0, 99.0
    keep = np.ones(len(projected), dtype=bool)
    for d in range(n_comp):
        lo, hi = np.percentile(projected[:, d], [lo_pct, hi_pct])
        keep &= (projected[:, d] >= lo) & (projected[:, d] <= hi)
    proj = projected[keep]
    labs = labels_arr[keep]
    n_removed = len(projected) - keep.sum()
    if n_removed > 0:
        print(f"[plot] Clipped {n_removed} outlier points "
              f"({lo_pct:.0f}–{hi_pct:.0f} percentile)")

    # ── figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    ax = fig.add_subplot(111, projection='3d', facecolor=BG)

    # ── scatter each class ───────────────────────────────────────────
    for lbl in unique_labels:
        mask = labs == lbl
        n = mask.sum()
        if n == 0:
            continue
        hex_c = _PALETTE.get(lbl, '#ffffff')
        marker = '^' if lbl.startswith('office') else 'o'
        zs = proj[mask, 2] if n_comp >= 3 else np.zeros(n)
        ax.scatter(proj[mask, 0], proj[mask, 1], zs,
                   c=hex_c, marker=marker, s=12, alpha=0.65,
                   edgecolors='none', depthshade=True,
                   label=f'{lbl}  ({n})')

    # ── pane + grid styling ──────────────────────────────────────────
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(PANE)
        pane.set_edgecolor(GRID_C)
        pane.set_alpha(0.40)
    ax.grid(True, color=GRID_C, alpha=0.20, linewidth=0.4)

    # ── axis labels ──────────────────────────────────────────────────
    lbl_kw = dict(fontsize=11, color=TEXT_C, fontweight='medium', labelpad=8)
    ax.set_xlabel(f'PC 1  ({var_explained[0]:.1f}%)', **lbl_kw)
    ax.set_ylabel(f'PC 2  ({var_explained[1]:.1f}%)', **lbl_kw)
    if n_comp >= 3:
        ax.set_zlabel(f'PC 3  ({var_explained[2]:.1f}%)', **lbl_kw)
    ax.tick_params(colors='#8b949e', labelsize=7, pad=1, length=3)

    # ── robust axis limits from clipped data ─────────────────────────
    pad_frac = 0.08
    for d, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        if d >= n_comp:
            break
        lo, hi = proj[:, d].min(), proj[:, d].max()
        margin = (hi - lo) * pad_frac
        setter(lo - margin, hi + margin)

    # ── title ────────────────────────────────────────────────────────
    total_var = sum(var_explained[:n_comp])
    ax.set_title(
        f'CSI WiFi Sensing — 3D PCA   ({total_var:.1f}% variance captured)',
        fontsize=15, fontweight='bold', color=ACCENT, pad=16, loc='left')

    # ── legend ───────────────────────────────────────────────────────
    leg = ax.legend(
        fontsize=7, framealpha=0.6, facecolor=PANE, edgecolor=GRID_C,
        loc='upper left', ncol=2, columnspacing=0.8, handletextpad=0.3,
        borderpad=0.5, labelspacing=0.30, markerscale=1.6)
    for txt in leg.get_texts():
        txt.set_color(TEXT_C)

    # ── info box ─────────────────────────────────────────────────────
    n_total = len(labels)
    info = (f'Samples: {n_total:,}  ({n_removed} clipped)\n'
            f'Classes: {len(unique_labels)}\n'
            f'Var: {var_explained[0]:.1f} + {var_explained[1]:.1f}'
            + (f' + {var_explained[2]:.1f}%' if n_comp >= 3 else '%'))
    ax.text2D(0.98, 0.04, info, transform=ax.transAxes,
              fontsize=7.5, color='#8b949e', family='monospace',
              va='bottom', ha='right',
              bbox=dict(boxstyle='round,pad=0.35', facecolor=PANE,
                        edgecolor=GRID_C, alpha=0.75))

    # ── camera ───────────────────────────────────────────────────────
    ax.view_init(elev=25, azim=-55)
    ax.dist = 10.2

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=BG,
                    bbox_inches='tight', pad_inches=0.25)
        print(f"[plot] Saved to {save_path}")

    plt.show()


def save_pca_model(mean, components, var_explained, config, save_path):
    """Save PCA model for live inference.

    Saved dict keys:
        mean : (n_features,) array — subtract before projecting
        components : (n_components, n_features) — project via X @ components.T
        var_explained : array of % variance per component
        config : dict with window, guaranteed_sr, n_subcarriers
        labels : list of label names used in training
    """
    model = {
        'mean': mean,
        'components': components,
        'var_explained': var_explained,
        'config': config,
    }
    # training_samples and training_labels are added by main() before saving
    with open(save_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[model] Saved PCA model to {save_path}")
    print(f"        To load: model = pickle.load(open('{save_path}', 'rb'))")
    print(f"        Project: pc = (window_vec - model['mean']) @ model['components'].T")


def main():
    parser = argparse.ArgumentParser(description='Train PCA on CSI environment data')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', '..', 'wifi_sensing_data'),
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--window', type=int, default=300,
                        help='PCA window size in samples (default: 300)')
    parser.add_argument('--sr', type=int, default=150,
                        help='Guaranteed sample rate for CSI_Loader (default: 150)')
    parser.add_argument('--n-components', type=int, default=3,
                        help='Number of PCA components (default: 3)')
    parser.add_argument('--output', type=str, default='pca_model.pkl',
                        help='Output path for saved PCA model')
    parser.add_argument('--plot-output', type=str, default='pca_scatter.png',
                        help='Output path for PCA scatter plot image')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose CSI loading output')
    parser.add_argument('--max-per-class', type=int, default=0,
                        help='Max windows per combined class for balancing '
                             '(0 = no limit, auto-balance to smallest class)')
    parser.add_argument('--balance', action='store_true',
                        help='Undersample all classes to match the smallest class')
    parser.add_argument('--traj-len', type=int, default=10,
                        help='Trajectory chunk length in consecutive PCA points (default: 10)')
    parser.add_argument('--traj-stride', type=int, default=5,
                        help='Stride between trajectory chunks (default: 5)')
    parser.add_argument('--dtw-radius', type=int, default=3,
                        help='Sakoe-Chiba band radius for DTW (default: 3)')
    parser.add_argument('--var-window', type=int, default=100,
                        help='Rolling variance window size in samples (default: 20, 0=off)')

    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    print(f"[info] Data root: {data_root}")
    print(f"[info] Window: {args.window} samples @ {args.sr} Hz")
    print(f"[info] PCA components: {args.n_components}")
    print(f"[info] Traj chunk: len={args.traj_len}, stride={args.traj_stride}, "
          f"DTW radius={args.dtw_radius}")
    var_win = args.var_window
    if var_win > 1:
        print(f"[info] Rolling variance: window={var_win} samples")
    else:
        print(f"[info] Rolling variance: OFF")
    if args.balance:
        print(f"[info] Class balancing: ON (auto to smallest class)")
    elif args.max_per_class > 0:
        print(f"[info] Class balancing: max {args.max_per_class} per combined class")
    print()

    # Verify data root
    if not os.path.isdir(data_root):
        print(f"[error] Data root not found: {data_root}")
        sys.exit(1)

    t0 = time.time()

    # Load all datasets
    X, locations, activities, combined, file_info = load_all_datasets(
        data_root, guaranteed_sr=args.sr, window=args.window, verbose=args.verbose,
        var_window=var_win)

    # ----- Class balancing -----
    combined_arr = np.array(combined)
    locations_arr = np.array(locations)
    activities_arr = np.array(activities)
    file_arr = np.array(file_info)
    rng = np.random.default_rng(42)

    class_counts = Counter(combined)
    if args.balance or args.max_per_class > 0:
        if args.balance:
            cap = min(class_counts.values())
            print(f"\n[balance] Auto-balancing to smallest class: {cap} windows/class")
        else:
            cap = args.max_per_class
            print(f"\n[balance] Capping each class to {cap} windows")

        keep_bal = []
        for lbl in sorted(class_counts):
            idxs = np.where(combined_arr == lbl)[0]
            if len(idxs) > cap:
                idxs = np.sort(rng.choice(idxs, cap, replace=False))
            keep_bal.append(idxs)
        keep_bal = np.concatenate(keep_bal)
        keep_bal.sort()

        X = X[keep_bal]
        combined = combined_arr[keep_bal].tolist()
        locations = locations_arr[keep_bal].tolist()
        activities = activities_arr[keep_bal].tolist()
        file_info = file_arr[keep_bal].tolist()
        combined_arr = np.array(combined)
        locations_arr = np.array(locations)
        activities_arr = np.array(activities)
        file_arr = np.array(file_info)

        print(f"[balance] After balancing: {X.shape[0]} windows")
        for lbl in sorted(set(combined)):
            print(f"  {lbl}: {sum(1 for c in combined if c == lbl)}")

    # ----- Fit PCA -----
    print(f"\n[pca] Fitting PCA with {args.n_components} components on {X.shape}...")
    projected, mean, components, var_explained = fit_pca(X, n_components=args.n_components)

    print(f"[pca] Variance explained: {', '.join(f'PC{i+1}={v:.1f}%' for i, v in enumerate(var_explained[:args.n_components]))}")
    print(f"[pca] Total time: {time.time() - t0:.1f}s")

    # ----- Build trajectory chunks per file -----
    traj_len = args.traj_len
    traj_stride = args.traj_stride
    traj_chunks_pts = []    # list of (traj_len, 2) arrays
    traj_chunks_loc = []    # location per chunk
    traj_chunks_act = []    # activity per chunk
    traj_chunks_comb = []   # combined label per chunk

    unique_files = list(dict.fromkeys(file_info))  # preserves order, unique
    for fname in unique_files:
        mask = file_arr == fname
        idxs = np.where(mask)[0]
        traj_pts = projected[idxs]  # (T_i, 2) consecutive PCA points
        loc = locations_arr[idxs[0]]
        act = activities_arr[idxs[0]]
        comb = combined_arr[idxs[0]]
        for start in range(0, len(traj_pts) - traj_len + 1, traj_stride):
            traj_chunks_pts.append(traj_pts[start:start + traj_len].astype(np.float32))
            traj_chunks_loc.append(str(loc))
            traj_chunks_act.append(str(act))
            traj_chunks_comb.append(str(comb))

    print(f"\n[traj] Built {len(traj_chunks_pts)} trajectory chunks "
          f"(len={traj_len}, stride={traj_stride}) from {len(unique_files)} files")
    traj_comb_counts = Counter(traj_chunks_comb)
    for lbl in sorted(traj_comb_counts):
        print(f"  {lbl}: {traj_comb_counts[lbl]} chunks")

    # ----- Save model -----
    config = {
        'window': args.window,
        'guaranteed_sr': args.sr,
        'n_subcarriers': 52,
        'n_features': X.shape[1],
        'n_components': args.n_components,
        'labels': list(DATASET_FOLDERS.values()),
        'label_colors': LABEL_COLORS,
        'label_markers': LABEL_MARKERS,
        'traj_len': traj_len,
        'traj_stride': traj_stride,
        'dtw_radius': args.dtw_radius,
        'var_window': var_win,
    }
    # Compute axis limits with padding
    pad_frac = 0.05
    n_comp = projected.shape[1]
    x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
    y_min, y_max = projected[:, 1].min(), projected[:, 1].max()
    x_pad = (x_max - x_min) * pad_frac
    y_pad = (y_max - y_min) * pad_frac
    ax_limits = {
        'xlim': (float(x_min - x_pad), float(x_max + x_pad)),
        'ylim': (float(y_min - y_pad), float(y_max + y_pad)),
    }
    if n_comp >= 3:
        z_min, z_max = projected[:, 2].min(), projected[:, 2].max()
        z_pad = (z_max - z_min) * pad_frac
        ax_limits['zlim'] = (float(z_min - z_pad), float(z_max + z_pad))

    # Downsample training samples for the background scatter
    # (keep at most ~2000 per combined label to keep pkl size reasonable)
    max_per_label = 2000
    keep_idx = []
    for lbl in sorted(set(combined)):
        idxs = np.where(combined_arr == lbl)[0]
        if len(idxs) > max_per_label:
            idxs = np.sort(rng.choice(idxs, max_per_label, replace=False))
        keep_idx.append(idxs)
    keep_idx = np.concatenate(keep_idx)
    bg_projected = projected[keep_idx].astype(np.float32)
    bg_combined = combined_arr[keep_idx].tolist()
    bg_locations = locations_arr[keep_idx].tolist()
    bg_activities = activities_arr[keep_idx].tolist()

    print(f"\n[model] Background scatter: {len(bg_combined)} points "
          f"(downsampled from {len(combined)})")
    print(f"[model] Axis limits: x={ax_limits['xlim']}, y={ax_limits['ylim']}")

    save_pca_model(mean, components, var_explained, config, args.output)

    # Inject training samples + trajectory chunks into the saved model
    with open(args.output, 'rb') as f:
        model = pickle.load(f)
    model['training_projected'] = bg_projected
    model['training_labels'] = bg_combined       # combined labels (backward compat)
    model['training_locations'] = bg_locations   # 'home' / 'office'
    model['training_activities'] = bg_activities # 'empty', 'eat', 'work', etc.
    model['ax_limits'] = ax_limits
    # Trajectory chunks for DTW
    model['traj_chunks_pts'] = traj_chunks_pts     # list of (traj_len, 2) float32
    model['traj_chunks_loc'] = traj_chunks_loc     # list of str
    model['traj_chunks_act'] = traj_chunks_act     # list of str
    model['traj_chunks_comb'] = traj_chunks_comb   # list of str
    model['traj_len'] = traj_len
    model['dtw_radius'] = args.dtw_radius
    with open(args.output, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[model] Updated {args.output} with training samples + trajectories")
    print(f"[model] {len(set(bg_locations))} locations, {len(set(bg_activities))} activities, "
          f"{len(traj_chunks_pts)} traj chunks")

    # Plot
    plot_pca_3d(projected, combined, var_explained, save_path=args.plot_output)


if __name__ == '__main__':
    main()
