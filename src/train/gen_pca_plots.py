#!/usr/bin/env python3
"""
Generate 2D PCA scatter plots for all CSI and reference datasets.
Each subplot shows data projected onto first two principal components,
colored by class label, and annotated with Silhouette score and
Fisher Discriminant Ratio.

Saves a combined figure as PDF and PNG to stats_results/ and the paper
assets/ directory.

Usage:
    python gen_pca_plots.py
    python gen_pca_plots.py --data-root ../../data
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Add train/ to path
_train_dir = os.path.dirname(os.path.abspath(__file__))
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from utils import TrainingDataset, set_global_seed

OUT_DIR = os.path.join(_train_dir, 'stats_results')
ASSETS_DIR = os.path.join(_train_dir, '..', '..', '..', 'newpaper2026', 'assets')


# =============================================================================
# Metrics (same as dataset_stats.py)
# =============================================================================
def compute_silhouette(X, y, sample_size=5000):
    n = X.shape[0]
    if n > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, sample_size, replace=False)
        X, y = X[idx], y[idx]
    if len(np.unique(y)) < 2:
        return float('nan')
    return round(silhouette_score(X, y, sample_size=min(len(y), sample_size),
                                  random_state=42), 3)


def compute_fisher_ratio(X, y):
    classes = np.unique(y)
    n, d = X.shape
    overall_mean = X.mean(axis=0)
    S_W = np.zeros((d, d))
    S_B = np.zeros((d, d))
    for c in classes:
        X_c = X[y == c]
        n_c = X_c.shape[0]
        mean_c = X_c.mean(axis=0)
        diff_w = X_c - mean_c
        S_W += diff_w.T @ diff_w
        diff_b = (mean_c - overall_mean).reshape(-1, 1)
        S_B += n_c * (diff_b @ diff_b.T)
    tr_sw = np.trace(S_W)
    tr_sb = np.trace(S_B)
    if tr_sw < 1e-12:
        return float('inf')
    return round(tr_sb / tr_sw, 3)


# =============================================================================
# Reference dataset loaders
# =============================================================================
def load_iris():
    from sklearn.datasets import load_iris as _load
    data = _load()
    label_names = {i: n for i, n in enumerate(data.target_names)}
    return data.data, data.target, label_names, 'Iris'


def load_mnist(n_samples=5000):
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data.astype(np.float64), mnist.target.astype(int)
    except Exception:
        try:
            from torchvision import datasets
            ds = datasets.MNIST(root='/tmp/mnist', train=True, download=True)
            X = ds.data.numpy().reshape(-1, 784).astype(np.float64)
            y = ds.targets.numpy()
        except Exception as e:
            print(f"  [warn] Cannot load MNIST: {e}")
            return None, None, None, 'MNIST'
    rng = np.random.RandomState(42)
    idx = rng.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
    label_names = {i: str(i) for i in range(10)}
    return X[idx], y[idx], label_names, 'MNIST'


def load_cifar10(n_samples=5000):
    try:
        from torchvision import datasets
        ds = datasets.CIFAR10(root='/tmp/cifar10', train=True, download=True)
        X = np.array(ds.data).reshape(-1, 3072).astype(np.float64)
        y = np.array(ds.targets)
    except Exception:
        try:
            from sklearn.datasets import fetch_openml
            cifar = fetch_openml('CIFAR_10', version=1, as_frame=False, parser='auto')
            X = cifar.data.astype(np.float64)
            y = cifar.target.astype(int)
        except Exception as e:
            print(f"  [warn] Cannot load CIFAR-10: {e}")
            return None, None, None, 'CIFAR-10'
    rng = np.random.RandomState(42)
    idx = rng.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
    cifar10_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                     4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                     8: 'ship', 9: 'truck'}
    return X[idx], y[idx], cifar10_names, 'CIFAR-10'


def load_cifar100(n_samples=5000):
    try:
        from torchvision import datasets
        ds = datasets.CIFAR100(root='/tmp/cifar100', train=True, download=True)
        X = np.array(ds.data).reshape(-1, 3072).astype(np.float64)
        y = np.array(ds.targets)
    except Exception:
        try:
            from sklearn.datasets import fetch_openml
            cifar = fetch_openml('CIFAR_100', version=1, as_frame=False, parser='auto')
            X = cifar.data.astype(np.float64)
            y = cifar.target.astype(int)
        except Exception as e:
            print(f"  [warn] Cannot load CIFAR-100: {e}")
            return None, None, None, 'CIFAR-100'
    rng = np.random.RandomState(42)
    idx = rng.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
    label_names = {i: str(i) for i in range(100)}
    return X[idx], y[idx], label_names, 'CIFAR-100'


# =============================================================================
# Annotation helper
# =============================================================================
def sil_quality(sil):
    """Return a qualitative label for silhouette score (poor/fair/strong)."""
    if sil >= 0.5:
        return 'strong'
    elif sil >= 0.0:
        return 'fair'
    else:
        return 'poor'


def fisher_quality(fisher):
    """Return a qualitative label for Fisher ratio (poor/fair/strong)."""
    if fisher >= 1:
        return 'strong'
    elif fisher >= 0.1:
        return 'fair'
    else:
        return 'poor'


def annotation_color(sil):
    """Badge background color based on silhouette quality (poor/fair/strong)."""
    if sil >= 0.5:
        return '#2ca02c'   # green (strong)
    elif sil >= 0.0:
        return '#ff7f0e'   # orange (fair)
    else:
        return '#d62728'   # red (poor)


# =============================================================================
# Plotting
# =============================================================================
def plot_pca_2d(ax, X_pca, y, label_names, title, sil, fisher, var_ratio, is_individual=False):
    """Plot 2D PCA scatter on a given axes with smart annotations.
    
    Publication-quality styling with clean aesthetics.
    """
    classes = np.unique(y)
    n_cls = len(classes)

    # Publication-quality colormap
    if n_cls <= 10:
        cmap = plt.cm.get_cmap('Set1', max(n_cls, 3))
    else:
        cmap = plt.cm.get_cmap('tab20', n_cls)

    # Adjust sizes based on individual vs combined plot
    marker_size = 12 if is_individual else 8
    alpha_val = 0.6 if is_individual else 0.5
    font_scale = 1.2 if is_individual else 1.0

    for i, c in enumerate(classes):
        mask = y == c
        lbl = label_names.get(c, str(c)) if label_names else str(c)
        # Subsample for plotting if too many points
        idx = np.where(mask)[0]
        if len(idx) > 1500:
            rng = np.random.RandomState(42)
            idx = rng.choice(idx, 1500, replace=False)
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                   c=[cmap(i % cmap.N)], s=marker_size, alpha=alpha_val,
                   label=lbl, edgecolors='none', rasterized=False)

    # Axis labels with variance explained - publication styling
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)', fontsize=int(10 * font_scale), fontweight='medium')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)', fontsize=int(10 * font_scale), fontweight='medium')
    ax.set_title(title, fontsize=int(12 * font_scale), fontweight='bold', pad=12)
    ax.tick_params(labelsize=int(9 * font_scale), width=1.2)
    
    # Clean spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')

    # Clean annotation badge with numeric values only
    sil_str = f"Sil = {sil:+.2f}" if sil < 0 else f"Sil = {sil:.2f}"
    fisher_str = f"FDR = {fisher:.2f}"
    badge_col = annotation_color(sil)

    annotation_text = f"{sil_str}\n{fisher_str}"

    bbox_props = dict(boxstyle="round,pad=0.5", facecolor=badge_col,
                      edgecolor='white', alpha=0.9, linewidth=1.5)
    ax.text(0.03, 0.97, annotation_text, transform=ax.transAxes,
            fontsize=int(9 * font_scale), verticalalignment='top', fontfamily='sans-serif',
            color='white', fontweight='bold', bbox=bbox_props)

    # Legend — only if <= 10 classes, with better styling
    if n_cls <= 10:
        leg = ax.legend(fontsize=int(8 * font_scale), loc='lower right', framealpha=0.95,
                        markerscale=1.8, handletextpad=0.4,
                        borderpad=0.5, labelspacing=0.3, fancybox=True,
                        edgecolor='#cccccc')
        leg.get_frame().set_linewidth(1.0)


def generate_all_plots(data_root, n_pca_full=50, window_len=500,
                       guaranteed_sr=150, stride=100):
    """Load all datasets, compute 2D PCA projections, create figure."""
    set_global_seed(42)

    datasets = []  # list of (title, X_pca, y, label_names, sil, fisher, var_ratio)

    # --- CSI Datasets ---
    dataset_dirs = [
        ('home_har_data', 'Home HAR'),
        ('home_occupation_data', 'Home Occupation'),
        ('office_har_data', 'Office HAR'),
        ('office_localization_data', 'Office Localization'),
    ]

    for dname, display_name in dataset_dirs:
        dpath = os.path.join(data_root, dname)
        meta_path = os.path.join(dpath, 'dataset_metadata.json')
        if not os.path.isfile(meta_path):
            print(f"  [skip] {dname} -- no metadata")
            continue
        ds_stride = stride
        if dname == 'home_har_data':
            ds_stride = max(stride, window_len)
        try:
            train_ds, _ = TrainingDataset.from_metadata(
                root_dir=dpath, pipeline_name='amplitude',
                window_len=window_len, guaranteed_sr=guaranteed_sr,
                mode='flattened', stride=ds_stride, balance=True, verbose=False)
        except Exception as e:
            print(f"  [error] {dname}: {e}")
            continue

        X, y = train_ds.X, train_ds.y
        print(f"  {display_name}: {X.shape}, {train_ds.num_classes} classes")

        # Full PCA for metrics
        n_comp_full = min(n_pca_full, X.shape[1], X.shape[0])
        pca_full = PCA(n_components=n_comp_full)
        X_pca_full = pca_full.fit_transform(X)
        sil = compute_silhouette(X_pca_full, y)
        fisher = compute_fisher_ratio(X_pca_full, y)

        # 2-component PCA for plotting
        pca2 = PCA(n_components=2)
        X_pca2 = pca2.fit_transform(X)
        var_ratio = pca2.explained_variance_ratio_

        label_names = {v: k for k, v in train_ds.label_map.items()}
        datasets.append((display_name, X_pca2, y, label_names, sil, fisher, var_ratio))
        print(f"    Sil={sil}, Fisher={fisher}")

    # --- Reference Datasets ---
    ref_loaders = [load_iris, load_mnist, load_cifar10, load_cifar100]

    for loader_fn in ref_loaders:
        X, y, label_names, name = loader_fn()
        if X is None:
            print(f"  [skip] {name}")
            continue
        print(f"  {name}: {X.shape}, {len(np.unique(y))} classes")

        n_comp_full = min(n_pca_full, X.shape[1], X.shape[0])
        pca_full = PCA(n_components=n_comp_full)
        X_pca_full = pca_full.fit_transform(X)
        sil = compute_silhouette(X_pca_full, y)
        fisher = compute_fisher_ratio(X_pca_full, y)

        pca2 = PCA(n_components=2)
        X_pca2 = pca2.fit_transform(X)
        var_ratio = pca2.explained_variance_ratio_

        datasets.append((name, X_pca2, y, label_names, sil, fisher, var_ratio))
        print(f"    Sil={sil}, Fisher={fisher}")

    # --- Create figure ---
    n = len(datasets)
    if n == 0:
        print("[ERROR] No datasets loaded.")
        return

    # Publication-quality settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 1.2,
        'axes.labelweight': 'medium',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,  # TrueType fonts for better compatibility
        'ps.fonttype': 42,
    })

    # Layout: 2 columns, ceil(n/2) rows (single-column paper figure)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0, 3.5 * nrows),
                              constrained_layout=True, dpi=300)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, (title, X_pca2, y, label_names, sil, fisher, var_ratio) in enumerate(datasets):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        plot_pca_2d(ax, X_pca2, y, label_names, title, sil, fisher, var_ratio, is_individual=False)

    # Hide unused axes
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    # Save combined figure - PDF only for publication
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # Save as high-quality PDF
    path_stats = os.path.join(OUT_DIR, 'pca_2d_scatter.pdf')
    fig.savefig(path_stats, format='pdf', bbox_inches='tight', dpi=300)
    print(f"  Saved {path_stats}")

    path_assets = os.path.join(ASSETS_DIR, 'pca_2d_scatter.pdf')
    fig.savefig(path_assets, format='pdf', bbox_inches='tight', dpi=300)
    print(f"  Saved {path_assets}")

    # Also save PNG for quick preview
    path_stats_png = os.path.join(OUT_DIR, 'pca_2d_scatter.png')
    fig.savefig(path_stats_png, format='png', bbox_inches='tight', dpi=300)
    print(f"  Saved {path_stats_png}")

    path_assets_png = os.path.join(ASSETS_DIR, 'pca_2d_scatter.png')
    fig.savefig(path_assets_png, format='png', bbox_inches='tight', dpi=300)
    print(f"  Saved {path_assets_png}")

    plt.close(fig)

    # --- Generate individual plots for each dataset ---
    print("\nGenerating individual publication-quality plots...")
    for title, X_pca2, y, label_names, sil, fisher, var_ratio in datasets:
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=(6.0, 5.0), dpi=300)
        plot_pca_2d(ax_ind, X_pca2, y, label_names, title, sil, fisher, var_ratio, is_individual=True)
        fig_ind.tight_layout()
        
        # Create safe filename from title
        safe_name = title.lower().replace(' ', '_').replace('-', '_')
        
        # Save as high-quality PDF (primary format)
        path_stats = os.path.join(OUT_DIR, f'pca_{safe_name}.pdf')
        fig_ind.savefig(path_stats, format='pdf', bbox_inches='tight', dpi=300)
        
        path_assets = os.path.join(ASSETS_DIR, f'pca_{safe_name}.pdf')
        fig_ind.savefig(path_assets, format='pdf', bbox_inches='tight', dpi=300)
        
        # Also save PNG for quick preview
        path_stats_png = os.path.join(OUT_DIR, f'pca_{safe_name}.png')
        fig_ind.savefig(path_stats_png, format='png', bbox_inches='tight', dpi=300)
        
        path_assets_png = os.path.join(ASSETS_DIR, f'pca_{safe_name}.png')
        fig_ind.savefig(path_assets_png, format='png', bbox_inches='tight', dpi=300)
        
        plt.close(fig_ind)
        print(f"  Saved: pca_{safe_name}.pdf/.png")

    print(f"\nDone — {n} datasets plotted (combined + individual, PDF + PNG).")


# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 2D PCA scatter plots for dataset characterization')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', 'data'),
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--pca-components', type=int, default=50)
    parser.add_argument('--window', type=int, default=500)
    parser.add_argument('--sr', type=int, default=150)
    parser.add_argument('--stride', type=int, default=100)
    args = parser.parse_args()

    generate_all_plots(
        data_root=os.path.abspath(args.data_root),
        n_pca_full=args.pca_components,
        window_len=args.window,
        guaranteed_sr=args.sr,
        stride=args.stride)
