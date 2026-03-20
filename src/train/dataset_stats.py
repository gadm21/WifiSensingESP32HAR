#!/usr/bin/env python3
"""
Compute dataset characterization statistics for CSI datasets and reference
benchmarks (Iris, MNIST, CIFAR-10, CIFAR-100).

Outputs Silhouette score, Fisher Discriminant Ratio, number of classes,
and number of samples for each dataset. Results are saved to CSV and printed
in LaTeX table format suitable for the paper.

Usage:
    python dataset_stats.py
    python dataset_stats.py --data-root ../../data --pca-components 50
"""

import argparse
import os
import sys
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Add train/ to path
_train_dir = os.path.dirname(os.path.abspath(__file__))
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from utils import TrainingDataset, set_global_seed


# =============================================================================
# Metrics
# =============================================================================
def compute_silhouette(X, y, sample_size=5000):
    """Compute mean silhouette score, subsampling if needed."""
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
    """Compute Fisher Discriminant Ratio = tr(S_B) / tr(S_W)."""
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
# Reference Datasets
# =============================================================================
def load_iris():
    """Load Iris dataset."""
    from sklearn.datasets import load_iris
    data = load_iris()
    return data.data, data.target, 3


def load_mnist(n_samples=10000):
    """Load MNIST dataset (subset for speed)."""
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
            return None, None, 10
    if X.shape[0] > n_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], n_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y, 10


def load_cifar10(n_samples=10000):
    """Load CIFAR-10 dataset (subset for speed)."""
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
            return None, None, 10
    if X.shape[0] > n_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], n_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y, 10


def load_cifar100(n_samples=10000):
    """Load CIFAR-100 dataset (subset for speed)."""
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
            return None, None, 100
    if X.shape[0] > n_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], n_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y, 100


# =============================================================================
# Main computation
# =============================================================================
def compute_all_stats(data_root, n_pca=50, window_len=500, guaranteed_sr=150,
                      stride=100, output_dir=None):
    """Compute characterization stats for all CSI + reference datasets.

    Parameters
    ----------
    data_root : str
        Root folder containing CSI dataset subfolders.
    n_pca : int
        Number of PCA components for projection.
    window_len : int
        Window length for CSI datasets.
    guaranteed_sr : int
        Resampling rate.
    stride : int
        Window stride.
    output_dir : str or None
        Directory to save CSV results.

    Returns
    -------
    list of dict : stats per dataset
    """
    set_global_seed(42)
    results = []

    # --- CSI Datasets ---
    dataset_dirs = [
        'home_har_data',
        'home_occupation_data',
        'office_har_data',
        'office_localization_data',
    ]

    for dname in dataset_dirs:
        dpath = os.path.join(data_root, dname)
        meta_path = os.path.join(dpath, 'dataset_metadata.json')
        if not os.path.isfile(meta_path):
            print(f"  [skip] {dname} -- no metadata")
            continue
        # Use larger stride for large datasets to avoid OOM
        ds_stride = stride
        if dname == 'home_har_data':
            ds_stride = max(stride, window_len)  # non-overlapping windows
        try:
            train_ds, _ = TrainingDataset.from_metadata(
                root_dir=dpath, pipeline_name='amplitude',
                window_len=window_len, guaranteed_sr=guaranteed_sr,
                mode='flattened', stride=ds_stride, balance=True, verbose=False)
        except Exception as e:
            print(f"  [error] {dname}: {e}")
            continue

        X, y = train_ds.X, train_ds.y
        n_cls = train_ds.num_classes
        print(f"\n  {train_ds.name}: {X.shape}, {n_cls} classes")

        # PCA projection
        pca = PCA(n_components=min(n_pca, X.shape[1], X.shape[0]))
        X_pca = pca.fit_transform(X)

        sil = compute_silhouette(X_pca, y)
        fisher = compute_fisher_ratio(X_pca, y)

        row = {'dataset': train_ds.name, 'n_classes': n_cls,
               'n_samples': X.shape[0], 'n_features_raw': X.shape[1],
               'n_pca': X_pca.shape[1],
               'silhouette': sil, 'fisher': fisher}
        results.append(row)
        print(f"    Sil={sil}, Fisher={fisher}")

    # --- Reference Datasets ---
    ref_loaders = [
        ('Iris', load_iris),
        ('MNIST', load_mnist),
        ('CIFAR-10', load_cifar10),
        ('CIFAR-100', load_cifar100),
    ]

    for name, loader_fn in ref_loaders:
        print(f"\n  Loading {name}...")
        X, y, n_cls = loader_fn()
        if X is None:
            print(f"    [skip] Could not load {name}")
            continue
        print(f"    {name}: {X.shape}, {n_cls} classes")

        n_comp = min(n_pca, X.shape[1], X.shape[0])
        if n_comp < 2:
            n_comp = min(X.shape[1], X.shape[0])
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)

        sil = compute_silhouette(X_pca, y)
        fisher = compute_fisher_ratio(X_pca, y)

        row = {'dataset': name, 'n_classes': n_cls,
               'n_samples': X.shape[0], 'n_features_raw': X.shape[1],
               'n_pca': X_pca.shape[1],
               'silhouette': sil, 'fisher': fisher}
        results.append(row)
        print(f"    Sil={sil}, Fisher={fisher}")

    # --- Print results ---
    print(f"\n{'='*70}")
    print("DATASET CHARACTERIZATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'#Cls':>5} {'#Samp':>8} {'Sil':>8} {'Fisher':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['dataset']:<20} {r['n_classes']:>5} {r['n_samples']:>8} "
              f"{r['silhouette']:>8} {r['fisher']:>10}")

    # --- LaTeX table ---
    print(f"\n% LaTeX table for paper:")
    print(r"\begin{tabular}{@{}l r r r@{}}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{\#\,Cls} & \textbf{Sil.} & \textbf{Fisher} \\")
    print(r"\midrule")
    for r in results:
        sil_str = f"${'-' if r['silhouette'] < 0 else ''}${abs(r['silhouette'])}"
        if r['silhouette'] < 0:
            sil_str = f"$-${abs(r['silhouette'])}"
        else:
            sil_str = f"{r['silhouette']}"
        print(f"{r['dataset']:<20} & {r['n_classes']} & {sil_str} & {r['fisher']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # --- Save CSV ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'dataset_characterization.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Saved to {os.path.abspath(csv_path)}")

    return results


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute dataset characterization statistics')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', 'data'),
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--pca-components', type=int, default=50,
                        help='Number of PCA components for projection')
    parser.add_argument('--window', type=int, default=500, help='Window length')
    parser.add_argument('--sr', type=int, default=150, help='Guaranteed sample rate')
    parser.add_argument('--stride', type=int, default=100, help='Window stride')
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'stats_results')

    compute_all_stats(
        data_root=os.path.abspath(args.data_root),
        n_pca=args.pca_components,
        window_len=args.window,
        guaranteed_sr=args.sr,
        stride=args.stride,
        output_dir=output_dir)
