#!/usr/bin/env python3
"""
PCA-based classification test on 4 CSI datasets.

Loads the 4 datasets via dataset_metadata.json, applies the same preprocessing
pipeline as pca_train.py (CSI_Loader -> FeatureSelector -> rolling variance ->
windowing -> PCA), trains PCA on the train portion, projects both train and
test, then classifies using DTW (1-NN with Dynamic Time Warping), KNN, and SVC.

Usage:
    python pca_test.py
    python pca_test.py --data-root ../../wifi_sensing_data --window 100 --n-components 3
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from collections import Counter

# Add train/ to path for CSI_Loader, FeatureSelector, etc.
_train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train')
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from utils import (CSI_Loader, FeatureSelector, CSI_SUBCARRIER_MASK, METADATA_FILENAME,
                  TrainingDataset, compute_all_metrics, print_metrics_summary,
                  METRICS_CSV_FIELDS, set_global_seed)


# Use the canonical implementation from utils to avoid duplication
_rolling_variance = TrainingDataset._rolling_variance


# ---------------------------------------------------------------------------
# PCA (same as pca_train.fit_pca)
# ---------------------------------------------------------------------------
def fit_pca(X, n_components=3):
    """Fit PCA via SVD. Returns projected data, mean, components, explained variance."""
    mean = X.mean(axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    projected = X_centered @ components.T
    var_explained = (S ** 2) / (S ** 2).sum() * 100
    return projected, mean, components, var_explained


def project_pca(X, mean, components):
    """Project X using pre-fitted PCA."""
    return (X - mean) @ components.T


# ---------------------------------------------------------------------------
# DTW distance (Sakoe-Chiba band)
# ---------------------------------------------------------------------------
def dtw_distance(a, b, radius=3):
    """DTW distance between two 1D or 2D trajectories with Sakoe-Chiba band."""
    n, m = len(a), len(b)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - radius)
        j_hi = min(m, i + radius)
        for j in range(j_lo, j_hi + 1):
            d = np.sum((a[i - 1] - b[j - 1]) ** 2)
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return np.sqrt(cost[n, m])


# ---------------------------------------------------------------------------
# DTW 1-NN classifier
# ---------------------------------------------------------------------------
class DTW_1NN:
    """1-Nearest Neighbor classifier using DTW distance on PCA trajectories."""

    def __init__(self, traj_len=10, traj_stride=5, dtw_radius=3):
        self.traj_len = traj_len
        self.traj_stride = traj_stride
        self.dtw_radius = dtw_radius
        self._train_chunks = []
        self._train_labels = []

    def fit(self, X_projected, y, file_boundaries=None):
        """Build trajectory chunks from projected PCA points.

        Parameters
        ----------
        X_projected : np.ndarray, shape (N, n_components)
            PCA-projected training windows.
        y : np.ndarray, shape (N,)
            Labels for each window.
        file_boundaries : list of int, optional
            Indices where file boundaries occur (to avoid cross-file chunks).
            If None, treats all data as one continuous sequence per label.
        """
        self._train_chunks = []
        self._train_labels = []

        if file_boundaries is None:
            # Build chunks per class (treat each class as continuous)
            for cls in np.unique(y):
                mask = y == cls
                pts = X_projected[mask]
                for start in range(0, len(pts) - self.traj_len + 1, self.traj_stride):
                    chunk = pts[start:start + self.traj_len].astype(np.float32)
                    self._train_chunks.append(chunk)
                    self._train_labels.append(cls)
        else:
            # Build chunks respecting file boundaries
            boundaries = sorted(set([0] + list(file_boundaries) + [len(y)]))
            for i in range(len(boundaries) - 1):
                lo, hi = boundaries[i], boundaries[i + 1]
                seg_pts = X_projected[lo:hi]
                seg_y = y[lo:hi]
                if len(seg_pts) < self.traj_len:
                    continue
                cls = seg_y[0]  # assume single label per file segment
                for start in range(0, len(seg_pts) - self.traj_len + 1, self.traj_stride):
                    chunk = seg_pts[start:start + self.traj_len].astype(np.float32)
                    self._train_chunks.append(chunk)
                    self._train_labels.append(cls)

        self._train_labels = np.array(self._train_labels)
        print(f"  [DTW_1NN] Built {len(self._train_chunks)} trajectory chunks "
              f"(len={self.traj_len}, stride={self.traj_stride})")

    def predict(self, X_projected, y=None):
        """Predict labels for test PCA points using DTW 1-NN on trajectories.

        Builds test trajectory chunks and finds nearest training chunk.
        """
        if y is not None:
            # Build test chunks per class for fair evaluation
            all_preds = []
            all_true = []
            for cls in np.unique(y):
                mask = y == cls
                pts = X_projected[mask]
                for start in range(0, len(pts) - self.traj_len + 1, self.traj_stride):
                    chunk = pts[start:start + self.traj_len].astype(np.float32)
                    # Find nearest training chunk
                    best_dist = np.inf
                    best_label = -1
                    for tc, tl in zip(self._train_chunks, self._train_labels):
                        d = dtw_distance(chunk, tc, radius=self.dtw_radius)
                        if d < best_dist:
                            best_dist = d
                            best_label = tl
                    all_preds.append(best_label)
                    all_true.append(cls)
            return np.array(all_preds), np.array(all_true)
        else:
            # Build chunks from continuous sequence
            preds = []
            for start in range(0, len(X_projected) - self.traj_len + 1, self.traj_stride):
                chunk = X_projected[start:start + self.traj_len].astype(np.float32)
                best_dist = np.inf
                best_label = -1
                for tc, tl in zip(self._train_chunks, self._train_labels):
                    d = dtw_distance(chunk, tc, radius=self.dtw_radius)
                    if d < best_dist:
                        best_dist = d
                        best_label = tl
                preds.append(best_label)
            return np.array(preds), None


# ---------------------------------------------------------------------------
# Load a single dataset using pca_train-style pipeline via TrainingDataset
# ---------------------------------------------------------------------------
def load_dataset_pca_style(root_dir, window=100, guaranteed_sr=100, var_window=20):
    """Load a dataset using the rolling_variance pipeline via TrainingDataset.

    Returns
    -------
    train_X, train_y, test_X, test_y : np.ndarray
        Flattened windowed magnitude vectors (after rolling variance) and labels.
    metadata : dict
    labels : list of str
    label_map : dict
    """
    train_ds, test_ds = TrainingDataset.from_metadata(
        root_dir=root_dir,
        pipeline_name='rolling_variance',
        window_len=window,
        guaranteed_sr=guaranteed_sr,
        mode='flattened',
        stride=None,
        var_window=var_window,
        verbose=False,
    )

    meta_path = os.path.join(root_dir, METADATA_FILENAME)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    return train_ds.X, train_ds.y, test_ds.X, test_ds.y, metadata, train_ds.labels, train_ds.label_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='PCA Test: DTW/KNN/SVC on 4 datasets')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', '..', 'wifi_sensing_data'),
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--window', type=int, default=300,
                        help='Window size in samples (default: 300)')
    parser.add_argument('--sr', type=int, default=150,
                        help='Guaranteed sample rate (default: 150)')
    parser.add_argument('--n-components', type=int, default=3,
                        help='Number of PCA components (default: 3)')
    parser.add_argument('--var-window', type=int, default=20,
                        help='Rolling variance window (default: 20, 0=off)')
    parser.add_argument('--traj-len', type=int, default=10,
                        help='DTW trajectory chunk length (default: 10)')
    parser.add_argument('--traj-stride', type=int, default=5,
                        help='DTW trajectory stride (default: 5)')
    parser.add_argument('--dtw-radius', type=int, default=3,
                        help='DTW Sakoe-Chiba band radius (default: 3)')
    parser.add_argument('--cv', action='store_true',
                        help='Use temporal forward-chaining cross-validation')
    parser.add_argument('--n-folds', type=int, default=None,
                        help='Number of CV folds (auto if not set)')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    print(f"[info] Data root: {data_root}")
    print(f"[info] Window: {args.window}, SR: {args.sr}, PCA: {args.n_components} components")
    print(f"[info] Rolling variance: {'window=' + str(args.var_window) if args.var_window > 1 else 'OFF'}")
    print(f"[info] DTW: traj_len={args.traj_len}, stride={args.traj_stride}, radius={args.dtw_radius}")

    DATASET_DIRS = [
        'home_har_data',
        'home_occupation_data',
        'office_har_data',
        'office_localization_data',
    ]

    all_results = {}

    # Build list of (ds_dir, fold_idx, train_X, train_y, test_X, test_y, labels, label_map)
    ds_fold_list = []
    for ds_dir in DATASET_DIRS:
        ds_path = os.path.join(data_root, ds_dir)
        meta_path = os.path.join(ds_path, METADATA_FILENAME)
        if not os.path.isfile(meta_path):
            print(f"\n[warn] Skipping {ds_dir} — no {METADATA_FILENAME}")
            continue

        if args.cv:
            try:
                folds = TrainingDataset.from_metadata_cv(
                    root_dir=ds_path,
                    n_folds=args.n_folds,
                    pipeline_name='rolling_variance',
                    window_len=args.window,
                    guaranteed_sr=args.sr,
                    mode='flattened',
                    var_window=args.var_window,
                    verbose=False,
                )
                for fold_idx, train_ds, test_ds in folds:
                    ds_fold_list.append((
                        ds_dir, fold_idx,
                        train_ds.X, train_ds.y,
                        test_ds.X, test_ds.y,
                        train_ds.labels, train_ds.label_map,
                    ))
            except Exception as e:
                print(f"  ERROR loading CV folds for {ds_dir}: {e}")
                continue
        else:
            try:
                train_X, train_y, test_X, test_y, metadata, labels, label_map = \
                    load_dataset_pca_style(
                        ds_path, window=args.window, guaranteed_sr=args.sr,
                        var_window=args.var_window)
                ds_fold_list.append((
                    ds_dir, -1, train_X, train_y, test_X, test_y, labels, label_map,
                ))
            except Exception as e:
                print(f"  ERROR loading {ds_dir}: {e}")
                continue

    for ds_dir, fold_idx, train_X, train_y, test_X, test_y, labels, label_map in ds_fold_list:
        fold_tag = f"fold{fold_idx}" if fold_idx >= 0 else "fixed"

        print(f"\n{'='*80}")
        print(f"DATASET: {ds_dir}  |  Split: {fold_tag}")
        print(f"{'='*80}")

        if test_X.shape[0] == 0:
            print(f"  SKIP — no test data")
            continue

        print(f"\n  Train: {train_X.shape}  Test: {test_X.shape}  "
              f"Classes: {len(labels)} {labels}")
        for i, lbl in enumerate(labels):
            n_tr = (train_y == i).sum()
            n_te = (test_y == i).sum()
            print(f"    {lbl}: train={n_tr}, test={n_te}")

        # ---- Fit PCA on train ----
        print(f"\n  [pca] Fitting PCA ({args.n_components} components) on {train_X.shape}...")
        t0 = time.process_time()
        projected_train, pca_mean, pca_components, var_explained = \
            fit_pca(train_X, n_components=args.n_components)
        pca_time = time.process_time() - t0
        print(f"  [pca] Variance: {', '.join(f'PC{i+1}={v:.1f}%' for i, v in enumerate(var_explained[:args.n_components]))}")
        print(f"  [pca] Fit time: {pca_time:.2f}s")

        # Project test
        projected_test = project_pca(test_X, pca_mean, pca_components)

        n_classes = len(labels)
        ds_results = {}

        # ---- Classifier 1: DTW 1-NN ----
        print(f"\n  --- DTW 1-NN ---")
        dtw_clf = DTW_1NN(
            traj_len=args.traj_len,
            traj_stride=args.traj_stride,
            dtw_radius=args.dtw_radius,
        )
        t0 = time.process_time()
        dtw_clf.fit(projected_train, train_y)
        dtw_fit_time = time.process_time() - t0

        t0 = time.process_time()
        dtw_preds, dtw_true = dtw_clf.predict(projected_test, y=test_y)
        dtw_infer_time = time.process_time() - t0

        if len(dtw_preds) > 0 and len(dtw_true) > 0:
            dtw_m = compute_all_metrics(dtw_true, dtw_preds, n_classes=n_classes)
            dtw_m['fit_time_s'] = round(dtw_fit_time, 2)
            dtw_m['infer_time_s'] = round(dtw_infer_time, 2)
            dtw_m['n_chunks'] = len(dtw_preds)
            print_metrics_summary(dtw_m, title=f'DTW_1NN on {ds_dir}')
            ds_results['DTW_1NN'] = dtw_m
        else:
            print(f"    No DTW predictions (insufficient data)")

        # ---- Classifier 2: KNN on PCA features ----
        print(f"\n  --- KNN (k=5) ---")
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)
        t0 = time.process_time()
        knn.fit(projected_train, train_y)
        knn_fit_time = time.process_time() - t0

        t0 = time.process_time()
        knn_preds = knn.predict(projected_test)
        knn_infer_time = time.process_time() - t0

        knn_prob = knn.predict_proba(projected_test)
        knn_m = compute_all_metrics(test_y, knn_preds, y_prob=knn_prob, n_classes=n_classes)
        knn_m['fit_time_s'] = round(knn_fit_time, 2)
        knn_m['infer_time_s'] = round(knn_infer_time, 3)
        print_metrics_summary(knn_m, title=f'KNN on {ds_dir}')
        ds_results['KNN'] = knn_m

        # ---- Classifier 3: SVC on PCA features ----
        print(f"\n  --- SVC (RBF) ---")
        from sklearn.svm import SVC
        svc = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42, probability=True)
        t0 = time.process_time()
        svc.fit(projected_train, train_y)
        svc_fit_time = time.process_time() - t0

        t0 = time.process_time()
        svc_preds = svc.predict(projected_test)
        svc_infer_time = time.process_time() - t0

        svc_prob = svc.predict_proba(projected_test)
        svc_m = compute_all_metrics(test_y, svc_preds, y_prob=svc_prob, n_classes=n_classes)
        svc_m['fit_time_s'] = round(svc_fit_time, 2)
        svc_m['infer_time_s'] = round(svc_infer_time, 3)
        print_metrics_summary(svc_m, title=f'SVC on {ds_dir}')
        ds_results['SVC'] = svc_m

        result_key = f"{ds_dir}__{fold_tag}"
        all_results[result_key] = ds_results

    # ---- Final comparison table ----
    print(f"\n{'='*160}")
    print("FINAL PCA TEST COMPARISON: DTW / KNN / SVC  x  4 Datasets  (unified metrics)")
    print(f"{'='*160}")
    hdr = (f"{'Dataset':<30} {'Classifier':<12} | "
           f"{'Acc':>6} {'BalAcc':>6} {'F1w':>6} {'F1mac':>6} "
           f"{'Prec':>6} {'Rec':>6} {'Kappa':>6} {'MCC':>6} "
           f"{'ECE':>6} | {'Fit':>7} {'Infer':>7}")
    print(hdr)
    print("-" * 140)
    for result_key, ds_res in all_results.items():
        for clf_name, m in ds_res.items():
            ece_val = m.get('ece', float('nan'))
            print(f"{result_key:<35} {clf_name:<12} | "
                  f"{m['accuracy']:>6.4f} {m['balanced_accuracy']:>6.4f} "
                  f"{m['f1_weighted']:>6.4f} {m['f1_macro']:>6.4f} "
                  f"{m['precision_weighted']:>6.4f} {m['recall_weighted']:>6.4f} "
                  f"{m['cohen_kappa']:>6.4f} {m['mcc']:>6.4f} "
                  f"{ece_val:>6.4f} | "
                  f"{m['fit_time_s']:>6.2f}s {m['infer_time_s']:>6.3f}s")
        print("-" * 150)

    # ---- Best per dataset ----
    print(f"\n{'='*80}")
    print("BEST CLASSIFIER PER DATASET")
    print(f"{'='*80}")
    for result_key, ds_res in all_results.items():
        if not ds_res:
            continue
        best_clf = max(ds_res, key=lambda k: ds_res[k]['accuracy'])
        bm = ds_res[best_clf]
        print(f"  {result_key:<35}: {best_clf:<10}  "
              f"Acc={bm['accuracy']:.4f}  F1={bm['f1_weighted']:.4f}  "
              f"Kappa={bm['cohen_kappa']:.4f}")

    print(f"\n{'='*80}")
    print("PCA test experiments completed!")
    print(f"{'='*80}")

    # ---- Save full results to CSV (unified metric columns) ----
    import csv
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_tag = '_cv' if args.cv else ''
    csv_path = os.path.join(results_dir, f'pca_test_results{csv_tag}.csv')
    fieldnames = ['dataset', 'fold', 'classifier'] + METRICS_CSV_FIELDS + ['fit_time_s', 'infer_time_s']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for result_key, ds_res in all_results.items():
            parts = result_key.split('__')
            ds_name = parts[0]
            fold_tag = parts[1] if len(parts) > 1 else 'fixed'
            for clf_name, m in ds_res.items():
                row = {'dataset': ds_name, 'fold': fold_tag, 'classifier': clf_name}
                row.update(m)
                writer.writerow(row)
    print(f"\n[info] Results saved to {os.path.abspath(csv_path)}")

    return all_results


def visualize_pca_results(results_dir=None, save=True):
    """Load saved PCA test CSV results and produce comprehensive plots + tables.

    Parameters
    ----------
    results_dir : str or None
        Directory containing pca_test_results[_cv].csv. If None, uses ./results.
    save : bool
        If True, save plots as PNG files into results_dir.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from vis_gui import PlotGUI

    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    results_dir = os.path.abspath(results_dir)

    # Try both CV and non-CV filenames
    csv_path = None
    for tag in ['_cv', '']:
        p = os.path.join(results_dir, f'pca_test_results{tag}.csv')
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        print(f"[vis] ERROR: PCA test results not found in {results_dir}")
        print(f"[vis] Expected: pca_test_results[_cv].csv")
        print(f"[vis] Run pca_test.py first (without --vis) to generate results.")
        return

    df = pd.read_csv(csv_path)
    print(f"[vis] Loaded {len(df)} rows from {csv_path}")

    datasets = df['dataset'].unique()
    classifiers = df['classifier'].unique()
    n_ds = len(datasets)
    n_clf = len(classifiers)

    # Short display names
    ds_short = {d: d.replace('_data', '').replace('_', ' ').title() for d in datasets}

    # Color palettes
    clf_colors = {}
    cmap = plt.cm.Set2
    for i, c in enumerate(classifiers):
        clf_colors[c] = cmap(i / max(n_clf - 1, 1))

    os.makedirs(results_dir, exist_ok=True)

    # Aggregate per dataset+classifier (mean across folds if CV)
    agg_cols = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro',
                'precision_weighted', 'recall_weighted', 'cohen_kappa', 'mcc', 'ece']
    time_cols = ['fit_time_s', 'infer_time_s']
    agg_data = {}
    for ds in datasets:
        for clf in classifiers:
            sub = df[(df['dataset'] == ds) & (df['classifier'] == clf)]
            if len(sub) == 0:
                continue
            key = (ds, clf)
            agg_data[key] = {}
            for col in agg_cols + time_cols:
                if col in sub.columns:
                    vals = sub[col].dropna()
                    agg_data[key][f'{col}_mean'] = float(vals.mean()) if len(vals) > 0 else 0
                    agg_data[key][f'{col}_std'] = float(vals.std()) if len(vals) > 1 else 0

    # ================================================================
    # TABLE 1: Console summary
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  PCA TEST RESULTS: {n_ds} Datasets x {n_clf} Classifiers")
    print(f"{'=' * 130}")
    hdr = (f"{'Dataset':<30} {'Classifier':<12} | "
           f"{'Acc':>12} {'BalAcc':>12} {'F1w':>12} {'Kappa':>12} "
           f"{'MCC':>12} {'ECE':>12} | {'Fit':>8} {'Infer':>8}")
    print(hdr)
    print('-' * 130)

    best_per_ds = {}
    for (ds, clf), vals in agg_data.items():
        acc = vals.get('accuracy_mean', 0)
        if ds not in best_per_ds or acc > best_per_ds[ds][1]:
            best_per_ds[ds] = (clf, acc)

        def _fmt(k, v=vals):
            m = v.get(f'{k}_mean', float('nan'))
            s = v.get(f'{k}_std', 0)
            if s > 0:
                return f"{m:.4f}+/-{s:.4f}"
            return f"{m:.4f}"

        fit_t = vals.get('fit_time_s_mean', 0)
        inf_t = vals.get('infer_time_s_mean', 0)
        print(f"{ds_short.get(ds, ds):<30} {clf:<12} | "
              f"{_fmt('accuracy'):>12} {_fmt('balanced_accuracy'):>12} "
              f"{_fmt('f1_weighted'):>12} {_fmt('cohen_kappa'):>12} "
              f"{_fmt('mcc'):>12} {_fmt('ece'):>12} | "
              f"{fit_t:>7.2f}s {inf_t:>7.3f}s")

    print('-' * 130)
    print("  BEST per dataset:")
    for ds, (clf, acc) in best_per_ds.items():
        print(f"    {ds_short.get(ds, ds):<30} -> {clf:<12} Acc={acc:.4f}")
    print()

    # ================================================================
    # PLOT 1: Grouped bar — Accuracy by Classifier per Dataset
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(max(8, n_ds * 2.5), 5))
    fig1.suptitle('PCA Test: Accuracy by Classifier per Dataset',
                  fontsize=13, fontweight='bold')
    bar_w = 0.8 / n_clf
    for j, clf in enumerate(classifiers):
        x = np.arange(n_ds)
        vals, errs = [], []
        for ds in datasets:
            key = (ds, clf)
            if key in agg_data:
                vals.append(agg_data[key].get('accuracy_mean', 0))
                errs.append(agg_data[key].get('accuracy_std', 0))
            else:
                vals.append(0); errs.append(0)
        ax1.bar(x + j * bar_w, vals, bar_w, yerr=errs,
                label=clf, color=clf_colors[clf], edgecolor='white',
                linewidth=0.5, capsize=3, alpha=0.85)
    ax1.set_xticks(np.arange(n_ds) + bar_w * (n_clf - 1) / 2)
    ax1.set_xticklabels([ds_short.get(d, d) for d in datasets], fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    fig1.tight_layout()
    if save:
        fig1.savefig(os.path.join(results_dir, 'pca_plot_accuracy_bars.png'),
                     dpi=150, bbox_inches='tight')
        print(f"[vis] Saved pca_plot_accuracy_bars.png")

    # ================================================================
    # PLOT 2: Heatmap — All metrics for each Dataset x Classifier
    # ================================================================
    heat_metrics = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro',
                    'cohen_kappa', 'mcc']
    heat_labels = ['Acc', 'BalAcc', 'F1w', 'F1mac', 'Kappa', 'MCC']
    n_hm = len(heat_metrics)
    combo_labels = [f"{ds_short.get(ds, ds)}\n{clf}" for ds in datasets for clf in classifiers]
    heat_data = np.zeros((n_hm, n_ds * n_clf))
    for i, metric in enumerate(heat_metrics):
        for j, ds in enumerate(datasets):
            for k, clf in enumerate(classifiers):
                key = (ds, clf)
                idx = j * n_clf + k
                if key in agg_data:
                    heat_data[i, idx] = agg_data[key].get(f'{metric}_mean', 0)

    fig2, ax2 = plt.subplots(figsize=(max(10, n_ds * n_clf * 1.3), n_hm * 0.8 + 2))
    im = ax2.imshow(heat_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(np.arange(len(combo_labels)))
    ax2.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(np.arange(n_hm))
    ax2.set_yticklabels(heat_labels, fontsize=10)
    for i in range(n_hm):
        for j in range(len(combo_labels)):
            v = heat_data[i, j]
            color = 'white' if v < 0.5 else 'black'
            ax2.text(j, i, f'{v:.3f}', ha='center', va='center',
                     fontsize=8, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax2, label='Score', shrink=0.8)
    ax2.set_title('PCA Test: Multi-Metric Heatmap (Dataset x Classifier)',
                  fontsize=13, fontweight='bold')
    fig2.tight_layout()
    if save:
        fig2.savefig(os.path.join(results_dir, 'pca_plot_metric_heatmap.png'),
                     dpi=150, bbox_inches='tight')
        print(f"[vis] Saved pca_plot_metric_heatmap.png")

    # ================================================================
    # PLOT 3: Multi-metric radar per dataset
    # ================================================================
    radar_metrics = ['accuracy', 'f1_weighted', 'cohen_kappa', 'mcc', 'balanced_accuracy']
    radar_labels = ['Accuracy', 'F1 Weighted', 'Cohen k', 'MCC', 'Balanced Acc']
    n_rm = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_rm, endpoint=False).tolist()
    angles += angles[:1]

    fig3, axes3 = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5),
                               subplot_kw=dict(polar=True))
    if n_ds == 1:
        axes3 = [axes3]
    fig3.suptitle('PCA Test: Multi-Metric Radar by Dataset',
                  fontsize=13, fontweight='bold', y=1.05)
    for ax, ds in zip(axes3, datasets):
        for clf in classifiers:
            key = (ds, clf)
            if key not in agg_data:
                continue
            values = []
            for rm in radar_metrics:
                v = agg_data[key].get(f'{rm}_mean', 0)
                values.append(max(0, v))
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=1.5, label=clf,
                    color=clf_colors[clf], markersize=4)
            ax.fill(angles, values, alpha=0.1, color=clf_colors[clf])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(ds_short.get(ds, ds), fontsize=10, fontweight='bold', pad=15)
        ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1.3, -0.1))
    fig3.tight_layout()
    if save:
        fig3.savefig(os.path.join(results_dir, 'pca_plot_radar_metrics.png'),
                     dpi=150, bbox_inches='tight')
        print(f"[vis] Saved pca_plot_radar_metrics.png")

    # ================================================================
    # PLOT 4: Timing comparison — Fit time + Inference time grouped bars
    # ================================================================
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle('PCA Test: Timing Comparison', fontsize=13, fontweight='bold', y=1.02)

    for ax, time_col, title in [(ax4a, 'fit_time_s', 'Fit Time (s)'),
                                 (ax4b, 'infer_time_s', 'Inference Time (s)')]:
        bar_w4 = 0.8 / n_clf
        for j, clf in enumerate(classifiers):
            x = np.arange(n_ds)
            vals = []
            for ds in datasets:
                key = (ds, clf)
                if key in agg_data:
                    vals.append(agg_data[key].get(f'{time_col}_mean', 0))
                else:
                    vals.append(0)
            ax.bar(x + j * bar_w4, vals, bar_w4, label=clf,
                   color=clf_colors[clf], edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.set_xticks(np.arange(n_ds) + bar_w4 * (n_clf - 1) / 2)
        ax.set_xticklabels([ds_short.get(d, d) for d in datasets], fontsize=9)
        ax.set_ylabel(title, fontsize=10)
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_title(title, fontsize=11, fontweight='bold')
    fig4.tight_layout()
    if save:
        fig4.savefig(os.path.join(results_dir, 'pca_plot_timing.png'),
                     dpi=150, bbox_inches='tight')
        print(f"[vis] Saved pca_plot_timing.png")

    # ================================================================
    # PLOT 5: F1 Weighted comparison — grouped by dataset
    # ================================================================
    fig5, ax5 = plt.subplots(figsize=(max(8, n_ds * 2.5), 5))
    fig5.suptitle('PCA Test: F1 Weighted Score by Dataset x Classifier',
                  fontsize=13, fontweight='bold')
    bar_w5 = 0.8 / n_clf
    for j, clf in enumerate(classifiers):
        x = np.arange(n_ds)
        vals, errs = [], []
        for ds in datasets:
            key = (ds, clf)
            if key in agg_data:
                vals.append(agg_data[key].get('f1_weighted_mean', 0))
                errs.append(agg_data[key].get('f1_weighted_std', 0))
            else:
                vals.append(0); errs.append(0)
        ax5.bar(x + j * bar_w5, vals, bar_w5, yerr=errs,
                label=clf, color=clf_colors[clf], edgecolor='white',
                linewidth=0.5, capsize=3, alpha=0.85)
    ax5.set_xticks(np.arange(n_ds) + bar_w5 * (n_clf - 1) / 2)
    ax5.set_xticklabels([ds_short.get(d, d) for d in datasets], fontsize=10)
    ax5.set_ylabel('F1 Weighted', fontsize=11)
    ax5.set_ylim(0, 1.05)
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)
    fig5.tight_layout()
    if save:
        fig5.savefig(os.path.join(results_dir, 'pca_plot_f1_weighted.png'),
                     dpi=150, bbox_inches='tight')
        print(f"[vis] Saved pca_plot_f1_weighted.png")

    # ================================================================
    # PLOT 6: ECE vs Accuracy scatter
    # ================================================================
    if 'ece' in df.columns:
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        markers_clf = {'DTW_1NN': 'o', 'KNN': 's', 'SVC': '^'}
        for _, row in df.iterrows():
            acc = row.get('accuracy', np.nan)
            ece = row.get('ece', np.nan)
            if np.isnan(acc) or np.isnan(ece):
                continue
            clf = row['classifier']
            ds = row['dataset']
            marker = markers_clf.get(clf, 'o')
            ax6.scatter(acc, ece, c=[clf_colors.get(clf, 'gray')], marker=marker,
                        s=80, alpha=0.8, edgecolors='black', linewidths=0.5)

        legend_clf = [Line2D([0], [0], marker=markers_clf.get(c, 'o'), color='w',
                             markerfacecolor=clf_colors[c], markersize=8, label=c)
                      for c in classifiers]
        ax6.legend(handles=legend_clf, title='Classifier', fontsize=9)
        ax6.set_xlabel('Accuracy', fontsize=11)
        ax6.set_ylabel('ECE (lower = better)', fontsize=11)
        ax6.set_title('PCA Test: Calibration Quality', fontsize=13, fontweight='bold')
        ax6.grid(alpha=0.3, linestyle='--')
        ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False)
        fig6.tight_layout()
        if save:
            fig6.savefig(os.path.join(results_dir, 'pca_plot_ece_vs_accuracy.png'),
                         dpi=150, bbox_inches='tight')
            print(f"[vis] Saved pca_plot_ece_vs_accuracy.png")

    # ================================================================
    # PLOT 7: MCC vs Kappa scatter
    # ================================================================
    if 'mcc' in df.columns and 'cohen_kappa' in df.columns:
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        for _, row in df.iterrows():
            mcc = row.get('mcc', np.nan)
            kappa = row.get('cohen_kappa', np.nan)
            if np.isnan(mcc) or np.isnan(kappa):
                continue
            clf = row['classifier']
            marker = markers_clf.get(clf, 'o')
            ax7.scatter(kappa, mcc, c=[clf_colors.get(clf, 'gray')], marker=marker,
                        s=70, alpha=0.7, edgecolors='black', linewidths=0.4)
        ax7.plot([-0.5, 1], [-0.5, 1], 'k--', alpha=0.3, linewidth=0.8)
        ax7.set_xlabel("Cohen's Kappa", fontsize=11)
        ax7.set_ylabel('MCC', fontsize=11)
        ax7.set_title('PCA Test: Agreement Metrics', fontsize=13, fontweight='bold')
        ax7.grid(alpha=0.3, linestyle='--')
        ax7.spines['top'].set_visible(False); ax7.spines['right'].set_visible(False)
        fig7.tight_layout()
        if save:
            fig7.savefig(os.path.join(results_dir, 'pca_plot_mcc_vs_kappa.png'),
                         dpi=150, bbox_inches='tight')
            print(f"[vis] Saved pca_plot_mcc_vs_kappa.png")

    # ================================================================
    # PLOT 8: Best ranking table as figure
    # ================================================================
    fig8, ax8 = plt.subplots(figsize=(12, max(3, n_ds * 0.8 + 2)))
    ax8.axis('off')
    rank_cols = ['Dataset', 'Best Classifier', 'Accuracy', 'F1w',
                 'Kappa', 'MCC', 'ECE', 'Fit Time']
    rank_data = []
    for ds in datasets:
        best_clf = None
        best_acc = -1
        for clf in classifiers:
            key = (ds, clf)
            if key in agg_data:
                acc = agg_data[key].get('accuracy_mean', 0)
                if acc > best_acc:
                    best_acc = acc
                    best_clf = clf
        if best_clf is None:
            continue
        bk = (ds, best_clf)
        rank_data.append([
            ds_short.get(ds, ds), best_clf,
            f"{agg_data[bk].get('accuracy_mean', 0):.4f}",
            f"{agg_data[bk].get('f1_weighted_mean', 0):.4f}",
            f"{agg_data[bk].get('cohen_kappa_mean', 0):.4f}",
            f"{agg_data[bk].get('mcc_mean', 0):.4f}",
            f"{agg_data[bk].get('ece_mean', 0):.4f}",
            f"{agg_data[bk].get('fit_time_s_mean', 0):.2f}s",
        ])
    tbl = ax8.table(cellText=rank_data, colLabels=rank_cols,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.5)
    for j in range(len(rank_cols)):
        tbl[0, j].set_facecolor('#2d333b')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(len(rank_data)):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(rank_cols)):
            tbl[i + 1, j].set_facecolor(color)
    ax8.set_title('Best PCA Classifier per Dataset', fontsize=14,
                  fontweight='bold', pad=20)
    fig8.tight_layout()
    if save:
        fig8.savefig(os.path.join(results_dir, 'pca_plot_best_ranking.png'),
                     dpi=150, bbox_inches='tight')
        print(f"[vis] Saved pca_plot_best_ranking.png")

    # ================================================================
    # PLOT 9: Per-fold accuracy spread (violin plot, if CV data)
    # ================================================================
    if 'fold' in df.columns and df['fold'].nunique() > 1:
        fig9, axes9 = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=True)
        if n_ds == 1:
            axes9 = [axes9]
        fig9.suptitle('PCA Test: Per-Fold Accuracy Spread',
                      fontsize=13, fontweight='bold', y=1.02)
        for ax, ds in zip(axes9, datasets):
            sub = df[df['dataset'] == ds]
            box_data, box_labels = [], []
            for clf in classifiers:
                vals = sub[sub['classifier'] == clf]['accuracy'].dropna()
                if len(vals) > 0:
                    box_data.append(vals.values)
                    box_labels.append(clf)
            if box_data:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                                showmeans=True, meanprops=dict(marker='D',
                                markerfacecolor='red', markersize=4))
                for pi, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(clf_colors.get(box_labels[pi], '#cccccc'))
                    patch.set_alpha(0.6)
            ax.set_title(ds_short.get(ds, ds), fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axes9[0].set_ylabel('Accuracy', fontsize=10)
        fig9.tight_layout()
        if save:
            fig9.savefig(os.path.join(results_dir, 'pca_plot_fold_spread.png'),
                         dpi=150, bbox_inches='tight')
            print(f"[vis] Saved pca_plot_fold_spread.png")

    # ================================================================
    # PLOT 10: Precision vs Recall scatter
    # ================================================================
    if 'precision_weighted' in df.columns and 'recall_weighted' in df.columns:
        fig10, ax10 = plt.subplots(figsize=(8, 6))
        for _, row in df.iterrows():
            prec = row.get('precision_weighted', np.nan)
            rec = row.get('recall_weighted', np.nan)
            if np.isnan(prec) or np.isnan(rec):
                continue
            clf = row['classifier']
            marker = markers_clf.get(clf, 'o')
            ax10.scatter(rec, prec, c=[clf_colors.get(clf, 'gray')], marker=marker,
                         s=70, alpha=0.8, edgecolors='black', linewidths=0.4)
        ax10.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
        ax10.set_xlabel('Recall (Weighted)', fontsize=11)
        ax10.set_ylabel('Precision (Weighted)', fontsize=11)
        ax10.set_title('PCA Test: Precision vs Recall', fontsize=13, fontweight='bold')
        ax10.grid(alpha=0.3, linestyle='--')
        ax10.spines['top'].set_visible(False); ax10.spines['right'].set_visible(False)
        fig10.tight_layout()
        if save:
            fig10.savefig(os.path.join(results_dir, 'pca_plot_precision_vs_recall.png'),
                          dpi=150, bbox_inches='tight')
            print(f"[vis] Saved pca_plot_precision_vs_recall.png")

    # ================================================================
    # PLOT 11: Balanced Accuracy bars
    # ================================================================
    if any('balanced_accuracy_mean' in agg_data.get(k, {}) for k in agg_data):
        fig11, ax11b = plt.subplots(figsize=(max(8, n_ds * 2.5), 5))
        fig11.suptitle('PCA Test: Balanced Accuracy by Classifier',
                       fontsize=13, fontweight='bold')
        bar_w11 = 0.8 / n_clf
        for j, clf in enumerate(classifiers):
            x = np.arange(n_ds)
            vals = []
            for ds in datasets:
                key = (ds, clf)
                vals.append(agg_data[key].get('balanced_accuracy_mean', 0) if key in agg_data else 0)
            ax11b.bar(x + j * bar_w11, vals, bar_w11, label=clf,
                      color=clf_colors[clf], edgecolor='white',
                      linewidth=0.5, alpha=0.85)
        ax11b.set_xticks(np.arange(n_ds) + bar_w11 * (n_clf - 1) / 2)
        ax11b.set_xticklabels([ds_short.get(d, d) for d in datasets], fontsize=10)
        ax11b.set_ylabel('Balanced Accuracy', fontsize=11)
        ax11b.set_ylim(0, 1.05)
        ax11b.legend(fontsize=10)
        ax11b.grid(axis='y', alpha=0.3, linestyle='--')
        ax11b.spines['top'].set_visible(False); ax11b.spines['right'].set_visible(False)
        fig11.tight_layout()
        if save:
            fig11.savefig(os.path.join(results_dir, 'pca_plot_balanced_accuracy.png'),
                          dpi=150, bbox_inches='tight')
            print(f"[vis] Saved pca_plot_balanced_accuracy.png")

    # ================================================================
    # PLOT 12: Full results table (all combos)
    # ================================================================
    fig12, ax12t = plt.subplots(figsize=(14, max(3, n_ds * n_clf * 0.35 + 2)))
    ax12t.axis('off')
    tbl_cols = ['Dataset', 'Classifier', 'Acc', 'BalAcc', 'F1w',
                'Kappa', 'MCC', 'ECE', 'Fit(s)', 'Infer(s)']
    tbl_data = []
    for ds in datasets:
        for clf in classifiers:
            key = (ds, clf)
            if key not in agg_data:
                continue
            v = agg_data[key]
            tbl_data.append([
                ds_short.get(ds, ds), clf,
                f"{v.get('accuracy_mean', 0):.4f}",
                f"{v.get('balanced_accuracy_mean', 0):.4f}",
                f"{v.get('f1_weighted_mean', 0):.4f}",
                f"{v.get('cohen_kappa_mean', 0):.4f}",
                f"{v.get('mcc_mean', 0):.4f}",
                f"{v.get('ece_mean', 0):.4f}",
                f"{v.get('fit_time_s_mean', 0):.2f}",
                f"{v.get('infer_time_s_mean', 0):.3f}",
            ])
    tbl12 = ax12t.table(cellText=tbl_data, colLabels=tbl_cols,
                        loc='center', cellLoc='center')
    tbl12.auto_set_font_size(False)
    tbl12.set_fontsize(8)
    tbl12.scale(1.0, 1.3)
    for j in range(len(tbl_cols)):
        tbl12[0, j].set_facecolor('#2d333b')
        tbl12[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(len(tbl_data)):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(tbl_cols)):
            tbl12[i + 1, j].set_facecolor(color)
    ax12t.set_title('Full PCA Results Table (all configurations)',
                    fontsize=14, fontweight='bold', pad=20)
    fig12.tight_layout()
    if save:
        fig12.savefig(os.path.join(results_dir, 'pca_plot_full_results_table.png'),
                      dpi=150, bbox_inches='tight')
        print(f"[vis] Saved pca_plot_full_results_table.png")

    # ================================================================
    # PLOT 13: Metric correlation matrix
    # ================================================================
    corr_cols = [c for c in ['accuracy', 'f1_weighted', 'f1_macro',
                              'cohen_kappa', 'mcc', 'ece',
                              'precision_weighted', 'recall_weighted']
                 if c in df.columns]
    if len(corr_cols) >= 4:
        fig13c, ax13c = plt.subplots(figsize=(8, 6))
        corr_matrix = df[corr_cols].corr()
        im13 = ax13c.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1,
                            aspect='auto')
        n_corr = len(corr_cols)
        short_labels = [c.replace('_', '\n').replace('weighted', 'w')
                        .replace('precision', 'prec')
                        .replace('recall', 'rec') for c in corr_cols]
        ax13c.set_xticks(np.arange(n_corr))
        ax13c.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax13c.set_yticks(np.arange(n_corr))
        ax13c.set_yticklabels(short_labels, fontsize=8)
        for ci in range(n_corr):
            for cj in range(n_corr):
                v = corr_matrix.values[ci, cj]
                clr = 'white' if abs(v) > 0.5 else 'black'
                ax13c.text(cj, ci, f'{v:.2f}', ha='center', va='center',
                          fontsize=7, fontweight='bold', color=clr)
        plt.colorbar(im13, ax=ax13c, label='Correlation', shrink=0.8)
        ax13c.set_title('PCA Test: Metric Correlation Matrix',
                        fontsize=13, fontweight='bold')
        fig13c.tight_layout()
        if save:
            fig13c.savefig(os.path.join(results_dir, 'pca_plot_metric_correlation.png'),
                           dpi=150, bbox_inches='tight')
            print(f"[vis] Saved pca_plot_metric_correlation.png")

    # ================================================================
    # Launch tabbed GUI with all figures
    # ================================================================
    tab_names = [
        ('Accuracy Bars', 'pca_plot_accuracy_bars.png'),
        ('Metric Heatmap', 'pca_plot_metric_heatmap.png'),
        ('Radar Metrics', 'pca_plot_radar_metrics.png'),
        ('Timing', 'pca_plot_timing.png'),
        ('F1 Weighted', 'pca_plot_f1_weighted.png'),
        ('ECE vs Accuracy', 'pca_plot_ece_vs_accuracy.png'),
        ('MCC vs Kappa', 'pca_plot_mcc_vs_kappa.png'),
        ('Best Ranking', 'pca_plot_best_ranking.png'),
        ('Fold Spread', 'pca_plot_fold_spread.png'),
        ('Precision vs Recall', 'pca_plot_precision_vs_recall.png'),
        ('Balanced Accuracy', 'pca_plot_balanced_accuracy.png'),
        ('Full Results Table', 'pca_plot_full_results_table.png'),
        ('Metric Correlation', 'pca_plot_metric_correlation.png'),
    ]
    gui = PlotGUI("PCA Test Results", results_dir=results_dir)
    open_figs = {fig.number: fig for fig in [plt.figure(n) for n in plt.get_fignums()]}
    fig_list = list(open_figs.values())
    for i, fig in enumerate(fig_list):
        if i < len(tab_names):
            name, fname = tab_names[i]
        else:
            name, fname = f"Plot {i+1}", f"pca_plot_{i+1}.png"
        gui.add_plot(name, fig, fname)
    print(f"\n[vis] Launching PCA GUI with {len(fig_list)} plots...")
    gui.show()


if __name__ == '__main__':
    import argparse as _ap
    _parser = _ap.ArgumentParser(description='PCA Test: DTW/KNN/SVC on 4 datasets')
    _parser.add_argument('--vis', action='store_true',
                         help='Skip testing — load saved CSV results and show plots/tables')
    _parser.add_argument('--results-dir', type=str, default=None,
                         help='Directory containing result CSVs (default: ./results)')
    # Quick pre-parse to check for --vis before full argparse in main()
    _known, _remaining = _parser.parse_known_args()
    if _known.vis:
        visualize_pca_results(results_dir=_known.results_dir, save=True)
    else:
        main()
