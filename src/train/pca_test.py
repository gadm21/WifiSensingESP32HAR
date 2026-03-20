#!/usr/bin/env python3
"""
PCA-based classification test on 4 CSI datasets.

Loads the 4 datasets via dataset_metadata.json, applies the same preprocessing
pipeline as pca_train.py (CSI_Loader -> FeatureSelector -> rolling variance ->
windowing -> PCA), trains PCA on the train portion, projects both train and
test, then classifies using KNN and SVC.

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
    parser = argparse.ArgumentParser(description='PCA Test: KNN/SVC on 4 datasets')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', 'data'),
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--window', type=int, default=300,
                        help='Window size in samples (default: 300)')
    parser.add_argument('--sr', type=int, default=150,
                        help='Guaranteed sample rate (default: 150)')
    parser.add_argument('--n-components', type=int, default=3,
                        help='Number of PCA components (default: 3)')
    parser.add_argument('--var-window', type=int, default=20,
                        help='Rolling variance window (default: 20, 0=off)')
    parser.add_argument('--cv', action='store_true',
                        help='Use temporal forward-chaining cross-validation')
    parser.add_argument('--n-folds', type=int, default=None,
                        help='Number of CV folds (auto if not set)')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    print(f"[info] Data root: {data_root}")
    print(f"[info] Window: {args.window}, SR: {args.sr}, PCA: {args.n_components} components")
    print(f"[info] Rolling variance: {'window=' + str(args.var_window) if args.var_window > 1 else 'OFF'}")

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
            print(f"\n[warn] Skipping {ds_dir} â€” no {METADATA_FILENAME}")
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
            print(f"  SKIP â€” no test data")
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

        # ---- Classifier 1: KNN on PCA features ----
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

        # ---- Classifier 2: SVC on PCA features ----
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
    print("FINAL PCA TEST COMPARISON: KNN / SVC  x  4 Datasets  (unified metrics)")
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
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pca_results')
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


if __name__ == '__main__':
    main()
