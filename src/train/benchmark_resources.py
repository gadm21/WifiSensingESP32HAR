#!/usr/bin/env python3
"""
Benchmark resource consumption: rolling variance vs sanitization, DL vs ML vs PCA.

Measures wall-clock time and CPU time for each processing method
and learning category on a large sample to highlight practical differences.

Usage:
    python benchmark_resources.py
    python benchmark_resources.py --data-root ../../data --n-samples 50000
"""

import argparse
import os
import sys
import time
import csv
import numpy as np

# Add train/ to path
_train_dir = os.path.dirname(os.path.abspath(__file__))
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from utils import (CSI_Loader, FeatureSelector, CSI_SUBCARRIER_MASK,
                   CsiSanitizer, TrainingDataset, load_all_datasets,
                   set_global_seed)


# =============================================================================
# Helpers
# =============================================================================
def measure_resource(func, *args, **kwargs):
    """Run func and return (result, wall_time_s, cpu_time_s)."""
    t_wall_0 = time.perf_counter()
    t_cpu_0 = time.process_time()

    result = func(*args, **kwargs)

    t_cpu_1 = time.process_time()
    t_wall_1 = time.perf_counter()

    wall_s = round(t_wall_1 - t_wall_0, 4)
    cpu_s = round(t_cpu_1 - t_cpu_0, 4)
    return result, wall_s, cpu_s


# =============================================================================
# Phase 1: Rolling Variance vs Sanitization
# =============================================================================
def benchmark_rolling_variance(mag, var_window):
    """Compute rolling variance on amplitude array."""
    return TrainingDataset._rolling_variance(mag, var_window)


def benchmark_sanitization(real, imag):
    """Run SHARP-based phase sanitization."""
    csi_complex = real + 1j * imag
    sanitizer = CsiSanitizer(verbose=False)
    return sanitizer._sanitize(csi_complex)


def benchmark_preprocessing(data_root, n_samples=10000, output_dir=None):
    """Benchmark rolling variance vs sanitization on real CSI data.

    Parameters
    ----------
    data_root : str
        Root folder containing dataset subfolders.
    n_samples : int
        Number of CSI packets to use for benchmarking.
    output_dir : str or None
        Directory to save CSV results.

    Returns
    -------
    list of dict : benchmark results
    """
    # Load a single dataset to get real data
    loader = CSI_Loader(verbose=False)
    loader.guaranteed_sr = 150
    selector = FeatureSelector(mask=CSI_SUBCARRIER_MASK, verbose=False)

    # Find first available dataset
    dataset_dirs = ['home_har_data', 'home_occupation_data',
                    'office_har_data', 'office_localization_data']
    mag, real, imag = None, None, None
    for dname in dataset_dirs:
        import json
        dpath = os.path.join(data_root, dname)
        meta_path = os.path.join(dpath, 'dataset_metadata.json')
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            metadata = json.load(f)
        for entry in metadata['files']:
            fpath = os.path.join(dpath, entry['path'])
            if not os.path.isfile(fpath):
                continue
            try:
                data = loader.process(fpath)
                data = selector.process(data)
                mag = data['mag']
                real = data['real']
                imag = data['imag']
                break
            except Exception as e:
                print(f"  [warn] {fpath}: {e}")
                continue
        if mag is not None:
            break

    if mag is None:
        print("[ERROR] No CSI data found for benchmarking.")
        return []

    # Truncate/tile to n_samples
    if mag.shape[0] < n_samples:
        reps = int(np.ceil(n_samples / mag.shape[0]))
        mag = np.tile(mag, (reps, 1))[:n_samples]
        real = np.tile(real, (reps, 1))[:n_samples]
        imag = np.tile(imag, (reps, 1))[:n_samples]
    else:
        mag = mag[:n_samples]
        real = real[:n_samples]
        imag = imag[:n_samples]

    print(f"\n{'='*70}")
    print(f"PREPROCESSING BENCHMARK ({n_samples} CSI packets, {mag.shape[1]} subcarriers)")
    print(f"{'='*70}")

    results = []

    # Rolling variance with different windows
    for w in [20, 200, 2000]:
        _, wall, cpu = measure_resource(benchmark_rolling_variance, mag, w)
        row = {'method': f'RollingVariance_W{w}', 'n_samples': n_samples,
               'wall_time_s': wall, 'cpu_time_s': cpu}
        results.append(row)
        print(f"  Rolling Variance (W={w:>4d}):  wall={wall:>8.4f}s  "
              f"cpu={cpu:>8.4f}s")

    # Sanitization (use smaller sample for Lasso - it's very slow)
    san_samples = min(n_samples, 500)  # Lasso is O(N*K*P) per packet
    real_sub = real[:san_samples]
    imag_sub = imag[:san_samples]
    _, wall, cpu = measure_resource(benchmark_sanitization, real_sub, imag_sub)
    # Scale time estimate to full n_samples
    scale = n_samples / san_samples
    row = {'method': 'SHARP_Sanitization', 'n_samples': san_samples,
           'wall_time_s': wall, 'cpu_time_s': cpu,
           'estimated_wall_full': round(wall * scale, 2),
           'estimated_cpu_full': round(cpu * scale, 2)}
    results.append(row)
    print(f"  SHARP Sanitization ({san_samples} pkts): wall={wall:>8.4f}s  "
          f"cpu={cpu:>8.4f}s")
    print(f"    (estimated for {n_samples} pkts: wall~{wall*scale:.1f}s  "
          f"cpu~{cpu*scale:.1f}s)")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'preprocessing_benchmark.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()),
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Saved to {os.path.abspath(csv_path)}")

    return results


# =============================================================================
# Phase 2: DL vs ML vs PCA learning categories
# =============================================================================
def benchmark_learning(data_root, window_len=300, guaranteed_sr=150,
                       var_window=20, output_dir=None):
    """Benchmark training time and resources for PCA, ML, and DL pipelines.

    Parameters
    ----------
    data_root : str
        Root folder containing dataset subfolders.
    window_len : int
        Window length.
    guaranteed_sr : int
        Resampling rate.
    var_window : int
        Rolling variance window.
    output_dir : str or None
        Directory to save CSV results.

    Returns
    -------
    list of dict : benchmark results
    """
    import json

    set_global_seed(42)

    # Load one dataset for benchmarking (Office HAR - moderate size)
    dataset_dirs = ['office_har_data', 'home_occupation_data',
                    'office_localization_data', 'home_har_data']
    train_ds, test_ds = None, None
    ds_name = None
    for dname in dataset_dirs:
        dpath = os.path.join(data_root, dname)
        meta_path = os.path.join(dpath, 'dataset_metadata.json')
        if not os.path.isfile(meta_path):
            continue
        try:
            train_ds, test_ds = TrainingDataset.from_metadata(
                root_dir=dpath, pipeline_name='rolling_variance',
                window_len=window_len, guaranteed_sr=guaranteed_sr,
                mode='flattened', var_window=var_window, balance=True,
                verbose=False)
            ds_name = train_ds.name
            break
        except Exception as e:
            print(f"  [warn] {dname}: {e}")
            continue

    if train_ds is None:
        print("[ERROR] No dataset loaded for learning benchmark.")
        return []

    X_tr, y_tr = train_ds.X, train_ds.y
    X_te, y_te = test_ds.X, test_ds.y
    n_classes = train_ds.num_classes

    print(f"\n{'='*70}")
    print(f"LEARNING BENCHMARK (dataset={ds_name}, "
          f"train={X_tr.shape}, test={X_te.shape})")
    print(f"{'='*70}")

    results = []

    # --- PCA Baseline ---
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

    def run_pca():
        pca = PCA(n_components=3)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_tr_pca, y_tr)
        preds = knn.predict(X_te_pca)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_te, preds)

    acc, wall, cpu = measure_resource(run_pca)
    results.append({'category': 'PCA+KNN', 'dataset': ds_name,
                    'wall_time_s': wall, 'cpu_time_s': cpu,
                    'accuracy': round(acc, 4)})
    print(f"  PCA+KNN:     wall={wall:>8.4f}s  cpu={cpu:>8.4f}s  "
          f"acc={acc:.4f}")

    # --- ML: RandomForest ---
    from sklearn.ensemble import RandomForestClassifier

    def run_rf():
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_te)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_te, preds)

    acc, wall, cpu = measure_resource(run_rf)
    results.append({'category': 'ML_RandomForest', 'dataset': ds_name,
                    'wall_time_s': wall, 'cpu_time_s': cpu,
                    'accuracy': round(acc, 4)})
    print(f"  RF:          wall={wall:>8.4f}s  cpu={cpu:>8.4f}s  "
          f"acc={acc:.4f}")

    # --- ML: XGBoost ---
    from xgboost import XGBClassifier

    def run_xgb():
        xgb = XGBClassifier(n_estimators=100, max_depth=3, random_state=42,
                            n_jobs=-1, verbosity=0)
        xgb.fit(X_tr, y_tr)
        preds = xgb.predict(X_te)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_te, preds)

    acc, wall, cpu = measure_resource(run_xgb)
    results.append({'category': 'ML_XGBoost', 'dataset': ds_name,
                    'wall_time_s': wall, 'cpu_time_s': cpu,
                    'accuracy': round(acc, 4)})
    print(f"  XGBoost:     wall={wall:>8.4f}s  cpu={cpu:>8.4f}s  "
          f"acc={acc:.4f}")

    # --- DL: Conv1D ---
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from dl import make_conv1d_model, train_model

    def run_conv1d():
        model = make_conv1d_model(
            n_subcarriers=52, window_len=window_len, n_classes=n_classes,
            config='small', use_batch_norm=False, use_whitening=False)
        trained_model, info = train_model(
            model, X_tr, y_tr, X_te, X_te, y_te,
            epochs=50, lr=1e-3, batch_size=64,
            use_coral=False, verbose=False)
        device = next(trained_model.parameters()).device
        with torch.no_grad():
            logits = trained_model.predict(torch.FloatTensor(X_te).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_te, preds)

    acc, wall, cpu = measure_resource(run_conv1d)
    results.append({'category': 'DL_Conv1D', 'dataset': ds_name,
                    'wall_time_s': wall, 'cpu_time_s': cpu,
                    'accuracy': round(acc, 4)})
    print(f"  Conv1D:      wall={wall:>8.4f}s  cpu={cpu:>8.4f}s  "
          f"acc={acc:.4f}")

    # --- DL: MLP ---
    from dl import make_mlp_model

    def run_mlp():
        model = make_mlp_model(
            n_features=X_tr.shape[1], n_classes=n_classes,
            config='small', use_batch_norm=False, use_whitening=False)
        trained_model, info = train_model(
            model, X_tr, y_tr, X_te, X_te, y_te,
            epochs=50, lr=1e-3, batch_size=64,
            use_coral=False, verbose=False)
        device = next(trained_model.parameters()).device
        with torch.no_grad():
            logits = trained_model.predict(torch.FloatTensor(X_te).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_te, preds)

    acc, wall, cpu = measure_resource(run_mlp)
    results.append({'category': 'DL_MLP', 'dataset': ds_name,
                    'wall_time_s': wall, 'cpu_time_s': cpu,
                    'accuracy': round(acc, 4)})
    print(f"  MLP:         wall={wall:>8.4f}s  cpu={cpu:>8.4f}s  "
          f"acc={acc:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'learning_benchmark.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()),
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Saved to {os.path.abspath(csv_path)}")

    return results


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark resource consumption: preprocessing and learning')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', 'data'),
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--n-samples', type=int, default=50000,
                        help='Number of CSI packets for preprocessing benchmark')
    parser.add_argument('--window', type=int, default=300, help='Window length')
    parser.add_argument('--sr', type=int, default=150, help='Guaranteed sample rate')
    parser.add_argument('--var-window', type=int, default=20,
                        help='Rolling variance window for learning benchmark')
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'stats_results')

    # Phase 1: Preprocessing benchmark
    preprocess_results = benchmark_preprocessing(
        data_root=os.path.abspath(args.data_root),
        n_samples=args.n_samples,
        output_dir=output_dir)

    # Phase 2: Learning benchmark
    learning_results = benchmark_learning(
        data_root=os.path.abspath(args.data_root),
        window_len=args.window,
        guaranteed_sr=args.sr,
        var_window=args.var_window,
        output_dir=output_dir)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
