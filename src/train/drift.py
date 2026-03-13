"""
Publication-Grade Temporal Domain Drift & Dataset Characterization Suite
for CSI-based Activity Recognition.

22 experiments across three time-separated domains (D1=earliest, D2=middle, D3=latest):

Pairwise Drift Analysis (run for each pair D1→D2, D2→D3, D1→D3):
 1. Domain Classifier Test        — Is there distribution shift at all?
 2. Per-Class Centroid Shift       — Is shift global or class-dependent?
 3. Covariance Structure Shift     — Is geometry distorted?
 4. Label Prior Shift              — Do class frequencies differ?
 5. Classifier Logit Shift         — Is there systematic logit bias?
 6. Linear Separability Test       — Are features good but boundary shifted?
 7. Maximum Mean Discrepancy       — Distribution distance beyond mean/cov
 8. Class-Conditional MMD          — Per-class distribution distance
 9. Subspace Angle Analysis        — PCA subspace rotation/scaling
10. Feature Whitening Test         — Is drift mostly second-order?

Temporal 3-Domain Analysis:
11. Drift Velocity & Acceleration  — MMD-based drift speed and acceleration
12. Feature Variance Evolution     — Per-feature variance heatmap over time
13. Feature Distribution Shift     — KS-test heatmap per feature
14. Class Boundary Deformation     — Fisher margin collapse detection
15. Temporal Feature Collapse      — PCA explained variance evolution
16. Cross-Domain Confusion Matrix  — Activity confusion heatmaps
17. Representation Stability Score — Cosine feature consistency

Dataset Characterization:
18. Intrinsic Dimensionality       — PCA, participation ratio, MLE
19. Feature Correlation Structure  — Correlation heatmap & redundancy
20. Class Separability Score       — Fisher, Silhouette, Davies-Bouldin
21. Cluster Structure Analysis     — k-means, t-SNE visualization
22. Noise & Stability Analysis     — Within/between-class variance, SNR

Outputs: results/ directory with CSV tables, PNG plots, and summary.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine as cosine_dist
from scipy.linalg import sqrtm
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from utils import (TrainingDataset, CSI_Loader, FeatureSelector,
                   CSI_SUBCARRIER_MASK, METADATA_FILENAME, set_global_seed)
from dl import make_adaptive_model

# Publication-quality plot defaults
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


def _ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _save_csv(filename, rows, header):
    """Save a list of dicts/lists to CSV."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"  [CSV] Saved {path}")


def _save_plot(fig, filename):
    """Save figure to plots directory."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [PLOT] Saved {path}")


# =============================================================================
# Helper: Train source model and extract features
# =============================================================================
def train_source_model(X_source, y_source, n_features, n_classes,
                       epochs=50, batch_size=64, lr=1e-3, verbose=True):
    """Train an AdaptiveModel on source data only. Returns trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_adaptive_model(n_features, n_classes, config='small')
    model = model.to(device)

    ds = TensorDataset(torch.FloatTensor(X_source), torch.LongTensor(y_source))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(logits, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
        if verbose and ((epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0):
            print(f"    Epoch {epoch+1:3d}/{epochs} | Acc: {correct/total:.4f}")

    model.eval()
    return model


def extract_features(model, X, batch_size=256):
    """Extract features from a trained model."""
    device = next(model.parameters()).device
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i+batch_size]).to(device)
            f = model.extract_features(xb)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def extract_logits(model, X, batch_size=256):
    """Extract raw logits from a trained model."""
    device = next(model.parameters()).device
    model.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i+batch_size]).to(device)
            out = model(xb)
            logits_list.append(out.cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def _rbf_kernel(X, Y, sigma):
    """Compute RBF kernel matrix between X and Y."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dists = XX + YY.T - 2.0 * X @ Y.T
    return np.exp(-dists / (2.0 * sigma ** 2))


def _pairwise_sq_dists(X, Y):
    """Compute pairwise squared distances without 3D broadcast (memory-safe)."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    return XX + YY.T - 2.0 * X @ Y.T


def _compute_mmd(X, Y, sigma=None):
    """Compute MMD^2 with RBF kernel. Uses median heuristic for sigma if None."""
    if sigma is None:
        # Median heuristic on a subsample for efficiency
        max_med = min(200, len(X), len(Y))
        sub_X = X[np.random.choice(len(X), max_med, replace=False)] if len(X) > max_med else X
        sub_Y = Y[np.random.choice(len(Y), max_med, replace=False)] if len(Y) > max_med else Y
        sub = np.concatenate([sub_X, sub_Y], axis=0)
        dists = _pairwise_sq_dists(sub, sub)
        sigma = np.sqrt(np.median(dists[dists > 0]) / 2.0)
        if sigma < 1e-8:
            sigma = 1.0

    K_xx = _rbf_kernel(X, X, sigma)
    K_yy = _rbf_kernel(Y, Y, sigma)
    K_xy = _rbf_kernel(X, Y, sigma)

    n = len(X)
    m = len(Y)

    # Unbiased estimator
    np.fill_diagonal(K_xx, 0)
    np.fill_diagonal(K_yy, 0)

    mmd2 = K_xx.sum() / (n * (n - 1)) + K_yy.sum() / (m * (m - 1)) - 2.0 * K_xy.mean()
    return float(mmd2), float(sigma)


# =============================================================================
# Experiment 1: Domain Classifier Test
# =============================================================================
def exp1_domain_classifier(feat_source, feat_target):
    """Domain classifier with 5-fold CV, AUC, and linear vs MLP comparison.

    Returns
    -------
    dict
        Accuracy/AUC for linear and MLP classifiers with confidence intervals.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Domain Classifier Test (5-fold CV)")
    print("=" * 70)
    print("  Goal: Can a classifier tell source from target features?")
    print("  If MLP >> Linear → nonlinear domain shift")

    X = np.concatenate([feat_source, feat_target], axis=0)
    y = np.concatenate([np.zeros(len(feat_source)), np.ones(len(feat_target))])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    linear_accs, linear_aucs = [], []
    mlp_accs, mlp_aucs = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # Linear
        clf_lin = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf_lin.fit(X_tr, y_tr)
        linear_accs.append(accuracy_score(y_te, clf_lin.predict(X_te)))
        linear_aucs.append(roc_auc_score(y_te, clf_lin.predict_proba(X_te)[:, 1]))

        # MLP
        clf_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                random_state=42, early_stopping=True)
        clf_mlp.fit(X_tr, y_tr)
        mlp_accs.append(accuracy_score(y_te, clf_mlp.predict(X_te)))
        mlp_aucs.append(roc_auc_score(y_te, clf_mlp.predict_proba(X_te)[:, 1]))

    lin_acc_m, lin_acc_s = np.mean(linear_accs), np.std(linear_accs)
    lin_auc_m, lin_auc_s = np.mean(linear_aucs), np.std(linear_aucs)
    mlp_acc_m, mlp_acc_s = np.mean(mlp_accs), np.std(mlp_accs)
    mlp_auc_m, mlp_auc_s = np.mean(mlp_aucs), np.std(mlp_aucs)

    print(f"\n  {'Classifier':<12} | {'Accuracy':>18} | {'ROC-AUC':>18}")
    print("  " + "-" * 54)
    print(f"  {'Linear':<12} | {lin_acc_m:.4f} +/- {lin_acc_s:.4f} | {lin_auc_m:.4f} +/- {lin_auc_s:.4f}")
    print(f"  {'MLP':<12} | {mlp_acc_m:.4f} +/- {mlp_acc_s:.4f} | {mlp_auc_m:.4f} +/- {mlp_auc_s:.4f}")

    gap = mlp_acc_m - lin_acc_m
    print(f"\n  MLP - Linear accuracy gap: {gap:+.4f}")

    if lin_acc_m < 0.55:
        print("  Interpretation: ~50% → No real distribution shift")
    elif lin_acc_m < 0.75:
        print("  Interpretation: Moderate covariate shift")
    else:
        print("  Interpretation: Strong domain shift")

    if gap > 0.05:
        print("  MLP >> Linear → Nonlinear domain shift detected")
    else:
        print("  MLP ~ Linear → Shift is mostly linear")

    return {
        'linear_acc': round(lin_acc_m, 4), 'linear_acc_std': round(lin_acc_s, 4),
        'linear_auc': round(lin_auc_m, 4), 'linear_auc_std': round(lin_auc_s, 4),
        'mlp_acc': round(mlp_acc_m, 4), 'mlp_acc_std': round(mlp_acc_s, 4),
        'mlp_auc': round(mlp_auc_m, 4), 'mlp_auc_std': round(mlp_auc_s, 4),
        'mlp_linear_gap': round(gap, 4),
    }


# =============================================================================
# Experiment 2: Per-Class Centroid Shift
# =============================================================================
def exp2_centroid_shift(feat_source, y_source, feat_target, y_target, label_map):
    """Compare per-class centroids: L2, Mahalanobis, and cosine similarity.

    Mahalanobis accounts for covariance geometry.
    Cosine similarity distinguishes scaling vs directional shift.

    Returns
    -------
    dict
        Per-class distances and global statistics.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Per-Class Centroid Shift (L2 + Mahalanobis + Cosine)")
    print("=" * 70)
    print("  Goal: Is shift global or class-dependent?")
    print("  Cosine high + L2 large → scaling shift")
    print("  Cosine low → directional shift")

    inv_map = {v: k for k, v in label_map.items()}
    classes = sorted(np.unique(np.concatenate([y_source, y_target])))

    # Compute pooled source covariance for Mahalanobis
    s_centered = feat_source - feat_source.mean(axis=0, keepdims=True)
    cov_s = (s_centered.T @ s_centered) / max(len(feat_source) - 1, 1)
    # Regularize for inversion
    reg = 1e-4 * np.eye(cov_s.shape[0])
    cov_s_inv = np.linalg.inv(cov_s + reg)

    rows = []
    for c in classes:
        s_mask = y_source == c
        t_mask = y_target == c
        if s_mask.sum() == 0 or t_mask.sum() == 0:
            continue

        mu_s = feat_source[s_mask].mean(axis=0)
        mu_t = feat_target[t_mask].mean(axis=0)
        diff = mu_s - mu_t

        l2 = float(np.linalg.norm(diff))
        mahal = float(np.sqrt(diff @ cov_s_inv @ diff))
        cos_sim = float(1.0 - cosine_dist(mu_s, mu_t))
        name = inv_map.get(c, str(c))
        rows.append({'name': name, 'l2': l2, 'mahal': mahal, 'cos': cos_sim})

    print(f"\n  {'Class':<12} | {'L2':>10} {'Mahalanobis':>12} {'Cosine Sim':>11}")
    print("  " + "-" * 50)
    for r in rows:
        print(f"  {r['name']:<12} | {r['l2']:>10.4f} {r['mahal']:>12.4f} {r['cos']:>11.4f}")

    l2s = [r['l2'] for r in rows]
    mahals = [r['mahal'] for r in rows]
    coss = [r['cos'] for r in rows]

    mean_l2, std_l2 = np.mean(l2s), np.std(l2s)
    cv_l2 = std_l2 / mean_l2 if mean_l2 > 0 else 0
    mean_mahal = np.mean(mahals)
    cv_mahal = np.std(mahals) / mean_mahal if mean_mahal > 0 else 0
    mean_cos = np.mean(coss)

    print(f"\n  L2:   mean={mean_l2:.4f}  std={std_l2:.4f}  CV={cv_l2:.4f}")
    print(f"  Mahal: mean={mean_mahal:.4f}  CV={cv_mahal:.4f}")
    print(f"  Cosine: mean={mean_cos:.4f}")

    if cv_l2 < 0.3:
        print("  Interpretation: All classes shift similarly → Global shift → AdaBN should help")
    elif cv_l2 < 0.7:
        print("  Interpretation: Moderate variation → Mixed shift")
    else:
        print("  Interpretation: High variation → Class-conditional shift → AdaBN may fail")

    if mean_cos > 0.9:
        print("  Cosine high → Mostly scaling/magnitude shift")
    elif mean_cos > 0.7:
        print("  Cosine moderate → Mixed scaling + directional shift")
    else:
        print("  Cosine low → Strong directional shift")

    return {
        'per_class': {r['name']: {'l2': round(r['l2'], 4), 'mahalanobis': round(r['mahal'], 4),
                                   'cosine': round(r['cos'], 4)} for r in rows},
        'mean_l2': round(mean_l2, 4), 'cv_l2': round(cv_l2, 4),
        'mean_mahalanobis': round(mean_mahal, 4), 'cv_mahalanobis': round(cv_mahal, 4),
        'mean_cosine': round(mean_cos, 4),
    }


# =============================================================================
# Experiment 3: Covariance Structure Shift
# =============================================================================
def exp3_covariance_shift(feat_source, feat_target):
    """Compare covariance matrices: relative Frobenius, principal angles, Grassmann distance.

    Returns
    -------
    dict
        Relative Frobenius, principal angles, Grassmann distance, eigenvalue comparison.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Covariance Structure Shift")
    print("=" * 70)
    print("  Goal: Is the feature geometry distorted?")

    d = feat_source.shape[1]

    # Center
    s_centered = feat_source - feat_source.mean(axis=0, keepdims=True)
    t_centered = feat_target - feat_target.mean(axis=0, keepdims=True)

    # Covariance matrices
    cov_s = (s_centered.T @ s_centered) / max(len(feat_source) - 1, 1)
    cov_t = (t_centered.T @ t_centered) / max(len(feat_target) - 1, 1)

    # Frobenius norms
    frob_diff = np.linalg.norm(cov_s - cov_t, 'fro')
    frob_s = np.linalg.norm(cov_s, 'fro')
    # Relative Frobenius: ||C_s - C_t||_F / ||C_s||_F
    frob_rel = frob_diff / max(frob_s, 1e-8)

    # Eigendecomposition (full)
    k = min(20, d)
    eigvals_s, eigvecs_s = np.linalg.eigh(cov_s)
    eigvals_t, eigvecs_t = np.linalg.eigh(cov_t)
    # Sort descending
    idx_s = np.argsort(eigvals_s)[::-1]
    idx_t = np.argsort(eigvals_t)[::-1]
    eigvals_s, eigvecs_s = eigvals_s[idx_s], eigvecs_s[:, idx_s]
    eigvals_t, eigvecs_t = eigvals_t[idx_t], eigvecs_t[:, idx_t]

    eig_s_top = eigvals_s[:k]
    eig_t_top = eigvals_t[:k]
    eig_ratio = eig_t_top / np.clip(eig_s_top, 1e-8, None)

    # Principal angles between top-k eigenspaces
    U_s = eigvecs_s[:, :k]  # (d, k)
    U_t = eigvecs_t[:, :k]  # (d, k)
    # SVD of U_s^T @ U_t gives cos(principal angles)
    cos_angles = np.linalg.svd(U_s.T @ U_t, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    principal_angles_deg = np.degrees(np.arccos(cos_angles))

    # Grassmann distance: sqrt(sum of squared principal angles in radians)
    principal_angles_rad = np.radians(principal_angles_deg)
    grassmann_dist = float(np.sqrt(np.sum(principal_angles_rad ** 2)))

    rank_corr = float(np.corrcoef(eig_s_top, eig_t_top)[0, 1])

    print(f"\n  Feature dimension:              {d}")
    print(f"  ||C_s - C_t||_F:                {frob_diff:.4f}")
    print(f"  ||C_s||_F:                      {frob_s:.4f}")
    print(f"  Relative Frobenius:             {frob_rel:.4f}")

    print(f"\n  Top-{k} eigenvalues:")
    print(f"  {'Rank':<6} | {'Source':>12} {'Target':>12} {'Ratio(T/S)':>12}")
    print("  " + "-" * 48)
    for i in range(k):
        print(f"  {i+1:<6} | {eig_s_top[i]:>12.4f} {eig_t_top[i]:>12.4f} {eig_ratio[i]:>12.4f}")

    print(f"\n  Eigenvalue rank correlation:    {rank_corr:.4f}")

    print(f"\n  Principal angles (top-{k} subspace):")
    print(f"  {'Angle #':<8} | {'Degrees':>10}")
    print("  " + "-" * 22)
    for i in range(min(k, 10)):
        print(f"  {i+1:<8} | {principal_angles_deg[i]:>10.2f}")
    if k > 10:
        print(f"  ... ({k - 10} more angles omitted)")

    mean_angle = float(np.mean(principal_angles_deg))
    max_angle = float(np.max(principal_angles_deg))
    print(f"\n  Mean principal angle:           {mean_angle:.2f} deg")
    print(f"  Max principal angle:            {max_angle:.2f} deg")
    print(f"  Grassmann distance:             {grassmann_dist:.4f}")

    if frob_rel < 0.1:
        print("  Interpretation: Small relative covariance difference")
    elif rank_corr > 0.9 and mean_angle < 15:
        print("  Interpretation: Covariance shifted but subspace preserved → CORAL should help")
    elif mean_angle > 30:
        print("  Interpretation: Significant subspace rotation → Geometric distortion")
    else:
        print("  Interpretation: Moderate geometric shift")

    return {
        'frobenius_diff': round(float(frob_diff), 4),
        'frobenius_relative': round(float(frob_rel), 4),
        'eigenvalue_rank_correlation': rank_corr,
        'mean_principal_angle_deg': round(mean_angle, 2),
        'max_principal_angle_deg': round(max_angle, 2),
        'grassmann_distance': round(grassmann_dist, 4),
        'top_eigenvalues_source': eig_s_top.tolist(),
        'top_eigenvalues_target': eig_t_top.tolist(),
    }


# =============================================================================
# Experiment 4: Label Prior Shift
# =============================================================================
def exp4_label_prior_shift(y_source, y_target, label_map):
    """Compare class frequency distributions P_s(y) vs P_t(y).

    Returns
    -------
    dict
        Per-class frequencies and KL divergence.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Label Prior Shift")
    print("=" * 70)
    print("  Goal: Do class frequencies differ between domains?")

    inv_map = {v: k for k, v in label_map.items()}
    classes = sorted(np.unique(np.concatenate([y_source, y_target])))

    p_s = np.array([(y_source == c).sum() for c in classes], dtype=float)
    p_t = np.array([(y_target == c).sum() for c in classes], dtype=float)

    p_s_norm = p_s / p_s.sum()
    p_t_norm = p_t / p_t.sum()

    # KL divergence: KL(P_t || P_s)
    kl = np.sum(p_t_norm * np.log(p_t_norm / np.clip(p_s_norm, 1e-8, None)))

    print(f"\n  {'Class':<12} | {'Source Count':>12} {'Source %':>9} | {'Target Count':>12} {'Target %':>9} | {'Ratio T/S':>10}")
    print("  " + "-" * 75)
    for i, c in enumerate(classes):
        name = inv_map.get(c, str(c))
        ratio = p_t_norm[i] / max(p_s_norm[i], 1e-8)
        print(f"  {name:<12} | {int(p_s[i]):>12} {p_s_norm[i]:>8.2%} | {int(p_t[i]):>12} {p_t_norm[i]:>8.2%} | {ratio:>10.4f}")

    print(f"\n  KL(P_target || P_source): {kl:.6f}")

    if kl < 0.01:
        print("  Interpretation: Negligible label prior shift")
    elif kl < 0.1:
        print("  Interpretation: Mild label prior shift")
    else:
        print("  Interpretation: Significant label prior shift → Entropy minimization may collapse to dominant class")

    return {
        'source_freq': p_s_norm.tolist(),
        'target_freq': p_t_norm.tolist(),
        'kl_divergence': float(kl),
    }


# =============================================================================
# Experiment 5: Classifier Logit Shift
# =============================================================================
def exp5_logit_shift(model, X_source, y_source, X_target, y_target, label_map):
    """Logit shift + calibration diagnostics: ECE, confidence histogram, entropy.

    Returns
    -------
    dict
        Per-class logit shifts, ECE, mean entropy, confidence stats.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Classifier Logit Shift + Calibration")
    print("=" * 70)
    print("  Goal: Systematic logit bias? Overconfident but wrong? Underconfident?")

    logits_s = extract_logits(model, X_source)
    logits_t = extract_logits(model, X_target)

    inv_map = {v: k for k, v in label_map.items()}
    n_classes = logits_s.shape[1]

    # Softmax
    def _softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    probs_s = _softmax(logits_s)
    probs_t = _softmax(logits_t)

    # ---- Logit shift table ----
    mean_s = logits_s.mean(axis=0)
    mean_t = logits_t.mean(axis=0)

    print(f"\n  Global mean logits per class:")
    print(f"  {'Class':<12} | {'Source':>10} {'Target':>10} {'Diff':>10}")
    print("  " + "-" * 46)
    for c in range(n_classes):
        name = inv_map.get(c, str(c))
        diff = mean_t[c] - mean_s[c]
        print(f"  {name:<12} | {mean_s[c]:>10.4f} {mean_t[c]:>10.4f} {diff:>+10.4f}")

    # Per-class logit for correct class
    print(f"\n  Per-class mean logits (samples of that class):")
    print(f"  {'Class':<12} | {'Src logit':>10} {'Tgt logit':>10} {'Shift':>10}")
    print("  " + "-" * 46)

    classes = sorted(np.unique(np.concatenate([y_source, y_target])))
    shifts = {}
    for c in classes:
        s_mask = y_source == c
        t_mask = y_target == c
        if s_mask.sum() == 0 or t_mask.sum() == 0:
            continue
        src_logit = logits_s[s_mask, c].mean()
        tgt_logit = logits_t[t_mask, c].mean()
        shift = tgt_logit - src_logit
        name = inv_map.get(c, str(c))
        shifts[name] = float(shift)
        print(f"  {name:<12} | {src_logit:>10.4f} {tgt_logit:>10.4f} {shift:>+10.4f}")

    # ---- Prediction distribution ----
    pred_t = np.argmax(logits_t, axis=1)
    pred_counts = np.bincount(pred_t, minlength=n_classes)
    dominant = inv_map.get(np.argmax(pred_counts), str(np.argmax(pred_counts)))
    dominant_pct = pred_counts.max() / pred_counts.sum()

    print(f"\n  Target predictions distribution:")
    for c in range(n_classes):
        name = inv_map.get(c, str(c))
        pct = pred_counts[c] / pred_counts.sum()
        bar = "#" * int(pct * 40)
        print(f"  {name:<12} | {pred_counts[c]:>6} ({pct:>6.1%}) {bar}")

    if dominant_pct > 0.5:
        print(f"\n  WARNING: {dominant} dominates ({dominant_pct:.1%}) → Possible 1-class collapse")

    # ---- Expected Calibration Error (ECE) ----
    n_bins = 10
    def _compute_ece(probs, labels, n_bins=10):
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_data = []
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                bin_data.append((lo, hi, 0, 0, 0))
                continue
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_count = mask.sum()
            ece += (bin_count / len(probs)) * abs(bin_acc - bin_conf)
            bin_data.append((lo, hi, bin_acc, bin_conf, bin_count))
        return ece, bin_data

    ece_s, bins_s = _compute_ece(probs_s, y_source)
    ece_t, bins_t = _compute_ece(probs_t, y_target)

    print(f"\n  Expected Calibration Error (ECE):")
    print(f"    Source ECE: {ece_s:.4f}")
    print(f"    Target ECE: {ece_t:.4f}")

    # ---- Confidence histogram ----
    conf_s = np.max(probs_s, axis=1)
    conf_t = np.max(probs_t, axis=1)

    print(f"\n  Confidence histogram (target):")
    print(f"  {'Bin':>12} | {'Count':>6} {'Accuracy':>10} {'Avg Conf':>10}")
    print("  " + "-" * 44)
    for lo, hi, acc, conf, cnt in bins_t:
        if cnt > 0:
            print(f"  ({lo:.1f}, {hi:.1f}]   | {int(cnt):>6} {acc:>10.4f} {conf:>10.4f}")
        else:
            print(f"  ({lo:.1f}, {hi:.1f}]   | {int(cnt):>6}        -          -")

    # ---- Mean softmax entropy ----
    def _entropy(p):
        return -np.sum(p * np.log(np.clip(p, 1e-8, 1.0)), axis=1)

    ent_s = _entropy(probs_s)
    ent_t = _entropy(probs_t)
    max_ent = np.log(n_classes)

    print(f"\n  Mean softmax entropy (max={max_ent:.4f}):")
    print(f"    Source: {ent_s.mean():.4f} (std={ent_s.std():.4f})")
    print(f"    Target: {ent_t.mean():.4f} (std={ent_t.std():.4f})")

    if ece_t > 0.2 and conf_t.mean() > 0.7:
        print("  Interpretation: Overconfident but wrong → Representation collapse likely")
    elif ece_t > 0.2 and conf_t.mean() < 0.5:
        print("  Interpretation: Underconfident → Boundary misalignment")
    elif ece_t < 0.1:
        print("  Interpretation: Well-calibrated on target")
    else:
        print("  Interpretation: Moderate miscalibration")

    return {
        'per_class_shifts': shifts,
        'ece_source': round(float(ece_s), 4),
        'ece_target': round(float(ece_t), 4),
        'mean_conf_source': round(float(conf_s.mean()), 4),
        'mean_conf_target': round(float(conf_t.mean()), 4),
        'mean_entropy_source': round(float(ent_s.mean()), 4),
        'mean_entropy_target': round(float(ent_t.mean()), 4),
    }


# =============================================================================
# Experiment 6: Linear Separability Test
# =============================================================================
def exp6_linear_separability(feat_source, y_source, feat_target, y_target, label_map,
                             n_repeats=10):
    """Test if labeled target data fixes performance. 10 repeats with mean±std.

    Returns
    -------
    dict
        Accuracy mean±std for each scenario and percentage.
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 6: Linear Separability Test ({n_repeats} repeats)")
    print("=" * 70)
    print("  Goal: Are features good but boundary shifted?")
    print("  If curve saturates quickly → feature space is good")

    # Source only → target (deterministic, no variance)
    clf_src = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf_src.fit(feat_source, y_source)
    acc_src_only = accuracy_score(y_target, clf_src.predict(feat_target))

    percentages = [0.05, 0.10, 0.20, 0.30, 0.50]
    results = {'source_only': {'mean': round(float(acc_src_only), 4), 'std': 0.0}}

    for pct in percentages:
        combined_accs = []
        tgt_only_accs = []

        for rep in range(n_repeats):
            rng = np.random.RandomState(rep)
            idx_labeled = []
            idx_test = []
            for c in np.unique(y_target):
                c_idx = np.where(y_target == c)[0]
                c_idx = rng.permutation(c_idx)
                n_c = max(1, int(len(c_idx) * pct))
                idx_labeled.extend(c_idx[:n_c].tolist())
                idx_test.extend(c_idx[n_c:].tolist())

            idx_labeled = np.array(idx_labeled)
            idx_test = np.array(idx_test)
            if len(idx_test) == 0:
                continue

            # Source + small target
            X_comb = np.concatenate([feat_source, feat_target[idx_labeled]])
            y_comb = np.concatenate([y_source, y_target[idx_labeled]])
            clf_c = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf_c.fit(X_comb, y_comb)
            combined_accs.append(accuracy_score(y_target[idx_test], clf_c.predict(feat_target[idx_test])))

            # Target only
            clf_t = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf_t.fit(feat_target[idx_labeled], y_target[idx_labeled])
            tgt_only_accs.append(accuracy_score(y_target[idx_test], clf_t.predict(feat_target[idx_test])))

        pct_key = int(pct * 100)
        results[f'source+{pct_key}%target'] = {
            'mean': round(float(np.mean(combined_accs)), 4),
            'std': round(float(np.std(combined_accs)), 4),
        }
        results[f'{pct_key}%target_only'] = {
            'mean': round(float(np.mean(tgt_only_accs)), 4),
            'std': round(float(np.std(tgt_only_accs)), 4),
        }

    print(f"\n  {'Scenario':<30} | {'Accuracy':>20}")
    print("  " + "-" * 54)
    for scenario, v in results.items():
        if isinstance(v, dict):
            print(f"  {scenario:<30} | {v['mean']:.4f} +/- {v['std']:.4f}")
        else:
            print(f"  {scenario:<30} | {v:>10.4f}")

    # Accuracy vs % target labels curve
    print(f"\n  Accuracy vs % target labels (source+target):")
    print(f"  {'%':>5} | {'Accuracy':>20} | {'Bar'}")
    print("  " + "-" * 50)
    print(f"  {'0':>5} | {acc_src_only:.4f}               | {'#' * int(acc_src_only * 40)}")
    for pct in percentages:
        pct_key = int(pct * 100)
        v = results[f'source+{pct_key}%target']
        bar = "#" * int(v['mean'] * 40)
        print(f"  {pct_key:>5} | {v['mean']:.4f} +/- {v['std']:.4f} | {bar}")

    gain_10 = results.get('source+10%target', {}).get('mean', acc_src_only) - acc_src_only
    if gain_10 > 0.05:
        print("\n  Interpretation: Small labeled target data helps significantly")
        print("  → Features are good, decision boundary shifted")
    else:
        print("\n  Interpretation: Adding target labels doesn't help much")
        print("  → Feature space itself may be distorted")

    # Check saturation
    gain_50 = results.get('source+50%target', {}).get('mean', acc_src_only) - acc_src_only
    if gain_50 > 0 and gain_10 / max(gain_50, 1e-8) > 0.5:
        print("  Curve saturates quickly → Feature space is usable")

    return results


# =============================================================================
# Experiment 7: Maximum Mean Discrepancy (MMD)
# =============================================================================
def exp7_mmd(feat_source, feat_target, X_source_raw, X_target_raw):
    """Compute MMD before and after feature extraction.

    If MMD increases after feature extraction → model amplifies drift.

    Returns
    -------
    dict
        MMD values before and after feature extraction.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Maximum Mean Discrepancy (MMD)")
    print("=" * 70)
    print("  Goal: Distribution distance beyond mean/covariance")
    print("  Compare MMD before vs after feature extractor")

    # Subsample for kernel computation — use smaller n for high-dim raw features
    max_n_raw = min(200, len(X_source_raw), len(X_target_raw))
    max_n_feat = min(500, len(feat_source), len(feat_target))

    if len(X_source_raw) > max_n_raw:
        idx_s = np.random.choice(len(X_source_raw), max_n_raw, replace=False)
        raw_s = X_source_raw[idx_s]
    else:
        raw_s = X_source_raw
    if len(X_target_raw) > max_n_raw:
        idx_t = np.random.choice(len(X_target_raw), max_n_raw, replace=False)
        raw_t = X_target_raw[idx_t]
    else:
        raw_t = X_target_raw

    print(f"  Computing MMD on raw input features (n={len(raw_s)}, d={raw_s.shape[1]})...")
    mmd_raw, sigma_raw = _compute_mmd(raw_s, raw_t)

    # Subsample extracted features
    if len(feat_source) > max_n_feat:
        idx_s = np.random.choice(len(feat_source), max_n_feat, replace=False)
        fs = feat_source[idx_s]
    else:
        fs = feat_source
    if len(feat_target) > max_n_feat:
        idx_t = np.random.choice(len(feat_target), max_n_feat, replace=False)
        ft = feat_target[idx_t]
    else:
        ft = feat_target

    print("  Computing MMD on extracted features...")
    mmd_feat, sigma_feat = _compute_mmd(fs, ft)

    print(f"\n  {'Stage':<25} | {'MMD^2':>12} {'Sigma':>10}")
    print("  " + "-" * 52)
    print(f"  {'Before feature extractor':<25} | {mmd_raw:>12.6f} {sigma_raw:>10.4f}")
    print(f"  {'After feature extractor':<25} | {mmd_feat:>12.6f} {sigma_feat:>10.4f}")

    if mmd_feat > mmd_raw * 1.5:
        print("\n  Interpretation: MMD INCREASED after feature extraction")
        print("  → Model AMPLIFIES domain drift — features are domain-specific")
    elif mmd_feat < mmd_raw * 0.5:
        print("\n  Interpretation: MMD decreased after feature extraction")
        print("  → Model partially aligns domains in feature space")
    else:
        print("\n  Interpretation: MMD similar before/after")
        print("  → Feature extractor neither amplifies nor reduces drift")

    return {
        'mmd_raw': round(float(mmd_raw), 6),
        'mmd_features': round(float(mmd_feat), 6),
        'sigma_raw': round(float(sigma_raw), 4),
        'sigma_features': round(float(sigma_feat), 4),
        'amplification_ratio': round(float(mmd_feat / max(mmd_raw, 1e-8)), 4),
    }


# =============================================================================
# Experiment 8: Class-Conditional MMD
# =============================================================================
def exp8_class_conditional_mmd(feat_source, y_source, feat_target, y_target, label_map):
    """Compute MMD per class for true class-conditional shift magnitude.

    Returns
    -------
    dict
        Per-class MMD values.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Class-Conditional MMD")
    print("=" * 70)
    print("  Goal: Per-class distribution distance (beyond centroids)")

    inv_map = {v: k for k, v in label_map.items()}
    classes = sorted(np.unique(np.concatenate([y_source, y_target])))

    per_class = {}
    for c in classes:
        s_mask = y_source == c
        t_mask = y_target == c
        if s_mask.sum() < 5 or t_mask.sum() < 5:
            continue

        fs = feat_source[s_mask]
        ft = feat_target[t_mask]

        # Subsample if needed
        max_n = 300
        if len(fs) > max_n:
            fs = fs[np.random.choice(len(fs), max_n, replace=False)]
        if len(ft) > max_n:
            ft = ft[np.random.choice(len(ft), max_n, replace=False)]

        mmd2, sigma = _compute_mmd(fs, ft)
        name = inv_map.get(c, str(c))
        per_class[name] = round(float(mmd2), 6)

    print(f"\n  {'Class':<12} | {'MMD^2':>12}")
    print("  " + "-" * 28)
    mmds = list(per_class.values())
    for name, m in per_class.items():
        print(f"  {name:<12} | {m:>12.6f}")

    mean_mmd = np.mean(mmds)
    std_mmd = np.std(mmds)
    cv_mmd = std_mmd / mean_mmd if mean_mmd > 0 else 0

    print(f"\n  Mean MMD^2:  {mean_mmd:.6f}")
    print(f"  Std MMD^2:   {std_mmd:.6f}")
    print(f"  CV:          {cv_mmd:.4f}")

    if cv_mmd < 0.3:
        print("  Interpretation: Uniform shift across classes → Global adaptation should work")
    elif cv_mmd < 0.7:
        print("  Interpretation: Moderate class-dependent shift")
    else:
        print("  Interpretation: Highly class-dependent shift → Need class-conditional adaptation")

    return {'per_class_mmd': per_class, 'mean': round(mean_mmd, 6),
            'std': round(std_mmd, 6), 'cv': round(cv_mmd, 4)}


# =============================================================================
# Experiment 9: Subspace Angle Analysis (PCA)
# =============================================================================
def exp9_subspace_angles(feat_source, feat_target):
    """PCA subspace analysis: principal angles distinguish rotation vs scaling drift.

    Returns
    -------
    dict
        Principal angles, explained variance ratios, rotation vs scaling diagnosis.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: Subspace Angle Analysis (PCA)")
    print("=" * 70)
    print("  Goal: Rotation drift vs scaling drift?")
    print("  Large angles → rotation drift")
    print("  Small angles + eigenvalue change → scaling drift")

    d = feat_source.shape[1]
    k = min(20, d)

    # PCA on each domain
    s_centered = feat_source - feat_source.mean(axis=0, keepdims=True)
    t_centered = feat_target - feat_target.mean(axis=0, keepdims=True)

    U_s, S_s, Vt_s = np.linalg.svd(s_centered, full_matrices=False)
    U_t, S_t, Vt_t = np.linalg.svd(t_centered, full_matrices=False)

    # Explained variance ratios
    var_s = (S_s ** 2) / (S_s ** 2).sum()
    var_t = (S_t ** 2) / (S_t ** 2).sum()

    # Right singular vectors (principal directions)
    V_s = Vt_s[:k, :].T  # (d, k)
    V_t = Vt_t[:k, :].T  # (d, k)

    # Principal angles via SVD of V_s^T @ V_t
    cos_angles = np.linalg.svd(V_s.T @ V_t, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_angles))

    # Eigenvalue ratio (squared singular values ~ eigenvalues of covariance)
    eig_s = (S_s[:k] ** 2) / max(len(feat_source) - 1, 1)
    eig_t = (S_t[:k] ** 2) / max(len(feat_target) - 1, 1)
    eig_ratio = eig_t / np.clip(eig_s, 1e-8, None)

    print(f"\n  Top-{k} PCA subspace analysis:")
    print(f"  {'PC':<5} | {'Angle(deg)':>10} {'VarRatio_S':>11} {'VarRatio_T':>11} {'EigRatio':>10}")
    print("  " + "-" * 52)
    for i in range(min(k, 15)):
        print(f"  {i+1:<5} | {angles_deg[i]:>10.2f} {var_s[i]:>11.4f} {var_t[i]:>11.4f} {eig_ratio[i]:>10.4f}")
    if k > 15:
        print(f"  ... ({k - 15} more components omitted)")

    mean_angle = float(np.mean(angles_deg))
    max_angle = float(np.max(angles_deg))
    mean_eig_ratio = float(np.mean(eig_ratio))
    std_eig_ratio = float(np.std(eig_ratio))

    # Cumulative explained variance
    cum_var_s = np.cumsum(var_s[:k])
    cum_var_t = np.cumsum(var_t[:k])

    print(f"\n  Mean principal angle:        {mean_angle:.2f} deg")
    print(f"  Max principal angle:         {max_angle:.2f} deg")
    print(f"  Mean eigenvalue ratio (T/S): {mean_eig_ratio:.4f} +/- {std_eig_ratio:.4f}")
    print(f"  Cumulative variance (k={k}):  Source={cum_var_s[-1]:.4f}  Target={cum_var_t[-1]:.4f}")

    # Diagnosis
    rotation = mean_angle > 20
    scaling = abs(mean_eig_ratio - 1.0) > 0.3 or std_eig_ratio > 0.5

    if rotation and scaling:
        print("  Interpretation: Both rotation AND scaling drift")
    elif rotation:
        print("  Interpretation: Rotation drift (subspace rotated, eigenvalues stable)")
    elif scaling:
        print("  Interpretation: Scaling drift (subspace stable, eigenvalues changed)")
    else:
        print("  Interpretation: Minimal subspace drift")

    return {
        'mean_angle_deg': round(mean_angle, 2),
        'max_angle_deg': round(max_angle, 2),
        'mean_eig_ratio': round(mean_eig_ratio, 4),
        'std_eig_ratio': round(std_eig_ratio, 4),
        'rotation_drift': rotation,
        'scaling_drift': scaling,
    }


# =============================================================================
# Experiment 10: Feature Whitening Test
# =============================================================================
def exp10_whitening_test(feat_source, y_source, feat_target, y_target, label_map):
    """Whiten both domains using source statistics, test if drift is second-order.

    Apply: x' = C_s^{-1/2} (x - mu_s)
    Then train classifier on whitened source, test on whitened target.

    If performance jumps → drift is mostly second-order (mean + covariance).
    If not → nonlinear drift.

    Returns
    -------
    dict
        Accuracy before and after whitening.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: Feature Whitening Test")
    print("=" * 70)
    print("  Goal: Is drift mostly second-order (fixable by CORAL/AdaBN)?")

    # Baseline: source classifier on raw features
    clf_raw = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf_raw.fit(feat_source, y_source)
    acc_raw = accuracy_score(y_target, clf_raw.predict(feat_target))

    # Compute source whitening transform
    mu_s = feat_source.mean(axis=0)
    s_centered = feat_source - mu_s
    cov_s = (s_centered.T @ s_centered) / max(len(feat_source) - 1, 1)

    # Regularized inverse square root
    reg = 1e-4 * np.eye(cov_s.shape[0])
    cov_s_reg = cov_s + reg

    # Use eigendecomposition for stable C^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(cov_s_reg)
    eigvals = np.clip(eigvals, 1e-6, None)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    W = eigvecs @ D_inv_sqrt @ eigvecs.T  # Whitening matrix

    # Whiten source using source stats
    feat_source_w = (feat_source - mu_s) @ W.T

    # Whiten target using SOURCE stats (same transform)
    feat_target_w_src = (feat_target - mu_s) @ W.T

    # Whiten target using TARGET stats
    mu_t = feat_target.mean(axis=0)
    t_centered = feat_target - mu_t
    cov_t = (t_centered.T @ t_centered) / max(len(feat_target) - 1, 1)
    cov_t_reg = cov_t + reg
    eigvals_t, eigvecs_t = np.linalg.eigh(cov_t_reg)
    eigvals_t = np.clip(eigvals_t, 1e-6, None)
    D_inv_sqrt_t = np.diag(1.0 / np.sqrt(eigvals_t))
    W_t = eigvecs_t @ D_inv_sqrt_t @ eigvecs_t.T
    feat_target_w_tgt = (feat_target - mu_t) @ W_t.T

    # Test 1: Source whitened → target whitened with source stats
    clf_w1 = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf_w1.fit(feat_source_w, y_source)
    acc_w_src = accuracy_score(y_target, clf_w1.predict(feat_target_w_src))

    # Test 2: Source whitened → target whitened with own stats
    clf_w2 = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf_w2.fit(feat_source_w, y_source)
    acc_w_tgt = accuracy_score(y_target, clf_w2.predict(feat_target_w_tgt))

    print(f"\n  {'Scenario':<45} | {'Accuracy':>10}")
    print("  " + "-" * 60)
    print(f"  {'Raw features (no whitening)':<45} | {acc_raw:>10.4f}")
    print(f"  {'Both whitened with source stats':<45} | {acc_w_src:>10.4f}")
    print(f"  {'Each whitened with own stats (oracle)':<45} | {acc_w_tgt:>10.4f}")

    gain_src = acc_w_src - acc_raw
    gain_tgt = acc_w_tgt - acc_raw

    print(f"\n  Gain (source whitening):  {gain_src:+.4f}")
    print(f"  Gain (oracle whitening):  {gain_tgt:+.4f}")

    if gain_tgt > 0.1:
        print("  Interpretation: Oracle whitening helps significantly")
        print("  → Drift IS mostly second-order → CORAL/AdaBN should help")
        if gain_src > 0.05:
            print("  Source whitening also helps → Shared whitening transform works")
        else:
            print("  But source whitening doesn't help → Need target-specific stats (AdaBN)")
    elif gain_tgt > 0.03:
        print("  Interpretation: Moderate second-order component")
        print("  → Partial benefit from CORAL/AdaBN, but nonlinear component exists")
    else:
        print("  Interpretation: Whitening doesn't help")
        print("  → Drift is NOT second-order → Nonlinear adaptation needed")

    return {
        'acc_raw': round(float(acc_raw), 4),
        'acc_whitened_source_stats': round(float(acc_w_src), 4),
        'acc_whitened_own_stats': round(float(acc_w_tgt), 4),
        'gain_source_whitening': round(float(gain_src), 4),
        'gain_oracle_whitening': round(float(gain_tgt), 4),
    }


# =============================================================================
# Helper: Full MMD Suite (global + class-conditional)
# =============================================================================
def compute_full_mmd_suite(feat_source, y_source, feat_target, y_target):
    """Compute global MMD and class-conditional MMD in one call."""
    max_n = min(500, len(feat_source), len(feat_target))
    fs = feat_source[np.random.choice(len(feat_source), max_n, replace=False)] if len(feat_source) > max_n else feat_source
    ft = feat_target[np.random.choice(len(feat_target), max_n, replace=False)] if len(feat_target) > max_n else feat_target
    mmd_global, sigma = _compute_mmd(fs, ft)
    classes = sorted(np.unique(np.concatenate([y_source, y_target])))
    per_class = {}
    for c in classes:
        s_mask, t_mask = y_source == c, y_target == c
        if s_mask.sum() < 5 or t_mask.sum() < 5:
            continue
        fs_c = feat_source[s_mask]
        ft_c = feat_target[t_mask]
        if len(fs_c) > 300:
            fs_c = fs_c[np.random.choice(len(fs_c), 300, replace=False)]
        if len(ft_c) > 300:
            ft_c = ft_c[np.random.choice(len(ft_c), 300, replace=False)]
        mmd2, _ = _compute_mmd(fs_c, ft_c)
        per_class[int(c)] = round(float(mmd2), 6)
    return {'mmd_global': round(float(mmd_global), 6), 'sigma': round(float(sigma), 4),
            'per_class_mmd': per_class}


# =============================================================================
# Experiment 11: Drift Velocity & Acceleration (MMD-based)
# =============================================================================
def exp11_drift_velocity(F1, F2, F3):
    """MMD-based drift velocity and acceleration across 3 temporal domains.

    velocity = MMD(D1,D2), acceleration = MMD(D1,D3) - MMD(D1,D2)
    Also computes centroid-based velocity vectors and direction similarity.

    Returns
    -------
    dict
        MMD velocities, centroid velocities, acceleration, direction cosine.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 11: Drift Velocity & Acceleration")
    print("=" * 70)

    # MMD-based velocity
    max_n = min(300, len(F1), len(F2), len(F3))
    f1s = F1[np.random.choice(len(F1), max_n, replace=False)] if len(F1) > max_n else F1
    f2s = F2[np.random.choice(len(F2), max_n, replace=False)] if len(F2) > max_n else F2
    f3s = F3[np.random.choice(len(F3), max_n, replace=False)] if len(F3) > max_n else F3

    mmd12, _ = _compute_mmd(f1s, f2s)
    mmd23, _ = _compute_mmd(f2s, f3s)
    mmd13, _ = _compute_mmd(f1s, f3s)

    mmd_velocity = float(mmd12)
    mmd_accel = float(mmd13) - float(mmd12)

    # Centroid-based velocity
    mu1, mu2, mu3 = F1.mean(0), F2.mean(0), F3.mean(0)
    v12 = mu2 - mu1
    v23 = mu3 - mu2
    speed12 = float(np.linalg.norm(v12))
    speed23 = float(np.linalg.norm(v23))
    accel_mag = float(np.linalg.norm(v23 - v12))
    denom = np.linalg.norm(v12) * np.linalg.norm(v23)
    dir_cos = float(np.dot(v12, v23) / (denom + 1e-8))

    print(f"\n  MMD-based drift:")
    print(f"    MMD^2(D1,D2): {mmd12:.6f}")
    print(f"    MMD^2(D2,D3): {mmd23:.6f}")
    print(f"    MMD^2(D1,D3): {mmd13:.6f}")
    print(f"    Acceleration: {mmd_accel:+.6f}")

    print(f"\n  Centroid-based drift:")
    print(f"    Speed D1->D2: {speed12:.4f}")
    print(f"    Speed D2->D3: {speed23:.4f}")
    print(f"    Acceleration: {accel_mag:.4f}")
    print(f"    Direction cosine: {dir_cos:.4f}")

    if mmd_accel > 0.01:
        interp = "Drift is accelerating"
    elif mmd_accel < -0.01:
        interp = "Drift is decelerating"
    else:
        interp = "Drift speed is roughly constant"
    print(f"  Interpretation: {interp}")

    # Plot: drift magnitude vs time gap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gaps = ['D1-D2', 'D2-D3', 'D1-D3']
    mmds = [float(mmd12), float(mmd23), float(mmd13)]
    axes[0].bar(gaps, mmds, color=['#2196F3', '#FF9800', '#F44336'], edgecolor='black')
    axes[0].set_ylabel('MMD^2')
    axes[0].set_title('MMD^2 vs Domain Pair')
    speeds = [speed12, speed23]
    axes[1].plot(['D1->D2', 'D2->D3'], speeds, 'o-', color='#2196F3', linewidth=2, markersize=8)
    axes[1].set_ylabel('Centroid Speed')
    axes[1].set_title('Drift Velocity Over Time')
    fig.suptitle('Experiment 11: Drift Velocity & Acceleration', fontsize=14)
    fig.tight_layout()
    _save_plot(fig, 'exp11_drift_velocity.png')

    return {
        'mmd12': round(float(mmd12), 6), 'mmd23': round(float(mmd23), 6),
        'mmd13': round(float(mmd13), 6), 'mmd_acceleration': round(mmd_accel, 6),
        'speed12': round(speed12, 4), 'speed23': round(speed23, 4),
        'centroid_acceleration': round(accel_mag, 4),
        'direction_cosine': round(dir_cos, 4), 'interpretation': interp,
    }


# =============================================================================
# Experiment 12: Feature Variance Evolution
# =============================================================================
def exp12_feature_variance_evolution(F1, F2, F3):
    """Track per-feature variance across D1, D2, D3. Generates variance heatmap.

    Returns
    -------
    dict
        Per-domain variance stats and unstable feature indices.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 12: Feature Variance Evolution")
    print("=" * 70)

    var1 = np.var(F1, axis=0)
    var2 = np.var(F2, axis=0)
    var3 = np.var(F3, axis=0)

    # Stack for heatmap: (3, n_features)
    var_matrix = np.stack([var1, var2, var3])

    print(f"  Feature dim: {F1.shape[1]}")
    print(f"  Mean variance — D1: {var1.mean():.4f}, D2: {var2.mean():.4f}, D3: {var3.mean():.4f}")
    print(f"  Max variance  — D1: {var1.max():.4f}, D2: {var2.max():.4f}, D3: {var3.max():.4f}")

    # Identify unstable features (variance changes > 2x)
    ratio_12 = var2 / (var1 + 1e-8)
    ratio_23 = var3 / (var2 + 1e-8)
    unstable = np.where((ratio_12 > 2) | (ratio_12 < 0.5) | (ratio_23 > 2) | (ratio_23 < 0.5))[0]
    print(f"  Unstable features (>2x variance change): {len(unstable)} / {F1.shape[1]}")

    # Heatmap — show top 50 features by max variance
    n_show = min(50, F1.shape[1])
    top_idx = np.argsort(var_matrix.max(axis=0))[::-1][:n_show]
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(var_matrix[:, top_idx], ax=ax, yticklabels=['D1', 'D2', 'D3'],
                xticklabels=[str(i) for i in top_idx], cmap='YlOrRd', cbar_kws={'label': 'Variance'})
    ax.set_xlabel('Feature Index')
    ax.set_title(f'Feature Variance Heatmap (top {n_show} features)')
    fig.tight_layout()
    _save_plot(fig, 'exp12_variance_heatmap.png')

    return {
        'mean_var': {'D1': round(float(var1.mean()), 4), 'D2': round(float(var2.mean()), 4),
                     'D3': round(float(var3.mean()), 4)},
        'n_unstable': int(len(unstable)),
        'unstable_features': unstable.tolist()[:20],
    }


# =============================================================================
# Experiment 13: Feature Distribution Shift Heatmap (KS test)
# =============================================================================
def exp13_feature_ks_heatmap(F1, F2, F3):
    """Per-feature KS test across domain pairs. Generates shift heatmap.

    Returns
    -------
    dict
        KS statistic matrices and significantly shifted feature counts.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 13: Feature Distribution Shift (KS Test)")
    print("=" * 70)

    d = F1.shape[1]
    pairs = [('D1-D2', F1, F2), ('D2-D3', F2, F3), ('D1-D3', F1, F3)]
    ks_matrix = np.zeros((3, d))
    pval_matrix = np.zeros((3, d))

    for pi, (pname, Fa, Fb) in enumerate(pairs):
        for j in range(d):
            stat, pval = ks_2samp(Fa[:, j], Fb[:, j])
            ks_matrix[pi, j] = stat
            pval_matrix[pi, j] = pval

    sig_counts = {}
    for pi, (pname, _, _) in enumerate(pairs):
        n_sig = int((pval_matrix[pi] < 0.05).sum())
        sig_counts[pname] = n_sig
        print(f"  {pname}: {n_sig}/{d} features significantly shifted (p<0.05)")

    # Heatmap — top 50 features by max KS statistic
    n_show = min(50, d)
    top_idx = np.argsort(ks_matrix.max(axis=0))[::-1][:n_show]
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(ks_matrix[:, top_idx], ax=ax, yticklabels=['D1-D2', 'D2-D3', 'D1-D3'],
                xticklabels=[str(i) for i in top_idx], cmap='Reds', vmin=0, vmax=1,
                cbar_kws={'label': 'KS Statistic'})
    ax.set_xlabel('Feature Index')
    ax.set_title(f'Feature Distribution Shift Heatmap (top {n_show})')
    fig.tight_layout()
    _save_plot(fig, 'exp13_ks_heatmap.png')

    return {'sig_counts': sig_counts, 'mean_ks': {
        'D1-D2': round(float(ks_matrix[0].mean()), 4),
        'D2-D3': round(float(ks_matrix[1].mean()), 4),
        'D1-D3': round(float(ks_matrix[2].mean()), 4)}}


# =============================================================================
# Experiment 14: Class Boundary Deformation (Fisher Margin)
# =============================================================================
def exp14_class_boundary_deformation(F1, Y1, F2, Y2, F3, Y3, label_map):
    """Measure Fisher discriminant ratio per domain to detect margin collapse.

    Fisher ratio = between-class variance / within-class variance.

    Returns
    -------
    dict
        Per-domain Fisher ratios and pairwise margin distances.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 14: Class Boundary Deformation")
    print("=" * 70)

    inv_map = {v: k for k, v in label_map.items()}

    def _fisher_ratio(F, Y):
        classes = sorted(np.unique(Y))
        grand_mean = F.mean(0)
        Sw = np.zeros((F.shape[1], F.shape[1]))
        Sb = np.zeros((F.shape[1], F.shape[1]))
        for c in classes:
            Fc = F[Y == c]
            mu_c = Fc.mean(0)
            diff = Fc - mu_c
            Sw += diff.T @ diff
            d = (mu_c - grand_mean).reshape(-1, 1)
            Sb += len(Fc) * (d @ d.T)
        trace_sw = np.trace(Sw) + 1e-8
        trace_sb = np.trace(Sb)
        return float(trace_sb / trace_sw)

    def _pairwise_margin(F, Y):
        classes = sorted(np.unique(Y))
        margins = []
        for i, c1 in enumerate(classes):
            for c2 in classes[i+1:]:
                mu1 = F[Y == c1].mean(0)
                mu2 = F[Y == c2].mean(0)
                var1 = np.var(F[Y == c1], axis=0).mean()
                var2 = np.var(F[Y == c2], axis=0).mean()
                dist = float(np.linalg.norm(mu1 - mu2))
                margin = dist / (np.sqrt(var1 + var2) + 1e-8)
                margins.append(margin)
        return float(np.mean(margins))

    domains = {'D1': (F1, Y1), 'D2': (F2, Y2), 'D3': (F3, Y3)}
    fisher_scores = {}
    margin_scores = {}
    for name, (F, Y) in domains.items():
        fisher_scores[name] = round(_fisher_ratio(F, Y), 4)
        margin_scores[name] = round(_pairwise_margin(F, Y), 4)

    print(f"\n  {'Domain':<8} | {'Fisher Ratio':>14} {'Mean Margin':>14}")
    print("  " + "-" * 40)
    for name in ['D1', 'D2', 'D3']:
        print(f"  {name:<8} | {fisher_scores[name]:>14.4f} {margin_scores[name]:>14.4f}")

    if margin_scores['D3'] < margin_scores['D1'] * 0.7:
        print("  Interpretation: Class boundaries collapsing over time")
    elif margin_scores['D3'] > margin_scores['D1'] * 1.3:
        print("  Interpretation: Class boundaries expanding over time")
    else:
        print("  Interpretation: Class boundaries relatively stable")

    return {'fisher': fisher_scores, 'margins': margin_scores}


# =============================================================================
# Experiment 15: Temporal Feature Collapse Detection
# =============================================================================
def exp15_feature_collapse(F1, F2, F3):
    """Track PCA explained variance curves per domain to detect collapse.

    Returns
    -------
    dict
        Per-domain explained variance ratios and effective dimensionality.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 15: Temporal Feature Collapse Detection")
    print("=" * 70)

    domains = {'D1': F1, 'D2': F2, 'D3': F3}
    results = {}
    k = min(50, F1.shape[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'D1': '#2196F3', 'D2': '#FF9800', 'D3': '#F44336'}

    for name, F in domains.items():
        centered = F - F.mean(0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        var_explained = (S ** 2) / (S ** 2).sum()
        cum_var = np.cumsum(var_explained[:k])

        # Participation ratio = (sum eigenvalues)^2 / sum(eigenvalues^2)
        eigs = S[:k] ** 2
        pr = float((eigs.sum()) ** 2 / (eigs ** 2).sum())

        # Effective dim at 95% variance
        eff_dim_95 = int(np.searchsorted(cum_var, 0.95) + 1)

        results[name] = {
            'participation_ratio': round(pr, 2),
            'effective_dim_95': eff_dim_95,
            'cum_var_10': round(float(cum_var[min(9, k-1)]), 4),
            'cum_var_20': round(float(cum_var[min(19, k-1)]), 4),
        }
        print(f"  {name}: PR={pr:.1f}, EffDim95={eff_dim_95}, CumVar@10={cum_var[min(9,k-1)]:.4f}")

        ax.plot(range(1, k+1), cum_var, label=f'{name} (PR={pr:.1f})', color=colors[name], linewidth=2)

    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='95% threshold')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA Explained Variance Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, 'exp15_explained_variance.png')

    return results


# =============================================================================
# Experiment 16: Cross-Domain Confusion Matrices
# =============================================================================
def exp16_cross_domain_confusion(F1, Y1, F2, Y2, F3, Y3, label_map):
    """Train on D1, test on D2 and D3. Generate confusion matrix heatmaps.

    Returns
    -------
    dict
        Accuracy and confusion matrices for each cross-domain test.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 16: Cross-Domain Confusion Matrices")
    print("=" * 70)

    inv_map = {v: k for k, v in label_map.items()}
    classes = sorted(np.unique(np.concatenate([Y1, Y2, Y3])))
    class_names = [inv_map.get(c, str(c)) for c in classes]

    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(F1, Y1)

    test_pairs = [('D1->D2', F2, Y2), ('D1->D3', F3, Y3)]
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (pname, Ft, Yt) in enumerate(test_pairs):
        pred = clf.predict(Ft)
        acc = float(accuracy_score(Yt, pred))
        cm = confusion_matrix(Yt, pred, labels=classes)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        results[pname] = {'accuracy': round(acc, 4)}
        print(f"  {pname}: Accuracy = {acc:.4f}")

        sns.heatmap(cm_norm, ax=axes[idx], annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
        axes[idx].set_title(f'{pname} (Acc={acc:.3f})')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')

    fig.suptitle('Cross-Domain Confusion Matrices (trained on D1)', fontsize=14)
    fig.tight_layout()
    _save_plot(fig, 'exp16_confusion_matrices.png')

    return results


# =============================================================================
# Experiment 17: Representation Stability Score
# =============================================================================
def exp17_representation_stability(F1, Y1, F2, Y2, F3, Y3, label_map):
    """Compute per-class cosine similarity of mean representations across domains.

    Returns
    -------
    dict
        Per-class and global stability scores.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 17: Representation Stability Score")
    print("=" * 70)

    inv_map = {v: k for k, v in label_map.items()}
    classes = sorted(np.unique(np.concatenate([Y1, Y2, Y3])))

    rows = []
    for c in classes:
        m1, m2, m3 = Y1 == c, Y2 == c, Y3 == c
        if m1.sum() == 0 or m2.sum() == 0 or m3.sum() == 0:
            continue
        mu1, mu2, mu3 = F1[m1].mean(0), F2[m2].mean(0), F3[m3].mean(0)
        cos12 = float(1.0 - cosine_dist(mu1, mu2))
        cos23 = float(1.0 - cosine_dist(mu2, mu3))
        cos13 = float(1.0 - cosine_dist(mu1, mu3))
        name = inv_map.get(c, str(c))
        rows.append({'name': name, 'cos12': cos12, 'cos23': cos23, 'cos13': cos13})

    print(f"\n  {'Class':<12} | {'D1-D2':>8} {'D2-D3':>8} {'D1-D3':>8}")
    print("  " + "-" * 42)
    for r in rows:
        print(f"  {r['name']:<12} | {r['cos12']:>8.4f} {r['cos23']:>8.4f} {r['cos13']:>8.4f}")

    global_cos12 = float(1.0 - cosine_dist(F1.mean(0), F2.mean(0)))
    global_cos23 = float(1.0 - cosine_dist(F2.mean(0), F3.mean(0)))
    global_cos13 = float(1.0 - cosine_dist(F1.mean(0), F3.mean(0)))
    print(f"\n  Global:      | {global_cos12:>8.4f} {global_cos23:>8.4f} {global_cos13:>8.4f}")

    mean_stability = float(np.mean([r['cos13'] for r in rows]))
    print(f"  Mean class stability (D1-D3): {mean_stability:.4f}")

    return {
        'per_class': {r['name']: {'cos12': round(r['cos12'], 4), 'cos23': round(r['cos23'], 4),
                                   'cos13': round(r['cos13'], 4)} for r in rows},
        'global': {'cos12': round(global_cos12, 4), 'cos23': round(global_cos23, 4),
                   'cos13': round(global_cos13, 4)},
        'mean_stability': round(mean_stability, 4),
    }


# =============================================================================
# Experiment 18: Intrinsic Dimensionality Estimation
# =============================================================================
def exp18_intrinsic_dimensionality(F1, F2, F3):
    """Estimate intrinsic dimensionality via PCA, participation ratio, and MLE.

    Returns
    -------
    dict
        Per-domain intrinsic dimension estimates.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 18: Intrinsic Dimensionality Estimation")
    print("=" * 70)

    def _mle_id(X, k=10):
        """MLE intrinsic dimension estimator (Levina-Bickel)."""
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k+1).fit(X)
        dists, _ = nn.kneighbors(X)
        dists = dists[:, 1:]  # exclude self
        dists = np.clip(dists, 1e-10, None)
        log_ratios = np.log(dists[:, -1:] / dists[:, :-1])
        m_hat = (k - 1) / log_ratios.sum(axis=1)
        return float(np.mean(m_hat))

    domains = {'D1': F1, 'D2': F2, 'D3': F3}
    results = {}
    k = min(50, F1.shape[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'D1': '#2196F3', 'D2': '#FF9800', 'D3': '#F44336'}

    for name, F in domains.items():
        centered = F - F.mean(0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        eigs = S[:k] ** 2
        var_ratio = eigs / eigs.sum()
        cum_var = np.cumsum(var_ratio)

        pr = float(eigs.sum() ** 2 / (eigs ** 2).sum())
        eff_95 = int(np.searchsorted(cum_var, 0.95) + 1)

        # MLE on subsample for speed
        max_n = min(500, len(F))
        Fsub = F[np.random.choice(len(F), max_n, replace=False)] if len(F) > max_n else F
        mle_dim = _mle_id(Fsub, k=min(10, max_n - 1))

        results[name] = {
            'participation_ratio': round(pr, 2),
            'pca_95': eff_95,
            'mle_dimension': round(mle_dim, 2),
        }
        print(f"  {name}: PR={pr:.1f}, PCA@95%={eff_95}, MLE={mle_dim:.1f}")
        ax.plot(range(1, k+1), cum_var, label=f'{name}', color=colors[name], linewidth=2)

    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Intrinsic Dimensionality (PCA)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, 'exp18_intrinsic_dim.png')

    return results


# =============================================================================
# Experiment 19: Feature Correlation Structure
# =============================================================================
def exp19_feature_correlation(F1, F2, F3):
    """Compute and visualize feature correlation matrices per domain.

    Returns
    -------
    dict
        Mean absolute correlation and number of highly correlated pairs.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 19: Feature Correlation Structure")
    print("=" * 70)

    domains = {'D1': F1, 'D2': F2, 'D3': F3}
    results = {}
    n_show = min(50, F1.shape[1])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, F) in enumerate(domains.items()):
        corr = np.corrcoef(F.T)
        mean_abs = float(np.mean(np.abs(corr[np.triu_indices_from(corr, k=1)])))
        n_high = int((np.abs(corr[np.triu_indices_from(corr, k=1)]) > 0.8).sum())
        results[name] = {'mean_abs_corr': round(mean_abs, 4), 'n_high_corr_pairs': n_high}
        print(f"  {name}: Mean|corr|={mean_abs:.4f}, High-corr pairs(>0.8)={n_high}")

        sns.heatmap(corr[:n_show, :n_show], ax=axes[idx], cmap='RdBu_r', vmin=-1, vmax=1,
                    square=True, cbar_kws={'shrink': 0.8})
        axes[idx].set_title(f'{name} Correlation (top {n_show})')

    fig.suptitle('Feature Correlation Heatmaps', fontsize=14)
    fig.tight_layout()
    _save_plot(fig, 'exp19_correlation_heatmap.png')

    return results


# =============================================================================
# Experiment 20: Class Separability Score
# =============================================================================
def exp20_class_separability(F1, Y1, F2, Y2, F3, Y3):
    """Compute Fisher discriminant ratio, Silhouette score, Davies-Bouldin index.

    Returns
    -------
    dict
        Per-domain separability metrics.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 20: Class Separability Score")
    print("=" * 70)

    def _fisher(F, Y):
        classes = sorted(np.unique(Y))
        grand_mean = F.mean(0)
        Sw_trace, Sb_trace = 0.0, 0.0
        for c in classes:
            Fc = F[Y == c]
            mu_c = Fc.mean(0)
            Sw_trace += np.sum((Fc - mu_c) ** 2)
            Sb_trace += len(Fc) * np.sum((mu_c - grand_mean) ** 2)
        return float(Sb_trace / (Sw_trace + 1e-8))

    domains = {'D1': (F1, Y1), 'D2': (F2, Y2), 'D3': (F3, Y3)}
    results = {}

    print(f"\n  {'Domain':<8} | {'Fisher':>10} {'Silhouette':>12} {'DB Index':>10}")
    print("  " + "-" * 46)

    for name, (F, Y) in domains.items():
        fisher = _fisher(F, Y)
        # Subsample for silhouette/DB (expensive)
        max_n = min(1000, len(F))
        idx = np.random.choice(len(F), max_n, replace=False) if len(F) > max_n else np.arange(len(F))
        sil = float(silhouette_score(F[idx], Y[idx]))
        db = float(davies_bouldin_score(F[idx], Y[idx]))
        results[name] = {'fisher': round(fisher, 4), 'silhouette': round(sil, 4),
                         'davies_bouldin': round(db, 4)}
        print(f"  {name:<8} | {fisher:>10.4f} {sil:>12.4f} {db:>10.4f}")

    return results


# =============================================================================
# Experiment 21: Cluster Structure Analysis (k-means + t-SNE)
# =============================================================================
def exp21_cluster_structure(F1, Y1, F2, Y2, F3, Y3, label_map):
    """Run k-means and generate t-SNE visualizations per domain.

    Returns
    -------
    dict
        k-means accuracy (via majority vote) and cluster purity.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 21: Cluster Structure Analysis")
    print("=" * 70)

    inv_map = {v: k for k, v in label_map.items()}
    n_classes = len(np.unique(Y1))
    domains = {'D1': (F1, Y1), 'D2': (F2, Y2), 'D3': (F3, Y3)}
    results = {}

    # t-SNE on combined data (subsample for speed)
    max_per_domain = min(300, len(F1), len(F2), len(F3))
    combined_F, combined_Y, combined_D = [], [], []
    for name, (F, Y) in domains.items():
        idx = np.random.choice(len(F), max_per_domain, replace=False) if len(F) > max_per_domain else np.arange(len(F))
        combined_F.append(F[idx])
        combined_Y.append(Y[idx])
        combined_D.extend([name] * len(idx))
    combined_F = np.concatenate(combined_F)
    combined_Y = np.concatenate(combined_Y)

    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_F) - 1))
    emb = tsne.fit_transform(combined_F)

    # Figure 1: colored by class
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    class_labels = sorted(np.unique(combined_Y))
    cmap = plt.cm.get_cmap('tab10', len(class_labels))
    for i, c in enumerate(class_labels):
        mask = combined_Y == c
        name = inv_map.get(c, str(c))
        axes[0].scatter(emb[mask, 0], emb[mask, 1], c=[cmap(i)], label=name, s=10, alpha=0.6)
    axes[0].set_title('t-SNE colored by Activity')
    axes[0].legend(markerscale=3, fontsize=8)

    # Figure 2: colored by domain
    domain_colors = {'D1': '#2196F3', 'D2': '#FF9800', 'D3': '#F44336'}
    for dname in ['D1', 'D2', 'D3']:
        mask = np.array([d == dname for d in combined_D])
        axes[1].scatter(emb[mask, 0], emb[mask, 1], c=domain_colors[dname], label=dname, s=10, alpha=0.6)
    axes[1].set_title('t-SNE colored by Domain')
    axes[1].legend(markerscale=3)

    fig.suptitle('Cluster Structure (t-SNE)', fontsize=14)
    fig.tight_layout()
    _save_plot(fig, 'exp21_tsne.png')

    # k-means per domain
    for name, (F, Y) in domains.items():
        km = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        pred = km.fit_predict(F)
        # Cluster purity
        purity = 0
        for cl in range(n_classes):
            mask = pred == cl
            if mask.sum() > 0:
                counts = np.bincount(Y[mask].astype(int), minlength=n_classes)
                purity += counts.max()
        purity = float(purity / len(Y))
        results[name] = {'cluster_purity': round(purity, 4)}
        print(f"  {name}: k-means purity = {purity:.4f}")

    return results


# =============================================================================
# Experiment 22: Noise & Stability Analysis
# =============================================================================
def exp22_noise_stability(F1, Y1, F2, Y2, F3, Y3, label_map):
    """Compute within-class variance, between-class variance, and SNR.

    Returns
    -------
    dict
        Per-domain noise metrics.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 22: Noise & Stability Analysis")
    print("=" * 70)

    inv_map = {v: k for k, v in label_map.items()}

    def _snr_metrics(F, Y):
        classes = sorted(np.unique(Y))
        grand_mean = F.mean(0)
        within_var, between_var = 0.0, 0.0
        n_total = len(F)
        for c in classes:
            Fc = F[Y == c]
            mu_c = Fc.mean(0)
            within_var += np.sum(np.var(Fc, axis=0))
            between_var += len(Fc) * np.sum((mu_c - grand_mean) ** 2)
        within_var /= n_total
        between_var /= n_total
        snr = float(between_var / (within_var + 1e-8))
        return float(within_var), float(between_var), snr

    domains = {'D1': (F1, Y1), 'D2': (F2, Y2), 'D3': (F3, Y3)}
    results = {}

    print(f"\n  {'Domain':<8} | {'Within-Var':>12} {'Between-Var':>13} {'SNR':>8}")
    print("  " + "-" * 48)
    for name, (F, Y) in domains.items():
        wv, bv, snr = _snr_metrics(F, Y)
        results[name] = {'within_var': round(wv, 4), 'between_var': round(bv, 4), 'snr': round(snr, 4)}
        print(f"  {name:<8} | {wv:>12.4f} {bv:>13.4f} {snr:>8.4f}")

    if results['D3']['snr'] < results['D1']['snr'] * 0.7:
        print("  Interpretation: SNR degrading over time — signal quality declining")
    else:
        print("  Interpretation: SNR relatively stable")

    return results


# =============================================================================
# Publication Tables & Figures
# =============================================================================
def generate_publication_tables(F1, Y1, F2, Y2, F3, Y3, label_map, all_results):
    """Generate publication-quality CSV tables."""
    _ensure_dirs()
    inv_map = {v: k for k, v in label_map.items()}

    # Table A: Dataset Summary
    rows = []
    for name, F, Y in [('D1', F1, Y1), ('D2', F2, Y2), ('D3', F3, Y3)]:
        idim = all_results.get('exp18', {}).get(name, {})
        rows.append([name, len(F), F.shape[1], len(np.unique(Y)),
                     idim.get('participation_ratio', ''), idim.get('mle_dimension', '')])
    _save_csv('table_a_dataset_summary.csv', rows,
              ['Dataset', 'Samples', 'Features', 'Classes', 'Intrinsic Dim (PR)', 'Intrinsic Dim (MLE)'])

    # Table B: Drift Summary
    rows = []
    for pair in ['D1->D2', 'D2->D3', 'D1->D3']:
        pr = all_results.get(pair, {})
        dc = pr.get('domain_classifier', {})
        cs = pr.get('covariance_shift', {})
        mmd = pr.get('mmd', {})
        rows.append([pair, mmd.get('mmd_global', ''), dc.get('linear_acc', ''),
                     cs.get('mean_principal_angle_deg', ''),
                     pr.get('centroid_shift', {}).get('mean_l2', '')])
    _save_csv('table_b_drift_summary.csv', rows,
              ['Comparison', 'MMD^2', 'Domain Acc', 'Principal Angle', 'Centroid Shift'])

    # Table C: Class Drift Summary
    rows = []
    classes = sorted(np.unique(np.concatenate([Y1, Y2, Y3])))
    for c in classes:
        name = inv_map.get(c, str(c))
        # Get centroid shift from D1->D3
        cs = all_results.get('D1->D3', {}).get('centroid_shift', {}).get('per_class', {})
        mmd_pc = all_results.get('D1->D3', {}).get('mmd', {}).get('per_class_mmd', {})
        l2 = cs.get(name, {}).get('l2', '') if isinstance(cs.get(name), dict) else ''
        mmd_v = mmd_pc.get(int(c), '')
        rows.append([name, l2, mmd_v])
    _save_csv('table_c_class_drift.csv', rows, ['Class', 'Centroid Shift (L2)', 'MMD^2'])

    # Table D: Separability
    rows = []
    sep = all_results.get('exp20', {})
    for name in ['D1', 'D2', 'D3']:
        s = sep.get(name, {})
        rows.append([name, s.get('silhouette', ''), s.get('fisher', ''), s.get('davies_bouldin', '')])
    _save_csv('table_d_separability.csv', rows, ['Dataset', 'Silhouette', 'Fisher Score', 'DB Index'])

    print("  Publication tables generated.")


def generate_publication_figures(F1, Y1, F2, Y2, F3, Y3, label_map, all_results):
    """Generate publication-quality figures."""
    _ensure_dirs()
    inv_map = {v: k for k, v in label_map.items()}

    # Figure 1 & 2: PCA projection per dataset
    print("  Generating PCA projections...")
    max_n = min(500, len(F1), len(F2), len(F3))
    combined = np.concatenate([
        F1[np.random.choice(len(F1), max_n, replace=False)] if len(F1) > max_n else F1,
        F2[np.random.choice(len(F2), max_n, replace=False)] if len(F2) > max_n else F2,
        F3[np.random.choice(len(F3), max_n, replace=False)] if len(F3) > max_n else F3,
    ])
    combined_labels = np.concatenate([
        Y1[np.random.choice(len(Y1), max_n, replace=False)] if len(Y1) > max_n else Y1,
        Y2[np.random.choice(len(Y2), max_n, replace=False)] if len(Y2) > max_n else Y2,
        Y3[np.random.choice(len(Y3), max_n, replace=False)] if len(Y3) > max_n else Y3,
    ])
    domain_ids = np.concatenate([np.full(min(max_n, len(F1)), 0),
                                  np.full(min(max_n, len(F2)), 1),
                                  np.full(min(max_n, len(F3)), 2)])

    # PCA
    centered = combined - combined.mean(0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca2d = centered @ Vt[:2].T

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    domain_colors = ['#2196F3', '#FF9800', '#F44336']
    domain_names = ['D1', 'D2', 'D3']
    for d in range(3):
        mask = domain_ids == d
        axes[0].scatter(pca2d[mask, 0], pca2d[mask, 1], c=domain_colors[d],
                       label=domain_names[d], s=10, alpha=0.5)
    axes[0].set_title('PCA Projection (by Domain)')
    axes[0].legend(markerscale=3)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    class_labels = sorted(np.unique(combined_labels))
    cmap = plt.cm.get_cmap('tab10', len(class_labels))
    for i, c in enumerate(class_labels):
        mask = combined_labels == c
        name = inv_map.get(c, str(c))
        axes[1].scatter(pca2d[mask, 0], pca2d[mask, 1], c=[cmap(i)], label=name, s=10, alpha=0.5)
    axes[1].set_title('PCA Projection (by Class)')
    axes[1].legend(markerscale=3, fontsize=8)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')

    fig.suptitle('Figure 1: PCA Projections', fontsize=14)
    fig.tight_layout()
    _save_plot(fig, 'fig1_pca_projection.png')

    # Figure 3: Drift magnitude vs time
    print("  Generating drift magnitude plot...")
    exp11 = all_results.get('exp11', {})
    fig, ax = plt.subplots(figsize=(8, 5))
    pairs_labels = ['D1-D2', 'D2-D3', 'D1-D3']
    mmds = [exp11.get('mmd12', 0), exp11.get('mmd23', 0), exp11.get('mmd13', 0)]
    bars = ax.bar(pairs_labels, mmds, color=['#2196F3', '#FF9800', '#F44336'], edgecolor='black')
    ax.set_ylabel('MMD^2')
    ax.set_title('Figure 3: Drift Magnitude vs Domain Pair')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mmds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}',
                ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    _save_plot(fig, 'fig3_drift_magnitude.png')

    # Figure 7: Domain classifier ROC curves (from pairwise results)
    print("  Generating domain classifier bar chart...")
    fig, ax = plt.subplots(figsize=(8, 5))
    pair_names = ['D1->D2', 'D2->D3', 'D1->D3']
    lin_accs = [all_results.get(p, {}).get('domain_classifier', {}).get('linear_acc', 0) for p in pair_names]
    mlp_accs = [all_results.get(p, {}).get('domain_classifier', {}).get('mlp_acc', 0) for p in pair_names]
    x = np.arange(len(pair_names))
    w = 0.35
    ax.bar(x - w/2, lin_accs, w, label='Linear', color='#2196F3', edgecolor='black')
    ax.bar(x + w/2, mlp_accs, w, label='MLP', color='#FF9800', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.set_ylabel('Accuracy')
    ax.set_title('Figure 7: Domain Classifier Accuracy')
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    _save_plot(fig, 'fig7_domain_classifier.png')

    # Figure 8: Class centroid trajectories
    print("  Generating centroid trajectory plot...")
    classes = sorted(np.unique(Y1))
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap_cls = plt.cm.get_cmap('tab10', len(classes))
    for i, c in enumerate(classes):
        m1, m2, m3 = Y1 == c, Y2 == c, Y3 == c
        if m1.sum() == 0 or m2.sum() == 0 or m3.sum() == 0:
            continue
        d12 = float(np.linalg.norm(F1[m1].mean(0) - F2[m2].mean(0)))
        d13 = float(np.linalg.norm(F1[m1].mean(0) - F3[m3].mean(0)))
        name = inv_map.get(c, str(c))
        ax.plot([0, 1, 2], [0, d12, d13], 'o-', color=cmap_cls(i), label=name, linewidth=2, markersize=8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['D1 (origin)', 'D2', 'D3'])
    ax.set_ylabel('Centroid Distance from D1')
    ax.set_title('Figure 8: Class Centroid Drift Trajectories')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, 'fig8_centroid_trajectories.png')

    print("  Publication figures generated.")


# =============================================================================
# Drift Severity Score
# =============================================================================
def compute_drift_severity_score(all_results):
    """Compute unified Drift Severity Score (0-1 scale).

    DriftScore = 0.25*norm_MMD + 0.25*domain_acc + 0.25*norm_angle + 0.25*norm_centroid
    """
    print("\n" + "=" * 70)
    print("DRIFT SEVERITY SCORE")
    print("=" * 70)

    # Use D1->D3 (max temporal gap) for scoring
    r = all_results.get('D1->D3', {})
    mmd = r.get('mmd', {}).get('mmd_global', 0)
    domain_acc = r.get('domain_classifier', {}).get('linear_acc', 0.5)
    angle = r.get('covariance_shift', {}).get('mean_principal_angle_deg', 0)
    centroid = r.get('centroid_shift', {}).get('mean_l2', 0)

    # Normalize each to [0, 1]
    norm_mmd = min(float(mmd) / 0.5, 1.0)  # 0.5 is "extreme" MMD
    norm_acc = min(max((float(domain_acc) - 0.5) * 2, 0), 1.0)  # 0.5=no shift, 1.0=perfect separation
    norm_angle = min(float(angle) / 90.0, 1.0)
    norm_centroid = min(float(centroid) / 10.0, 1.0)  # 10.0 is "extreme" shift

    score = 0.25 * norm_mmd + 0.25 * norm_acc + 0.25 * norm_angle + 0.25 * norm_centroid

    print(f"  Components (D1->D3):")
    print(f"    MMD^2:           {mmd:.6f} -> norm={norm_mmd:.4f}")
    print(f"    Domain Acc:      {domain_acc:.4f} -> norm={norm_acc:.4f}")
    print(f"    Principal Angle: {angle:.2f} deg -> norm={norm_angle:.4f}")
    print(f"    Centroid Shift:  {centroid:.4f} -> norm={norm_centroid:.4f}")
    print(f"\n  DRIFT SEVERITY SCORE: {score:.4f}")

    if score < 0.2:
        print("  Level: LOW — minimal adaptation needed")
    elif score < 0.5:
        print("  Level: MODERATE — standard adaptation (CORAL/AdaBN) recommended")
    elif score < 0.8:
        print("  Level: HIGH — aggressive adaptation required")
    else:
        print("  Level: CRITICAL — re-training likely necessary")

    return {'score': round(score, 4), 'norm_mmd': round(norm_mmd, 4),
            'norm_domain_acc': round(norm_acc, 4), 'norm_angle': round(norm_angle, 4),
            'norm_centroid': round(norm_centroid, 4)}


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================
def bootstrap_mmd_ci(F_src, F_tgt, n_bootstrap=100, alpha=0.05):
    """Bootstrap confidence interval for MMD^2."""
    mmds = []
    n_s, n_t = len(F_src), len(F_tgt)
    for _ in range(n_bootstrap):
        idx_s = np.random.choice(n_s, n_s, replace=True)
        idx_t = np.random.choice(n_t, n_t, replace=True)
        max_n = min(200, n_s, n_t)
        mmd2, _ = _compute_mmd(F_src[idx_s[:max_n]], F_tgt[idx_t[:max_n]])
        mmds.append(float(mmd2))
    lo = float(np.percentile(mmds, 100 * alpha / 2))
    hi = float(np.percentile(mmds, 100 * (1 - alpha / 2)))
    return {'mean': round(float(np.mean(mmds)), 6), 'ci_low': round(lo, 6),
            'ci_high': round(hi, 6), 'std': round(float(np.std(mmds)), 6)}


def exp_statistical_significance(F1, F2, F3, n_bootstrap=100):
    """Bootstrap CIs for all pairwise MMD values."""
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (Bootstrap CIs)")
    print("=" * 70)

    pairs = [('D1-D2', F1, F2), ('D2-D3', F2, F3), ('D1-D3', F1, F3)]
    results = {}
    for pname, Fa, Fb in pairs:
        ci = bootstrap_mmd_ci(Fa, Fb, n_bootstrap=n_bootstrap)
        results[pname] = ci
        print(f"  {pname}: MMD^2 = {ci['mean']:.6f} [{ci['ci_low']:.6f}, {ci['ci_high']:.6f}]")

    return results


# =============================================================================
# Master Wrapper: Full 22-Experiment Suite
# =============================================================================
def run_full_drift_suite(F1, Y1, F2, Y2, F3, Y3, label_map, n_bootstrap=50):
    """Run all 22 experiments plus publication outputs.

    Parameters
    ----------
    F1, Y1 : features/labels for D1 (earliest)
    F2, Y2 : features/labels for D2 (middle)
    F3, Y3 : features/labels for D3 (latest)
    label_map : dict mapping label name to int
    n_bootstrap : int, number of bootstrap samples for CIs

    Returns
    -------
    dict : all results
    """
    _ensure_dirs()

    domains = {"D1": (F1, Y1), "D2": (F2, Y2), "D3": (F3, Y3)}
    pairs = [("D1", "D2"), ("D2", "D3"), ("D1", "D3")]
    results = {}

    # ---- Pairwise experiments 1-4 + MMD ----
    for src, tgt in pairs:
        print("\n\n" + "#" * 80)
        print(f"TEMPORAL DRIFT: {src} -> {tgt}")
        print("#" * 80)
        Fs, Ys = domains[src]
        Ft, Yt = domains[tgt]
        pair_key = f"{src}->{tgt}"
        results[pair_key] = {
            "domain_classifier": exp1_domain_classifier(Fs, Ft),
            "centroid_shift": exp2_centroid_shift(Fs, Ys, Ft, Yt, label_map),
            "covariance_shift": exp3_covariance_shift(Fs, Ft),
            "prior_shift": exp4_label_prior_shift(Ys, Yt, label_map),
            "mmd": compute_full_mmd_suite(Fs, Ys, Ft, Yt),
        }

    # ---- Temporal 3-domain experiments 11-17 ----
    print("\n\n" + "#" * 80)
    print("TEMPORAL 3-DOMAIN ANALYSIS (Experiments 11-17)")
    print("#" * 80)

    results["exp11"] = exp11_drift_velocity(F1, F2, F3)
    results["exp12"] = exp12_feature_variance_evolution(F1, F2, F3)
    results["exp13"] = exp13_feature_ks_heatmap(F1, F2, F3)
    results["exp14"] = exp14_class_boundary_deformation(F1, Y1, F2, Y2, F3, Y3, label_map)
    results["exp15"] = exp15_feature_collapse(F1, F2, F3)
    results["exp16"] = exp16_cross_domain_confusion(F1, Y1, F2, Y2, F3, Y3, label_map)
    results["exp17"] = exp17_representation_stability(F1, Y1, F2, Y2, F3, Y3, label_map)

    # ---- Dataset characterization 18-22 ----
    print("\n\n" + "#" * 80)
    print("DATASET CHARACTERIZATION (Experiments 18-22)")
    print("#" * 80)

    results["exp18"] = exp18_intrinsic_dimensionality(F1, F2, F3)
    results["exp19"] = exp19_feature_correlation(F1, F2, F3)
    results["exp20"] = exp20_class_separability(F1, Y1, F2, Y2, F3, Y3)
    results["exp21"] = exp21_cluster_structure(F1, Y1, F2, Y2, F3, Y3, label_map)
    results["exp22"] = exp22_noise_stability(F1, Y1, F2, Y2, F3, Y3, label_map)

    # ---- Statistical significance ----
    results["bootstrap_ci"] = exp_statistical_significance(F1, F2, F3, n_bootstrap=n_bootstrap)

    # ---- Drift severity score ----
    results["drift_score"] = compute_drift_severity_score(results)

    # ---- Publication outputs ----
    print("\n\n" + "#" * 80)
    print("GENERATING PUBLICATION OUTPUTS")
    print("#" * 80)

    generate_publication_tables(F1, Y1, F2, Y2, F3, Y3, label_map, results)
    generate_publication_figures(F1, Y1, F2, Y2, F3, Y3, label_map, results)

    return results


# =============================================================================
# Final Summary
# =============================================================================
def print_full_summary(results):
    """Print comprehensive summary of all 22 experiments."""
    print(f"\n{'='*80}")
    print("FULL DRIFT & DATASET CHARACTERIZATION SUMMARY (22 Experiments)")
    print(f"{'='*80}")

    # Pairwise drift
    for pair in ['D1->D2', 'D2->D3', 'D1->D3']:
        r = results.get(pair, {})
        dc = r.get('domain_classifier', {})
        cs = r.get('centroid_shift', {})
        mmd = r.get('mmd', {})
        cov = r.get('covariance_shift', {})
        print(f"\n  {pair}:")
        print(f"    Domain Acc (Lin/MLP): {dc.get('linear_acc','?')}/{dc.get('mlp_acc','?')}")
        print(f"    Centroid L2/Cosine:   {cs.get('mean_l2','?')}/{cs.get('mean_cosine','?')}")
        print(f"    MMD^2:               {mmd.get('mmd_global','?')}")
        print(f"    Principal Angle:     {cov.get('mean_principal_angle_deg','?')} deg")

    # Temporal dynamics
    e11 = results.get('exp11', {})
    e12 = results.get('exp12', {})
    e13 = results.get('exp13', {})
    e14 = results.get('exp14', {})
    e15 = results.get('exp15', {})
    e17 = results.get('exp17', {})
    print(f"\n  --- Temporal Dynamics ---")
    print(f"  11. Velocity: {e11.get('interpretation','?')}")
    print(f"  12. Unstable features: {e12.get('n_unstable','?')}")
    print(f"  13. KS-shifted features: {e13.get('sig_counts',{})}")
    print(f"  14. Fisher ratios: {e14.get('fisher',{})}")
    eff_dims = ', '.join(f'{k}={v.get("effective_dim_95", "?")}' for k, v in e15.items())
    print(f"  15. Effective dim@95%: {eff_dims}")
    print(f"  17. Mean stability: {e17.get('mean_stability','?')}")

    # Dataset characterization
    e18 = results.get('exp18', {})
    e20 = results.get('exp20', {})
    e22 = results.get('exp22', {})
    print(f"\n  --- Dataset Characterization ---")
    idims = ', '.join(f'{k}=PR{v.get("participation_ratio", "?")}' for k, v in e18.items())
    print(f"  18. Intrinsic dim: {idims}")
    sils = ', '.join(f'{k}=Sil{v.get("silhouette", "?")}' for k, v in e20.items())
    print(f"  20. Separability: {sils}")
    snrs = ', '.join(f'{k}={v.get("snr", "?")}' for k, v in e22.items())
    print(f"  22. SNR: {snrs}")

    # Drift score
    ds = results.get('drift_score', {})
    print(f"\n  DRIFT SEVERITY SCORE: {ds.get('score','?')}")

    # Bootstrap CIs
    ci = results.get('bootstrap_ci', {})
    for p, v in ci.items():
        if isinstance(v, dict):
            print(f"  {p} MMD^2: {v.get('mean','?')} [{v.get('ci_low','?')}, {v.get('ci_high','?')}]")

    print(f"\n{'='*80}")
    print("Analysis complete! Results saved to results/ directory.")
    print(f"{'='*80}")


# =============================================================================
# Load dataset into temporal domains from metadata
# =============================================================================
def load_dataset_temporal_domains(root_dir, pipeline_name='amplitude',
                                  window_len=300, guaranteed_sr=150,
                                  var_window=20, balance=True, verbose=False):
    """Load a metadata-driven dataset and split into 2-3 temporal domains.

    Domain assignment depends on the dataset's split structure:

    - **session-based** (home_har_data): data1 → D1, data2 → D2, data3 → D3.
    - **file-based** (home_occupation, office_localization): files per label
      are ordered chronologically. With 3 files → D1/D2/D3; with 2 files →
      D1/D2.
    - **percentage** (office_har_data): the single file per label is split
      into 3 equal temporal blocks → D1/D2/D3.

    Parameters
    ----------
    root_dir : str
        Path to one dataset folder containing dataset_metadata.json.

    Returns
    -------
    domains : list of (X, y)
        2 or 3 (X, y) tuples ordered earliest → latest.
    label_map : dict
    labels_list : list of str
    ds_name : str
    """
    import json

    root_dir = os.path.abspath(root_dir)
    meta_path = os.path.join(root_dir, METADATA_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No metadata at {meta_path}")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    ds_name = metadata['name']
    labels_list = sorted(metadata['labels'])
    label_map = {lbl: i for i, lbl in enumerate(labels_list)}
    stride = window_len

    loader = CSI_Loader(verbose=verbose)
    loader.guaranteed_sr = guaranteed_sr
    selector = FeatureSelector(mask=CSI_SUBCARRIER_MASK, verbose=verbose)

    print(f"\n[drift] Loading '{ds_name}' for temporal domain analysis "
          f"(pipeline={pipeline_name}, window={window_len})")

    # ----- Phase 1: load all files, keep per-label session order -----
    per_label_sessions = {lbl: [] for lbl in labels_list}

    for entry in metadata['files']:
        fpath = os.path.join(root_dir, entry['path'])
        label = entry['label']
        if label not in label_map:
            continue
        try:
            data = loader.process(fpath)
            data = selector.process(data)
        except Exception as e:
            print(f"  ERROR {entry['path']}: {e}")
            continue

        mag = data['mag']
        cls = TrainingDataset

        if pipeline_name == 'amplitude':
            X = cls._window_array_static(mag, window_len, stride, 'flattened')
        elif pipeline_name == 'rolling_variance':
            rv = cls._rolling_variance(mag, var_window)
            X = cls._window_array_static(rv, window_len, stride, 'flattened')
        elif pipeline_name == 'amplitude_phase':
            phase = data.get('phase')
            mag_w = cls._window_array_static(mag, window_len, stride, 'flattened')
            phase_w = cls._window_array_static(phase, window_len, stride, 'flattened')
            if mag_w is None or phase_w is None:
                continue
            X = np.concatenate([mag_w, phase_w], axis=1)
        else:
            X = cls._window_array_static(mag, window_len, stride, 'flattened')

        if X is None or len(X) == 0:
            continue

        per_label_sessions[label].append(X)
        print(f"  {entry['path']:30s}  label={label:8s}  windows={X.shape[0]}")

    # ----- Phase 2: determine temporal domain structure -----
    is_percentage = all(
        entry.get('split', 'percentage') == 'percentage'
        for entry in metadata['files']
    )

    session_counts = {lbl: len(s) for lbl, s in per_label_sessions.items() if s}
    if not session_counts:
        raise RuntimeError(f"No data loaded for '{ds_name}'")

    min_sessions = min(session_counts.values())

    if is_percentage:
        # Single file per label → split each into 3 temporal blocks
        n_domains = 3
        print(f"[drift] Percentage mode: splitting into {n_domains} temporal blocks")
        domain_data = {d: ([], []) for d in range(n_domains)}

        for lbl in labels_list:
            sessions = per_label_sessions.get(lbl, [])
            if not sessions:
                continue
            X_all = np.concatenate(sessions, axis=0)
            block_size = X_all.shape[0] // n_domains
            if block_size < 1:
                continue
            y_val = label_map[lbl]
            for di in range(n_domains):
                start = di * block_size
                end = start + block_size if di < n_domains - 1 else X_all.shape[0]
                block = X_all[start:end]
                domain_data[di][0].append(block)
                domain_data[di][1].append(np.full(block.shape[0], y_val, dtype=np.int64))
    else:
        # File-based or session-based: each file per label is one session
        n_domains = min(min_sessions, 3)
        print(f"[drift] File/session mode: {min_sessions} min sessions/label "
              f"→ using {n_domains} temporal domains")
        domain_data = {d: ([], []) for d in range(n_domains)}

        for lbl in labels_list:
            sessions = per_label_sessions.get(lbl, [])
            if not sessions:
                continue
            y_val = label_map[lbl]
            for di in range(n_domains):
                if di < len(sessions):
                    domain_data[di][0].append(sessions[di])
                    domain_data[di][1].append(
                        np.full(sessions[di].shape[0], y_val, dtype=np.int64))

    # ----- Phase 3: build domain arrays -----
    domains = []
    for di in range(n_domains):
        Xs, Ys = domain_data[di]
        if not Xs:
            continue
        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)

        # Balance each domain
        if balance:
            classes, counts = np.unique(Y, return_counts=True)
            min_count = counts.min()
            rng = np.random.RandomState(42)
            bal_idx = []
            for c in classes:
                c_idx = np.where(Y == c)[0]
                if len(c_idx) > min_count:
                    c_idx = rng.choice(c_idx, size=min_count, replace=False)
                bal_idx.append(c_idx)
            bal_idx = np.sort(np.concatenate(bal_idx))
            X, Y = X[bal_idx], Y[bal_idx]

        domains.append((X, Y))
        print(f"  D{di+1}: {X.shape}, classes={np.unique(Y).tolist()}")

    if len(domains) < 2:
        raise RuntimeError(f"Need >=2 temporal domains, got {len(domains)} for '{ds_name}'")

    return domains, label_map, labels_list, ds_name


def _run_drift_for_dataset(domains, label_map, labels_list, ds_name,
                           epochs=50, n_bootstrap=50):
    """Run the full drift suite for one dataset's temporal domains.

    Handles both 2-domain and 3-domain cases.  For 2-domain datasets
    the 3-domain experiments (11-17) are run with D2 duplicated as D3
    (results are still meaningful for the pairwise subset).

    Returns
    -------
    dict : full results for this dataset
    """
    n_dom = len(domains)
    X1, Y1 = domains[0]
    X2, Y2 = domains[1]

    if n_dom >= 3:
        X3, Y3 = domains[2]
    else:
        # 2-domain: duplicate D2 as D3 for the 3-domain experiments
        X3, Y3 = X2.copy(), Y2.copy()

    n_features = X1.shape[1]
    n_classes = len(np.unique(np.concatenate([Y1, Y2, Y3])))

    print(f"\n{'='*80}")
    print(f"Training source model on D1 for '{ds_name}'...")
    print(f"{'='*80}")
    model = train_source_model(X1, Y1, n_features, n_classes,
                               epochs=epochs, verbose=True)

    print("\nExtracting features...")
    F1 = extract_features(model, X1)
    F2 = extract_features(model, X2)
    F3 = extract_features(model, X3)
    print(f"  D1 features: {F1.shape}")
    print(f"  D2 features: {F2.shape}")
    print(f"  D3 features: {F3.shape}")

    results = run_full_drift_suite(F1, Y1, F2, Y2, F3, Y3,
                                   label_map, n_bootstrap=n_bootstrap)
    return results


def _save_aggregate_csv(all_dataset_results, results_dir):
    """Save a cross-dataset drift summary CSV."""
    import json as _json

    csv_path = os.path.join(results_dir, 'drift_results_all_datasets.csv')
    rows = []

    for ds_name, results in all_dataset_results.items():
        # Pairwise experiments
        for pair in ['D1->D2', 'D2->D3', 'D1->D3']:
            r = results.get(pair, {})
            dc = r.get('domain_classifier', {})
            cs = r.get('centroid_shift', {})
            mmd = r.get('mmd', {})
            cov = r.get('covariance_shift', {})
            ps = r.get('prior_shift', {})
            rows.append({
                'dataset': ds_name, 'experiment': 'pairwise', 'pair': pair,
                'domain_clf_linear_acc': dc.get('linear_acc'),
                'domain_clf_mlp_acc': dc.get('mlp_acc'),
                'centroid_mean_l2': cs.get('mean_l2'),
                'centroid_mean_cosine': cs.get('mean_cosine'),
                'mmd_global': mmd.get('mmd_global'),
                'cov_principal_angle_deg': cov.get('mean_principal_angle_deg'),
                'prior_kl_div': ps.get('kl_divergence'),
            })
        # Temporal experiments
        for exp_key in ['exp11', 'exp12', 'exp13', 'exp14', 'exp15', 'exp16',
                        'exp17', 'exp18', 'exp19', 'exp20', 'exp21', 'exp22']:
            r = results.get(exp_key, {})
            rows.append({
                'dataset': ds_name, 'experiment': exp_key, 'pair': 'all',
                'summary': _json.dumps(r, default=str)[:500],
            })
        # Drift score
        ds = results.get('drift_score', {})
        rows.append({
            'dataset': ds_name, 'experiment': 'drift_score', 'pair': 'all',
            'drift_severity_score': ds.get('score'),
            'summary': _json.dumps(ds, default=str)[:500],
        })

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n[info] Aggregate drift CSV saved to {os.path.abspath(csv_path)}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Temporal Domain Drift & Dataset Characterization Suite (22 Experiments)')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', '..', 'wifi_sensing_data'),
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--window', type=int, default=300,
                        help='Window length in samples (default: 300)')
    parser.add_argument('--sr', type=int, default=150,
                        help='Guaranteed sample rate (default: 150)')
    parser.add_argument('--pipeline', type=str, default='amplitude',
                        choices=['amplitude', 'rolling_variance', 'amplitude_phase'],
                        help='Feature pipeline (default: amplitude)')
    parser.add_argument('--var-window', type=int, default=20,
                        help='Rolling variance window (default: 20)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs for source model (default: 50)')
    parser.add_argument('--n-bootstrap', type=int, default=50,
                        help='Bootstrap samples for CIs (default: 50)')
    parser.add_argument('--datasets', type=str, nargs='*', default=None,
                        help='Specific dataset folder names to run (default: all 4)')
    args = parser.parse_args()

    DATASET_DIRS = args.datasets or [
        'home_har_data',
        'home_occupation_data',
        'office_har_data',
        'office_localization_data',
    ]

    set_global_seed(42)

    data_root = os.path.abspath(args.data_root)
    print("=" * 80)
    print("PUBLICATION-GRADE TEMPORAL DRIFT & DATASET CHARACTERIZATION SUITE")
    print(f"  22 Experiments | {len(DATASET_DIRS)} Datasets | CSV + Plots")
    print(f"  Data root:  {data_root}")
    print(f"  Pipeline:   {args.pipeline}")
    print(f"  Window:     {args.window} @ {args.sr} Hz")
    print(f"  Epochs:     {args.epochs}")
    print("=" * 80)

    all_dataset_results = {}

    for ds_dir in DATASET_DIRS:
        ds_path = os.path.join(data_root, ds_dir)
        if not os.path.isdir(ds_path):
            print(f"\n[warn] Skipping {ds_dir} — directory not found")
            continue
        meta_path = os.path.join(ds_path, METADATA_FILENAME)
        if not os.path.isfile(meta_path):
            print(f"\n[warn] Skipping {ds_dir} — no {METADATA_FILENAME}")
            continue

        print(f"\n\n{'#'*80}")
        print(f"  DATASET: {ds_dir}")
        print(f"{'#'*80}")

        try:
            domains, label_map, labels_list, ds_name = load_dataset_temporal_domains(
                root_dir=ds_path,
                pipeline_name=args.pipeline,
                window_len=args.window,
                guaranteed_sr=args.sr,
                var_window=args.var_window,
                balance=True,
                verbose=False,
            )

            n_dom = len(domains)
            print(f"\n  '{ds_name}': {n_dom} temporal domains, "
                  f"{len(labels_list)} classes {labels_list}")
            for di, (X, Y) in enumerate(domains):
                print(f"    D{di+1}: {X.shape[0]:5d} samples, "
                      f"{X.shape[1]} features, "
                      f"classes={np.unique(Y).tolist()}")

            # Update plot/results dirs to be per-dataset
            RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', ds_name)
            PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

            results = _run_drift_for_dataset(
                domains, label_map, labels_list, ds_name,
                epochs=args.epochs, n_bootstrap=args.n_bootstrap,
            )

            print_full_summary(results)
            all_dataset_results[ds_name] = results

        except Exception as e:
            print(f"\n[ERROR] {ds_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ---- Aggregate cross-dataset CSV ----
    if all_dataset_results:
        agg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(agg_dir, exist_ok=True)
        _save_aggregate_csv(all_dataset_results, agg_dir)

        # ---- Print cross-dataset comparison ----
        print(f"\n\n{'='*80}")
        print("CROSS-DATASET DRIFT COMPARISON")
        print(f"{'='*80}")
        print(f"\n  {'Dataset':<30} {'Drift Score':>12} {'MMD(D1-D3)':>12} "
              f"{'DomAcc(D1-D3)':>14} {'Angle(D1-D3)':>13}")
        print("  " + "-" * 85)
        for ds_name, results in all_dataset_results.items():
            ds_score = results.get('drift_score', {}).get('score', '?')
            r13 = results.get('D1->D3', {})
            mmd13 = r13.get('mmd', {}).get('mmd_global', '?')
            dom_acc = r13.get('domain_classifier', {}).get('linear_acc', '?')
            angle = r13.get('covariance_shift', {}).get('mean_principal_angle_deg', '?')
            print(f"  {ds_name:<30} {ds_score:>12} {mmd13:>12} {dom_acc:>14} {angle:>13}")

    print(f"\n{'='*80}")
    print("All datasets processed. Results saved to results/ directory.")
    print(f"{'='*80}")
