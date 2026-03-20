#!/usr/bin/env python3
"""
Deep Learning components for CSI-based activity recognition.

Architectures:
- MLP: Multi-layer perceptron classifier
- Conv1DClassifier: Pure 1D-CNN (temporal convolution) classifier
- CnnLstmClassifier: CNN+BiLSTM sequential classifier

Usage:
    python dl.py                              # default paths, 3 seeds
    python dl.py --data-root ../../data --seeds 5
    python dl.py --window 300 --sr 150 --var-windows 20 200 2000
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MLP (Basic Multi-Layer Perceptron)
# =============================================================================
class MLP(nn.Module):
    """Multi-layer perceptron classifier.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int
        Hidden layer dimensions.
    output_dim : int
        Number of output classes.
    dropout : float
        Dropout probability. Default: 0.0
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# 1D-CNN Classifier (Temporal Convolution Network)
# =============================================================================
class Conv1DClassifier(nn.Module):
    """Pure 1D-CNN classifier for sequential CSI data.

    Reshapes flat input (B, T*C) -> (B, C, T), applies stacked Conv1d blocks
    with BatchNorm, ReLU, and MaxPool, then global average pools and classifies.
    Very fast on CPU; captures local temporal patterns without recurrence or
    attention.

    Parameters
    ----------
    n_subcarriers : int
        Number of CSI subcarriers (channels). Default 52.
    window_len : int
        Temporal length of one window. Default 100.
    num_classes : int
        Number of output classes.
    conv_channels : list of int
        Output channels for each Conv1d block.
    kernel_size : int
        Kernel size for all Conv1d layers. Default 7.
    dropout : float
        Dropout probability. Default 0.2.
    use_batch_norm : bool
        Whether to use BatchNorm after each Conv1d. Default True.
    use_whitening : bool
        Whether to use FeatureWhitening after pooling. Default False.

    Example
    -------
    >>> model = Conv1DClassifier(52, 100, num_classes=6)
    >>> logits = model(torch.randn(32, 5200))
    >>> logits.shape
    torch.Size([32, 6])
    """

    def __init__(self, n_subcarriers=52, window_len=100, num_classes=7,
                 conv_channels=None, kernel_size=7, dropout=0.2,
                 use_batch_norm=True, use_whitening=False):
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128, 128]
        self.n_subcarriers = n_subcarriers
        self.window_len = window_len
        self.num_classes = num_classes

        # Build Conv1d blocks
        layers = []
        in_ch = n_subcarriers
        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                                    padding=kernel_size // 2))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.convs = nn.Sequential(*layers)

        repr_dim = conv_channels[-1]
        self.pool_norm = nn.BatchNorm1d(repr_dim) if use_batch_norm else nn.Identity()

        self.use_whitening = use_whitening
        if use_whitening:
            self.whitening = FeatureWhitening(repr_dim)

        # Classification head
        self.label_classifier = nn.Sequential(
            nn.Linear(repr_dim, repr_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(repr_dim // 2, num_classes),
        )
        self._repr_dim = repr_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x):
        B = x.size(0)
        # (B, T*C) -> (B, C, T)
        x = x.view(B, self.window_len, self.n_subcarriers).permute(0, 2, 1)
        x = self.convs(x)
        # Global average pooling over time
        z = x.mean(dim=2)
        z = self.pool_norm(z)
        if self.use_whitening:
            z = self.whitening(z)
        return z

    def forward(self, x):
        z = self.extract_features(x)
        return self.label_classifier(z)

    def predict(self, x, batch_size=256):
        if x.size(0) <= batch_size:
            return self.forward(x)
        parts = [self.forward(x[i:i+batch_size]) for i in range(0, x.size(0), batch_size)]
        return torch.cat(parts, 0)


def make_conv1d_model(n_subcarriers, window_len, n_classes, config='small',
                      use_batch_norm=True, use_whitening=False):
    """Factory for Conv1DClassifier with preset configs."""
    configs = {
        'small': dict(conv_channels=[32, 64],    kernel_size=5, dropout=0.2),
        'mid':   dict(conv_channels=[64, 128],    kernel_size=7, dropout=0.25),
        'large': dict(conv_channels=[64, 128, 256], kernel_size=7, dropout=0.3),
    }
    cfg = configs.get(config, configs['small'])
    return Conv1DClassifier(
        n_subcarriers=n_subcarriers, window_len=window_len,
        num_classes=n_classes, use_batch_norm=use_batch_norm,
        use_whitening=use_whitening, **cfg,
    )


# =============================================================================
# Residual MLP Classifier (MLP with skip connections)
# =============================================================================
class _ResBlock(nn.Module):
    """Single residual block: Linear -> BN -> ReLU -> Dropout -> Linear -> BN + skip."""

    def __init__(self, dim, dropout=0.2, use_batch_norm=True):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return F.relu(x + self.block(x))



# =============================================================================
# Feature Extractor
# =============================================================================
class FeatureExtractor(nn.Module):
    """Feature extraction network for domain adaptation.
    
    Extracts domain-invariant features from input data.
    Used as the shared backbone in DANN architecture.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int
        Hidden layer dimensions.
    output_dim : int
        Dimension of extracted features (embedding size).
    dropout : float
        Dropout probability. Default: 0.0
    batch_norm : bool
        Whether to use batch normalization. Default: True
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, batch_norm=True):
        super().__init__()
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        
        # Final projection to embedding space
        layers.append(nn.Linear(prev, output_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# Label Classifier
# =============================================================================
class LabelClassifier(nn.Module):
    """Label classification head for main task.
    
    Takes features from FeatureExtractor and predicts class labels.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features (from FeatureExtractor).
    hidden_dims : list of int
        Hidden layer dimensions. Can be empty for linear classifier.
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout probability. Default: 0.0
    """
    
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# Deep CORAL Loss
# =============================================================================
def coral_loss(source_features, target_features):
    """Compute Deep CORAL loss between source and target feature batches.
    
    Minimizes the Frobenius norm of the difference between source and target
    covariance matrices, aligning the full second-order statistics.
    
    L_CORAL = ||C_s - C_t||_F^2 / (4 * d^2)
    
    Parameters
    ----------
    source_features : torch.Tensor
        Source domain features of shape (n_s, d).
    target_features : torch.Tensor
        Target domain features of shape (n_t, d).
    
    Returns
    -------
    torch.Tensor
        Scalar CORAL loss.
    
    Reference
    ---------
    Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016
    """
    d = source_features.size(1)
    n_s = source_features.size(0)
    n_t = target_features.size(0)
    
    # Center features
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)
    
    # Covariance matrices
    cov_s = (source_centered.t() @ source_centered) / max(n_s - 1, 1)
    cov_t = (target_centered.t() @ target_centered) / max(n_t - 1, 1)
    
    # Frobenius norm squared of difference, normalized by 4*d^2
    loss = (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)
    return loss


# =============================================================================
# Class-Conditional CORAL Loss
# =============================================================================
def conditional_coral_loss(source_features, target_features, source_labels,
                           target_logits, confidence_threshold=0.8):
    """Class-conditional CORAL: align per-class covariances using pseudo-labels.
    
    Instead of global ||C_s - C_t||_F, computes:
        sum_c ||C_s^c - C_t^c||_F
    
    Target pseudo-labels are obtained from model predictions, thresholded
    by confidence to avoid noisy alignment.
    
    Parameters
    ----------
    source_features : torch.Tensor
        Source features (n_s, d).
    target_features : torch.Tensor
        Target features (n_t, d). Must be detached or from no_grad context.
    source_labels : torch.Tensor
        Source ground-truth labels (n_s,).
    target_logits : torch.Tensor
        Target logits (n_t, n_classes). Used to derive pseudo-labels.
    confidence_threshold : float
        Only use target samples with max softmax prob >= threshold.
    
    Returns
    -------
    torch.Tensor
        Scalar conditional CORAL loss.
    """
    d = source_features.size(1)
    
    # Target pseudo-labels with confidence thresholding
    with torch.no_grad():
        probs = F.softmax(target_logits, dim=1)
        max_probs, pseudo_labels = probs.max(dim=1)
        confident_mask = max_probs >= confidence_threshold
    
    classes = source_labels.unique()
    total_loss = torch.tensor(0.0, device=source_features.device)
    n_aligned = 0
    
    for c in classes:
        # Source samples of class c
        s_mask = source_labels == c
        s_feats = source_features[s_mask]
        
        # Target samples pseudo-labeled as c AND confident
        t_mask = (pseudo_labels == c) & confident_mask
        t_feats = target_features[t_mask]
        
        # Need at least 2 samples per class per domain for covariance
        if s_feats.size(0) < 2 or t_feats.size(0) < 2:
            continue
        
        # Per-class covariance matrices
        s_centered = s_feats - s_feats.mean(dim=0, keepdim=True)
        t_centered = t_feats - t_feats.mean(dim=0, keepdim=True)
        
        cov_s = (s_centered.t() @ s_centered) / max(s_feats.size(0) - 1, 1)
        cov_t = (t_centered.t() @ t_centered) / max(t_feats.size(0) - 1, 1)
        
        total_loss = total_loss + (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)
        n_aligned += 1
    
    if n_aligned > 0:
        total_loss = total_loss / n_aligned
    
    return total_loss


# =============================================================================
# Feature Whitening Layer
# =============================================================================
class FeatureWhitening(nn.Module):
    """Learnable feature whitening layer.
    
    Applies running ZCA-style whitening: f_out = W @ (f - mu)
    where W approximates C^{-1/2}.
    
    Uses running statistics (like BatchNorm) so it can whiten at test time.
    The whitening matrix is recomputed periodically from running covariance.
    
    For efficiency, uses BatchNorm-without-affine as the core transform
    (normalizes each dimension independently), then adds a learnable
    linear decorrelation layer to handle cross-feature correlations
    (the subspace rotation that drift diagnostics identified).
    
    Parameters
    ----------
    num_features : int
        Feature dimension.
    momentum : float
        Momentum for running stats update. Default: 0.1
    """
    
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        
        # BatchNorm without affine: normalizes per-dimension (mean=0, var=1)
        # This handles the diagonal of the covariance (scaling)
        self.bn = nn.BatchNorm1d(num_features, affine=False, momentum=momentum)
        
        # Learnable decorrelation: a linear layer initialized to identity
        # This learns to undo cross-feature correlations (rotation)
        self.decorrelate = nn.Linear(num_features, num_features, bias=False)
        nn.init.eye_(self.decorrelate.weight)
    
    def forward(self, x):
        # Step 1: Per-dimension normalization (handles scaling drift)
        x = self.bn(x)
        # Step 2: Learnable decorrelation (handles rotation drift)
        x = self.decorrelate(x)
        return x


# =============================================================================
# Adaptive Batch Normalization (AdaBN)
# =============================================================================
def adapt_batchnorm(model, target_loader, device, alpha=1.0):
    """Re-estimate BatchNorm running statistics on target domain data.
    
    Instead of a hard reset, blends source and target statistics:
        new_stat = alpha * target_stat + (1 - alpha) * source_stat
    
    When alpha=1.0 (default), this is pure target stats (standard AdaBN).
    When alpha<1.0, source stats are partially retained, which is more
    robust when the target dataset is small.
    
    Parameters
    ----------
    model : nn.Module
        Model with BatchNorm layers (trained on source domain).
    target_loader : DataLoader
        DataLoader yielding target domain batches (unlabeled).
    device : torch.device
        Device to run on.
    alpha : float
        Blending ratio in [0, 1]. 1.0 = pure target stats, 0.0 = keep source.
        Default: 1.0
    
    Reference
    ---------
    Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation", ICLR 2017
    """
    # Save source BN stats before overwriting
    source_stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            source_stats[name] = {
                'mean': m.running_mean.clone(),
                'var': m.running_var.clone(),
            }
    
    # Reset BN running stats for fresh target accumulation
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
    
    # Forward pass in train mode to accumulate target statistics
    model.train()
    with torch.no_grad():
        for batch in target_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            model(x)
    
    # Blend source and target stats
    if alpha < 1.0:
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and name in source_stats:
                src = source_stats[name]
                m.running_mean.copy_(alpha * m.running_mean + (1.0 - alpha) * src['mean'])
                m.running_var.copy_(alpha * m.running_var + (1.0 - alpha) * src['var'])
    
    model.eval()


# =============================================================================
# Test-Time Adaptation (Entropy Minimization)
# =============================================================================
def tta_entropy_minimization(model, target_loader, device, tta_steps=1, tta_lr=1e-4):
    """Test-time adaptation via entropy minimization on BN parameters.
    
    For each batch of unlabeled target data:
    1. Forward pass to get predictions
    2. Compute entropy of predictions
    3. Backprop and update BN affine parameters (gamma, beta) only
    
    This pushes the model toward confident predictions on target data
    without requiring any labels.
    
    Parameters
    ----------
    model : nn.Module
        Model with BatchNorm layers.
    target_loader : DataLoader
        DataLoader yielding target domain batches (unlabeled).
    device : torch.device
        Device to run on.
    tta_steps : int
        Number of optimization steps per batch. Default: 1
    tta_lr : float
        Learning rate for TTA updates. Default: 1e-4
    
    Reference
    ---------
    Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021
    """
    # Collect only BN affine parameters (weight=gamma, bias=beta)
    bn_params = []
    bn_modules = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_modules.append(m)
            if m.affine:
                bn_params.append(m.weight)
                bn_params.append(m.bias)
    
    if not bn_params:
        return
    
    optimizer = torch.optim.Adam(bn_params, lr=tta_lr)
    
    # Disable running stat tracking (proper Tent behavior):
    # BN will use batch stats for normalization but NOT update running_mean/running_var.
    # This prevents drift instability from mixing AdaBN stats with TTA batch stats.
    saved_tracking = {}
    for i, m in enumerate(bn_modules):
        saved_tracking[i] = m.track_running_stats
        m.track_running_stats = False
    
    # Set model to train mode so BN uses batch stats, but freeze all non-BN params
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in bn_params:
        p.requires_grad_(True)
    
    for batch in target_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        
        for _ in range(tta_steps):
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1).mean()
            
            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()
    
    # Restore tracking and requires_grad
    for i, m in enumerate(bn_modules):
        m.track_running_stats = saved_tracking[i]
    for p in model.parameters():
        p.requires_grad_(True)
    
    model.eval()


# =============================================================================
# Adaptive Model (FeatureExtractor + LabelClassifier with AdaBN/CORAL/TTA)
# =============================================================================
class AdaptiveModel(nn.Module):
    """Domain adaptation model using AdaBN, Deep CORAL, TTA, and Feature Whitening.
    
    Architecture:
        Input -> FeatureExtractor (with BatchNorm) -> [FeatureWhitening] -> LabelClassifier
    
    Adaptation is achieved through:
    1. Class-Conditional CORAL loss during training (per-class covariance alignment)
    2. Feature Whitening layer (standardizes feature space, attacks subspace rotation)
    3. AdaBN at adaptation time (re-estimates BN stats on target data)
    4. TTA at test time (entropy minimization on BN affine params)
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    feature_dims : list of int
        Hidden dimensions for feature extractor.
    feature_output_dim : int
        Output dimension of feature extractor.
    label_hidden_dims : list of int
        Hidden dimensions for label classifier.
    num_classes : int
        Number of activity classes.
    dropout : float
        Dropout probability. Default: 0.3
    use_batch_norm : bool
        Whether to use BatchNorm in the feature extractor. Default: True.
    use_whitening : bool
        Whether to insert a FeatureWhitening layer after the feature extractor.
        Default: False (backward compatible).
    
    Example
    -------
    >>> model = AdaptiveModel(
    ...     input_dim=1024,
    ...     feature_dims=[512, 256],
    ...     feature_output_dim=128,
    ...     label_hidden_dims=[64],
    ...     num_classes=6,
    ...     use_whitening=True,
    ... )
    >>> logits = model(x)
    >>> features = model.extract_features(x)
    """
    
    def __init__(
        self,
        input_dim,
        feature_dims,
        feature_output_dim,
        label_hidden_dims,
        num_classes,
        dropout=0.3,
        use_batch_norm=True,
        use_whitening=False,
    ):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=feature_dims,
            output_dim=feature_output_dim,
            dropout=dropout,
            batch_norm=use_batch_norm,
        )
        
        self.use_whitening = use_whitening
        if use_whitening:
            self.whitening = FeatureWhitening(feature_output_dim)
        
        self.label_classifier = LabelClassifier(
            input_dim=feature_output_dim,
            hidden_dims=label_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """Forward pass: extract features, [whiten], then classify.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        
        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        features = self.feature_extractor(x)
        if self.use_whitening:
            features = self.whitening(features)
        return self.label_classifier(features)
    
    def extract_features(self, x):
        """Extract features (after whitening if enabled).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        
        Returns
        -------
        torch.Tensor
            Features of shape (batch_size, feature_output_dim).
        """
        features = self.feature_extractor(x)
        if self.use_whitening:
            features = self.whitening(features)
        return features
    
    def predict(self, x):
        """Predict labels (for inference)."""
        return self.forward(x)


# =============================================================================
# Utility: Create adaptive models for experiments
# =============================================================================
def make_adaptive_model(n_features, n_classes, config='default',
                        use_batch_norm=True, use_whitening=False):
    """Factory function to create AdaptiveModel with preset configurations.
    
    Parameters
    ----------
    n_features : int
        Number of input features.
    n_classes : int
        Number of activity classes.
    config : str
        Configuration preset: 'default', 'small', 'large'.
    use_batch_norm : bool
        Whether to use BatchNorm in the feature extractor. Default: True.
    use_whitening : bool
        Whether to enable the FeatureWhitening layer. Default: False.
    
    Returns
    -------
    AdaptiveModel
    """
    configs = {
        'default': {
            'feature_dims': [512, 256],
            'feature_output_dim': 128,
            'label_hidden_dims': [64],
            'dropout': 0.3,
        },
        'small': {
            'feature_dims': [256, 128],
            'feature_output_dim': 64,
            'label_hidden_dims': [32],
            'dropout': 0.2,
        },
        'large': {
            'feature_dims': [1024, 512, 256],
            'feature_output_dim': 256,
            'label_hidden_dims': [128, 64],
            'dropout': 0.4,
        },
    }
    
    cfg = configs.get(config, configs['default'])
    
    return AdaptiveModel(
        input_dim=n_features,
        feature_dims=cfg['feature_dims'],
        feature_output_dim=cfg['feature_output_dim'],
        label_hidden_dims=cfg['label_hidden_dims'],
        num_classes=n_classes,
        dropout=cfg['dropout'],
        use_batch_norm=use_batch_norm,
        use_whitening=use_whitening,
    )


# =============================================================================
# Training (returns trained model + training info)
# =============================================================================
def train_model(model, X_source, y_source, X_target, X_test, y_test,
                epochs=50, batch_size=64, lr=1e-3, coral_weight=0.5,
                use_coral=False, use_conditional_coral=False,
                confidence_threshold=0.8, verbose=True):
    """Train model on source data with optional CORAL alignment.

    Returns the trained model (in eval mode) and a training info dict.
    Does NOT apply any post-training adaptation (AdaBN/TTA/fewshot).

    Parameters
    ----------
    model : AdaptiveModel
    X_source, y_source : np.ndarray
        Source domain training data.
    X_target : np.ndarray
        Target domain features (unlabeled, used for CORAL).
    X_test, y_test : np.ndarray
        Test data for periodic evaluation during training.
    epochs, batch_size, lr : training hyperparameters
    coral_weight : float
        Weight for CORAL loss term.
    use_coral : bool
        Use global Deep CORAL loss.
    use_conditional_coral : bool
        Use class-conditional CORAL (overrides use_coral).
    confidence_threshold : float
        Pseudo-label confidence for conditional CORAL.
    verbose : bool

    Returns
    -------
    model : AdaptiveModel (trained, eval mode, on device)
    info : dict with train_time_s, train_accuracy, coral_mode
    """
    import time
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    source_ds = TensorDataset(torch.FloatTensor(X_source), torch.LongTensor(y_source))
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Class-weighted loss for imbalanced data
    import numpy as _np
    _classes, _counts = _np.unique(y_source, return_counts=True)
    _weights = 1.0 / _counts.astype(_np.float64)
    _weights = _weights / _weights.sum() * len(_classes)  # normalize so mean=1
    _weight_tensor = torch.zeros(int(_classes.max()) + 1, device=device)
    for _c, _w in zip(_classes, _weights):
        _weight_tensor[int(_c)] = _w
    criterion = nn.CrossEntropyLoss(weight=_weight_tensor.float())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    any_coral = use_coral or use_conditional_coral
    coral_mode = 'conditional' if use_conditional_coral else ('global' if use_coral else 'none')

    if verbose:
        print(f"  Device: {device}, Epochs: {epochs}, LR: {lr}")
        print(f"  CORAL: {coral_mode} (weight={coral_weight})")
        if use_conditional_coral:
            print(f"  Conditional CORAL confidence threshold: {confidence_threshold}")
        if hasattr(model, 'use_whitening') and model.use_whitening:
            print(f"  Feature Whitening: enabled")

    X_target_tensor = torch.FloatTensor(X_target).to(device) if any_coral else None
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    t0 = time.process_time()
    report_interval = max(1, epochs // 10)

    for epoch in range(epochs):
        model.train()
        total_task_loss = 0.0
        total_coral_loss = 0.0
        correct = 0
        total = 0

        if any_coral:
            with torch.no_grad():
                target_feats_all = model.extract_features(X_target_tensor)
                if use_conditional_coral:
                    target_logits_all = model.label_classifier(target_feats_all)
            target_feats_ref = target_feats_all.detach()
            if use_conditional_coral:
                target_logits_ref = target_logits_all.detach()

        for xb, yb in source_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            source_features = model.extract_features(xb)
            logits = model.label_classifier(source_features)
            task_loss = criterion(logits, yb)
            loss = task_loss

            if use_conditional_coral:
                c_loss = conditional_coral_loss(
                    source_features, target_feats_ref, yb,
                    target_logits_ref, confidence_threshold=confidence_threshold)
                loss = loss + coral_weight * c_loss
                total_coral_loss += c_loss.item() * xb.size(0)
            elif use_coral:
                c_loss = coral_loss(source_features, target_feats_ref)
                loss = loss + coral_weight * c_loss
                total_coral_loss += c_loss.item() * xb.size(0)

            loss.backward()
            optimizer.step()

            total_task_loss += task_loss.item() * xb.size(0)
            _, pred = torch.max(logits, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()

        train_acc = correct / total
        train_loss = total_task_loss / total

        if verbose and ((epoch + 1) % report_interval == 0 or epoch == 0):
            model.eval()
            with torch.no_grad():
                test_logits = model.predict(X_test_tensor)
                test_loss_val = criterion(test_logits, y_test_tensor).item()
                _, test_preds = torch.max(test_logits, 1)
                test_correct = (test_preds == y_test_tensor).sum().item()
                test_acc = test_correct / len(y_test)
            model.train()

            msg = (f"  Epoch {epoch+1:3d}/{epochs} | "
                   f"TrainLoss: {train_loss:.4f}  TrainAcc: {train_acc:.4f} | "
                   f"TestLoss: {test_loss_val:.4f}  TestAcc: {test_acc:.4f}")
            if any_coral:
                msg += f" | CORAL: {total_coral_loss/total:.6f}"
            print(msg)

    train_time = round(time.process_time() - t0, 2)
    model.eval()

    if verbose:
        print(f"  Training complete in {train_time}s, final train acc: {train_acc:.4f}")

    info = {
        'train_time_s': train_time,
        'train_accuracy': round(train_acc, 4),
        'coral_mode': coral_mode,
    }
    return model, info


# =============================================================================
# Comprehensive metrics computation
# =============================================================================
def compute_metrics(model_or_logits, X_test, y_test, device=None):
    """Compute comprehensive metrics from a model or pre-computed logits.

    Parameters
    ----------
    model_or_logits : AdaptiveModel or torch.Tensor
        If a model, runs forward pass on X_test. If tensor, uses directly.
    X_test : np.ndarray
        Test features (ignored if model_or_logits is a tensor).
    y_test : np.ndarray
        Test labels.
    device : torch.device or None

    Returns
    -------
    dict with all metrics.
    """
    import numpy as np
    from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                                 precision_score, recall_score,
                                 cohen_kappa_score, matthews_corrcoef,
                                 balanced_accuracy_score, log_loss)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(model_or_logits, nn.Module):
        model_or_logits.eval()
        with torch.no_grad():
            logits_tensor = model_or_logits.predict(torch.FloatTensor(X_test).to(device))
    else:
        logits_tensor = model_or_logits

    probs = F.softmax(logits_tensor, dim=1).cpu().numpy()
    preds = logits_tensor.argmax(dim=1).cpu().numpy()
    n = len(y_test)
    n_cls = logits_tensor.size(1)

    acc = round(accuracy_score(y_test, preds), 4)
    bal_acc = round(balanced_accuracy_score(y_test, preds), 4)

    f1_w = round(f1_score(y_test, preds, average='weighted', zero_division=0), 4)
    f1_macro = round(f1_score(y_test, preds, average='macro', zero_division=0), 4)
    f1_micro = round(f1_score(y_test, preds, average='micro', zero_division=0), 4)

    prec_w = round(precision_score(y_test, preds, average='weighted', zero_division=0), 4)
    rec_w = round(recall_score(y_test, preds, average='weighted', zero_division=0), 4)
    prec_macro = round(precision_score(y_test, preds, average='macro', zero_division=0), 4)
    rec_macro = round(recall_score(y_test, preds, average='macro', zero_division=0), 4)

    prec_per = np.round(precision_score(y_test, preds, average=None, zero_division=0, labels=list(range(n_cls))), 4).tolist()
    rec_per = np.round(recall_score(y_test, preds, average=None, zero_division=0, labels=list(range(n_cls))), 4).tolist()
    f1_per = np.round(f1_score(y_test, preds, average=None, zero_division=0, labels=list(range(n_cls))), 4).tolist()

    cm = confusion_matrix(y_test, preds, labels=list(range(n_cls)))
    per_class_acc = []
    for c in range(n_cls):
        total_c = cm[c].sum()
        per_class_acc.append(round(cm[c, c] / total_c, 4) if total_c > 0 else 0.0)

    kappa = round(cohen_kappa_score(y_test, preds), 4)
    mcc = round(matthews_corrcoef(y_test, preds), 4)

    try:
        ll = round(log_loss(y_test, probs, labels=list(range(n_cls))), 4)
    except Exception:
        ll = float('nan')

    max_probs = probs.max(axis=1)
    mean_conf = round(float(np.mean(max_probs)), 4)
    std_conf = round(float(np.std(max_probs)), 4)
    median_conf = round(float(np.median(max_probs)), 4)

    ent = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)
    mean_ent = round(float(np.mean(ent)), 4)
    std_ent = round(float(np.std(ent)), 4)

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (max_probs > lo) & (max_probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = (preds[mask] == y_test[mask]).mean()
        bin_conf = max_probs[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    ece = round(float(ece), 4)

    return {
        'accuracy': acc, 'balanced_accuracy': bal_acc,
        'f1_weighted': f1_w, 'f1_macro': f1_macro, 'f1_micro': f1_micro,
        'precision_weighted': prec_w, 'recall_weighted': rec_w,
        'precision_macro': prec_macro, 'recall_macro': rec_macro,
        'precision_per_class': prec_per, 'recall_per_class': rec_per,
        'f1_per_class': f1_per, 'accuracy_per_class': per_class_acc,
        'cohen_kappa': kappa, 'mcc': mcc, 'log_loss': ll,
        'mean_confidence': mean_conf, 'std_confidence': std_conf,
        'median_confidence': median_conf,
        'mean_entropy': mean_ent, 'std_entropy': std_ent, 'ece': ece,
        'confusion_matrix': cm.tolist(),
    }


# =============================================================================
# Adapt and evaluate (post-training adaptation on a copy of trained model)
# =============================================================================
def adapt_and_evaluate(trained_model, X_target, X_test, y_test,
                       train_info, adapt_name='none',
                       use_adabn=False, adabn_alpha=1.0,
                       use_tta=False, tta_steps=1, tta_lr=1e-4,
                       X_target_labeled=None, y_target_labeled=None,
                       fewshot_epochs=20, fewshot_lr=1e-4,
                       batch_size=64, verbose=True):
    """Apply post-training adaptation to a copy of trained_model and evaluate.

    Parameters
    ----------
    trained_model : AdaptiveModel (trained, eval mode)
        Will be deepcopied â€” original is never modified.
    X_target : np.ndarray
        Target domain features (unlabeled).
    X_test, y_test : np.ndarray
        Test data for evaluation.
    train_info : dict
        Output from train_model (train_time_s, train_accuracy, coral_mode).
    adapt_name : str
        Label for this adaptation variant.
    use_adabn, adabn_alpha, use_tta, tta_steps, tta_lr : AdaBN/TTA params
    X_target_labeled, y_target_labeled : few-shot labeled data
    fewshot_epochs, fewshot_lr : few-shot fine-tuning params
    batch_size : int
    verbose : bool

    Returns
    -------
    dict with all metrics + adaptation info.
    """
    import copy
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pre-adaptation metrics (from the original trained model)
    pre = compute_metrics(trained_model, X_test, y_test, device)

    if verbose:
        print(f"    [{adapt_name}] Pre-adapt  -> Acc: {pre['accuracy']}, F1w: {pre['f1_weighted']}")

    # Deep copy so we don't mutate the shared trained model
    adapt_model = copy.deepcopy(trained_model).to(device)

    # AdaBN
    if use_adabn:
        if verbose:
            print(f"    [{adapt_name}] Applying AdaBN (alpha={adabn_alpha})...")
        adapt_target_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_target)),
            batch_size=batch_size, shuffle=False)
        adapt_batchnorm(adapt_model, adapt_target_loader, device, alpha=adabn_alpha)

    # TTA
    if use_tta:
        if verbose:
            print(f"    [{adapt_name}] Applying TTA (steps={tta_steps}, lr={tta_lr})...")
        tta_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_target)),
            batch_size=batch_size, shuffle=True)
        tta_entropy_minimization(adapt_model, tta_loader, device,
                                 tta_steps=tta_steps, tta_lr=tta_lr)

    # Few-shot fine-tuning
    has_fewshot = X_target_labeled is not None and y_target_labeled is not None and len(X_target_labeled) > 0
    if has_fewshot:
        if verbose:
            print(f"    [{adapt_name}] Few-shot fine-tuning: {len(X_target_labeled)} samples, "
                  f"{fewshot_epochs} epochs, lr={fewshot_lr}")
        adapt_model.train()
        fs_ds = TensorDataset(torch.FloatTensor(X_target_labeled), torch.LongTensor(y_target_labeled))
        fs_loader = DataLoader(fs_ds, batch_size=min(batch_size, len(X_target_labeled)),
                               shuffle=True, drop_last=False)
        fs_optimizer = torch.optim.Adam(adapt_model.parameters(), lr=fewshot_lr)
        fs_criterion = nn.CrossEntropyLoss()
        for _ in range(fewshot_epochs):
            for fs_xb, fs_yb in fs_loader:
                fs_xb, fs_yb = fs_xb.to(device), fs_yb.to(device)
                fs_optimizer.zero_grad()
                fs_loss = fs_criterion(adapt_model(fs_xb), fs_yb)
                fs_loss.backward()
                fs_optimizer.step()
        adapt_model.eval()

    # Post-adaptation metrics
    post = compute_metrics(adapt_model, X_test, y_test, device)

    if verbose:
        delta = round(post['accuracy'] - pre['accuracy'], 4)
        sign = '+' if delta >= 0 else ''
        print(f"    [{adapt_name}] Post-adapt -> Acc: {post['accuracy']}, F1w: {post['f1_weighted']}, "
              f"Kappa: {post['cohen_kappa']}, ECE: {post['ece']}  (delta: {sign}{delta})")

    return {
        'train_time_s': train_info['train_time_s'],
        'train_accuracy': train_info['train_accuracy'],
        'pre_adapt_accuracy': pre['accuracy'],
        'pre_adapt_f1': pre['f1_weighted'],
        'test_accuracy': post['accuracy'],
        'test_f1': post['f1_weighted'],
        'post': post,
        'adaptation': {
            'coral': train_info['coral_mode'],
            'adabn': use_adabn,
            'tta': use_tta,
            'whitening': hasattr(trained_model, 'use_whitening') and trained_model.use_whitening,
            'fewshot': has_fewshot,
            'fewshot_n': len(X_target_labeled) if has_fewshot else 0,
        },
    }


def _print_dl_metrics(name, m, label_map=None):
    p = m.get('post', {})
    n_cls = len(p.get('accuracy_per_class', []))
    cls_names = [label_map.get(i, str(i)) for i in range(n_cls)] if label_map else [str(i) for i in range(n_cls)]

    print(f"\n  {'='*60}")
    print(f"  {name}")
    print(f"  {'='*60}")

    # Adaptation config
    if 'adaptation' in m:
        a = m['adaptation']
        parts = [f"CORAL={a['coral']}"]
        if a['adabn']: parts.append('AdaBN')
        if a['tta']: parts.append('TTA')
        if a.get('whitening'): parts.append('Whitening')
        if a.get('fewshot'): parts.append(f"FewShot({a['fewshot_n']})")
        print(f"    Config:         {', '.join(parts)}")

    print(f"    Train time:     {m['train_time_s']}s")
    print(f"    Train accuracy: {m['train_accuracy']}")

    # Pre-adaptation
    if 'pre_adapt_accuracy' in m:
        print(f"    Pre-adapt acc:  {m['pre_adapt_accuracy']}")
        print(f"    Pre-adapt F1:   {m['pre_adapt_f1']}")
        delta = round(m['test_accuracy'] - m['pre_adapt_accuracy'], 4)
        sign = '+' if delta >= 0 else ''
        print(f"    Adapt delta:    {sign}{delta}")

    # Global metrics
    print(f"    --- Global Metrics ---")
    print(f"    Accuracy:       {p.get('accuracy', m['test_accuracy'])}")
    print(f"    Balanced Acc:   {p.get('balanced_accuracy', 'N/A')}")
    print(f"    F1 weighted:    {p.get('f1_weighted', m['test_f1'])}")
    print(f"    F1 macro:       {p.get('f1_macro', 'N/A')}")
    print(f"    F1 micro:       {p.get('f1_micro', 'N/A')}")
    print(f"    Prec weighted:  {p.get('precision_weighted', 'N/A')}")
    print(f"    Rec  weighted:  {p.get('recall_weighted', 'N/A')}")
    print(f"    Prec macro:     {p.get('precision_macro', 'N/A')}")
    print(f"    Rec  macro:     {p.get('recall_macro', 'N/A')}")
    print(f"    Cohen Kappa:    {p.get('cohen_kappa', 'N/A')}")
    print(f"    MCC:            {p.get('mcc', 'N/A')}")
    print(f"    Log Loss:       {p.get('log_loss', 'N/A')}")

    # Calibration & confidence
    print(f"    --- Calibration & Confidence ---")
    print(f"    ECE:            {p.get('ece', 'N/A')}")
    print(f"    Mean conf:      {p.get('mean_confidence', 'N/A')}")
    print(f"    Std  conf:      {p.get('std_confidence', 'N/A')}")
    print(f"    Median conf:    {p.get('median_confidence', 'N/A')}")
    print(f"    Mean entropy:   {p.get('mean_entropy', 'N/A')}")
    print(f"    Std  entropy:   {p.get('std_entropy', 'N/A')}")

    # Per-class table
    if n_cls > 0:
        print(f"    --- Per-Class Breakdown ---")
        print(f"    {'Class':<10} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print(f"    {'-'*36}")
        acc_pc = p.get('accuracy_per_class', [])
        prec_pc = p.get('precision_per_class', [])
        rec_pc = p.get('recall_per_class', [])
        f1_pc = p.get('f1_per_class', [])
        for i in range(n_cls):
            cname = cls_names[i] if i < len(cls_names) else str(i)
            print(f"    {cname:<10} {acc_pc[i]:>6.4f} {prec_pc[i]:>6.4f} {rec_pc[i]:>6.4f} {f1_pc[i]:>6.4f}")

    # Confusion matrix
    cm = p.get('confusion_matrix', m.get('confusion_matrix', []))
    if cm:
        print(f"    --- Confusion Matrix ---")
        hdr = '    ' + ' ' * 10 + '  '.join(f'{c:>6}' for c in cls_names)
        print(hdr)
        for i, row in enumerate(cm):
            rname = cls_names[i] if i < len(cls_names) else str(i)
            print(f"    {rname:<10}" + '  '.join(f'{v:>6}' for v in row))


# =============================================================================
# CNN+LSTM Classifier (wraps SeqEncoder for the DL experiment)
# =============================================================================
class CnnLstmClassifier(nn.Module):
    """CNN+LSTM classifier for sequential CSI data.

    Reshapes flat input (B, T*C) -> (B, T, C), applies Conv1D blocks then
    Bi-LSTM, pools, and classifies.  Uses the SeqEncoder from seq.py internally.

    Parameters
    ----------
    n_subcarriers : int
        Number of CSI subcarriers (channels). Default 52.
    window_len : int
        Temporal length of one window. Default 100.
    num_classes : int
        Number of output classes.
    conv_channels : list
        Channel sizes for Conv1D blocks.
    lstm_hidden : int
        LSTM hidden size per direction.
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability.
    use_batch_norm : bool
        Accepted for API compat (Conv1D blocks always use BN). Default True.
    use_whitening : bool
        Whether to use FeatureWhitening after pooling. Default False.
    """

    def __init__(self, n_subcarriers=52, window_len=100, num_classes=7,
                 conv_channels=None, lstm_hidden=64, lstm_layers=1,
                 dropout=0.2, use_batch_norm=True, use_whitening=False):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]
        self.n_subcarriers = n_subcarriers
        self.window_len = window_len
        self.num_classes = num_classes

        # Conv1D blocks
        layers = []
        in_ch = n_subcarriers
        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.convs = nn.Sequential(*layers)

        # Compute temporal length after conv pooling
        t = window_len
        for _ in conv_channels:
            t = t // 2
        self._conv_time = t
        self._conv_out_ch = conv_channels[-1]

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=self._conv_out_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        repr_dim = lstm_hidden * 2
        self.pool_norm = nn.BatchNorm1d(repr_dim) if use_batch_norm else nn.Identity()

        self.use_whitening = use_whitening
        if use_whitening:
            self.whitening = FeatureWhitening(repr_dim)

        # Classification head
        self.label_classifier = nn.Sequential(
            nn.Linear(repr_dim, repr_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(repr_dim // 2, num_classes),
        )

        self._repr_dim = repr_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x):
        B = x.size(0)
        x = x.view(B, self.window_len, self.n_subcarriers).permute(0, 2, 1)
        x = self.convs(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        z = x.mean(dim=1)
        z = self.pool_norm(z)
        if self.use_whitening:
            z = self.whitening(z)
        return z

    def forward(self, x):
        z = self.extract_features(x)
        return self.label_classifier(z)

    def predict(self, x, batch_size=256):
        if x.size(0) <= batch_size:
            return self.forward(x)
        parts = [self.forward(x[i:i+batch_size]) for i in range(0, x.size(0), batch_size)]
        return torch.cat(parts, 0)


def make_cnn_lstm_model(n_subcarriers, window_len, n_classes, config='small',
                        use_batch_norm=True, use_whitening=False):
    """Factory for CnnLstmClassifier with preset configs."""
    configs = {
        'small': dict(conv_channels=[32, 64], lstm_hidden=64, lstm_layers=1, dropout=0.2),
        'mid':   dict(conv_channels=[64, 128], lstm_hidden=128, lstm_layers=2, dropout=0.25),
        'large': dict(conv_channels=[64, 128, 256], lstm_hidden=256, lstm_layers=2, dropout=0.3),
    }
    cfg = configs.get(config, configs['small'])
    return CnnLstmClassifier(
        n_subcarriers=n_subcarriers, window_len=window_len,
        num_classes=n_classes, use_batch_norm=use_batch_norm,
        use_whitening=use_whitening, **cfg,
    )


# =============================================================================
# MLP factory with small/mid/large configs
# =============================================================================
def make_mlp_model(n_features, n_classes, config='small',
                   use_batch_norm=True, use_whitening=False):
    """Factory for MLP with preset configs. Input is flattened."""
    configs = {
        'small': dict(hidden_dims=[256, 128], dropout=0.2),
        'mid':   dict(hidden_dims=[512, 256, 128], dropout=0.3),
        'large': dict(hidden_dims=[1024, 512, 256, 128], dropout=0.4),
    }
    cfg = configs.get(config, configs['small'])
    return AdaptiveModel(
        input_dim=n_features,
        feature_dims=cfg['hidden_dims'][:-1],
        feature_output_dim=cfg['hidden_dims'][-1],
        label_hidden_dims=[cfg['hidden_dims'][-1] // 2],
        num_classes=n_classes,
        dropout=cfg['dropout'],
        use_batch_norm=use_batch_norm,
        use_whitening=use_whitening,
    )


# =============================================================================
# DL Experiment: 5 conditions x 4 architectures x 4 datasets x 2 pipelines
# =============================================================================
def fewshot_finetune(model, X_fs, y_fs, epochs=20, lr=1e-5, batch_size=32):
    """Few-shot fine-tuning: freeze feature extractor, train classifier only.

    Parameters
    ----------
    model : nn.Module
        Must have extract_features() and label_classifier attributes.
    X_fs, y_fs : np.ndarray
        Few-shot labeled data (from beginning of test set).
    epochs : int
    lr : float
        Reduced learning rate for fine-tuning.
    batch_size : int
    """
    import copy
    from torch.utils.data import DataLoader, TensorDataset

    device = next(model.parameters()).device
    model = copy.deepcopy(model).to(device)

    # Freeze everything except the classifier head
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.label_classifier.parameters():
        p.requires_grad_(True)

    fs_ds = TensorDataset(torch.FloatTensor(X_fs), torch.LongTensor(y_fs))
    fs_loader = DataLoader(fs_ds, batch_size=min(batch_size, len(X_fs)),
                           shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in fs_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Unfreeze all for subsequent use
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return model


def run_dl_experiment(data_root, window_lens=None, guaranteed_sr=150,
                      epochs=50, lr=1e-3, model_size='small', verbose=True,
                      n_seeds=1, cv_mode=False, n_folds=None,
                      var_windows=None):
    """Run DL experiment: 3 architectures x 4 datasets x N window lengths x N variance windows.

    Each configuration is run ``n_seeds`` times with different random seeds.

    When ``cv_mode=True``, temporal forward-chaining cross-validation is
    used instead of the fixed metadata train/test split.  Metrics are
    collected per fold x seed, then aggregated for final mean +/- std.

    Parameters
    ----------
    data_root : str
        Root folder containing the 4 dataset subfolders.
    window_lens : list of int or None
        Window lengths to compare. Default [500, 1000, 2000].
    guaranteed_sr : int
        Resampling rate. Default 150.
    epochs : int
        Training epochs. Default 50.
    lr : float
        Learning rate. Default 1e-3.
    model_size : str
        'small', 'mid', or 'large'. Default 'small'.
    verbose : bool
    n_seeds : int
        Number of random seeds for multi-seed runs. Default 1.
    cv_mode : bool
        Use temporal forward-chaining cross-validation. Default False.
    n_folds : int or None
        Number of CV folds (None = auto). Only used when cv_mode=True.
    var_windows : list of int or None
        Rolling variance window sizes to compare. Default [20, 200, 2000].

    Returns
    -------
    dict : nested results with per-seed and aggregated metrics
    """
    import sys, os, time, copy, traceback
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import (load_all_datasets, load_all_datasets_cv,
                       set_global_seed,
                       compute_all_metrics as _compute_all_metrics,
                       aggregate_seed_metrics, print_metrics_summary,
                       METRICS_CSV_FIELDS)

    N_SUBCARRIERS = 52
    SEEDS = list(range(42, 42 + n_seeds))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Single condition: no domain adaptation
    CONDITIONS = ['noBN']

    # Model architectures
    ARCHITECTURES = ['MLP', 'Conv1D', 'CNN_LSTM']

    # Window lengths
    if window_lens is None:
        window_lens = [500, 1000, 2000]

    # Pipeline: rolling_variance with configurable window sizes
    if var_windows is None:
        var_windows = [20, 200, 2000]
    PIPELINES = [
        (f'rv_w{w}', dict(pipeline_name='rolling_variance', var_window=w))
        for w in var_windows
    ]

    # all_results[run_key] = {'seeds': [metrics_per_seed...], 'agg': aggregated}
    all_results = {}

    # ---- CSV setup (incremental saving) ----
    import csv
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dl_results')
    os.makedirs(results_dir, exist_ok=True)
    csv_tag = '_cv' if cv_mode else ''
    csv_path = os.path.join(results_dir, f'dl_results_per_seed{csv_tag}.csv')
    agg_csv_path = os.path.join(results_dir, f'dl_results_aggregated{csv_tag}.csv')
    seed_fieldnames = (['dataset', 'window_len', 'pipeline', 'architecture', 'condition',
                        'fold', 'seed']
                       + METRICS_CSV_FIELDS
                       + ['train_time_s', 'infer_time_s', 'data_load_time_s'])
    agg_fields = ['dataset', 'window_len', 'pipeline', 'architecture', 'condition', 'n_seeds']
    for _k in METRICS_CSV_FIELDS:
        agg_fields.extend([f'{_k}_mean', f'{_k}_std'])

    def _save_dl_results():
        """Rewrite both CSVs with all results collected so far."""
        with open(csv_path, 'w', newline='') as _f:
            w = csv.DictWriter(_f, fieldnames=seed_fieldnames, extrasaction='ignore')
            w.writeheader()
            for _key in sorted(all_results.keys()):
                _entry = all_results[_key]
                if _entry['agg'] is None:
                    continue
                for _sm in _entry['seeds']:
                    _row = {
                        'dataset': _entry['agg']['dataset'],
                        'window_len': _entry['agg']['window_len'],
                        'pipeline': _entry['agg']['pipeline'],
                        'architecture': _entry['agg']['architecture'],
                        'condition': _entry['agg']['condition'],
                    }
                    _row.update(_sm)
                    w.writerow(_row)
        with open(agg_csv_path, 'w', newline='') as _f:
            w = csv.DictWriter(_f, fieldnames=agg_fields, extrasaction='ignore')
            w.writeheader()
            for _key in sorted(all_results.keys()):
                if all_results[_key]['agg'] is not None:
                    w.writerow(all_results[_key]['agg'])
        print(f"  [saved] {os.path.abspath(csv_path)}")

    for window_len in window_lens:
     for pipe_name, pipe_kwargs in PIPELINES:
        wl_pipe = f"wl{window_len}__{pipe_name}"
        print(f"\n{'#'*80}")
        print(f"# WINDOW_LEN={window_len}  PIPELINE: {pipe_name}")
        print(f"{'#'*80}")

        t_data_start = time.perf_counter()

        if cv_mode:
            try:
                datasets_cv = load_all_datasets_cv(
                    data_root, n_folds=n_folds, window_len=window_len,
                    guaranteed_sr=guaranteed_sr, mode='flattened',
                    stride=None, verbose=False, **pipe_kwargs,
                )
            except Exception as e:
                print(f"  ERROR loading CV datasets for pipeline '{pipe_name}': {e}")
                traceback.print_exc()
                continue
            ds_fold_list = []
            for ds_name, folds in datasets_cv.items():
                for fold_idx, train_ds, test_ds in folds:
                    ds_fold_list.append((ds_name, fold_idx, train_ds, test_ds))
        else:
            try:
                datasets = load_all_datasets(
                    data_root, window_len=window_len, guaranteed_sr=guaranteed_sr,
                    mode='flattened', stride=None, verbose=False, **pipe_kwargs,
                )
            except Exception as e:
                print(f"  ERROR loading datasets for pipeline '{pipe_name}': {e}")
                traceback.print_exc()
                continue
            ds_fold_list = []
            for ds_name, (train_ds, test_ds) in datasets.items():
                ds_fold_list.append((ds_name, -1, train_ds, test_ds))

        data_load_time = round(time.perf_counter() - t_data_start, 3)
        print(f"  Data processing time: {data_load_time:.3f}s")

        for ds_name, fold_idx, train_ds, test_ds in ds_fold_list:
            fold_tag = f"fold{fold_idx}" if fold_idx >= 0 else "fixed"
            print(f"\n{'='*70}")
            print(f"  Dataset: {ds_name}  |  WL={window_len}  |  Pipeline: {pipe_name}  |  Split: {fold_tag}")
            print(f"  Train: {train_ds.X.shape}  Test: {test_ds.X.shape}  "
                  f"Classes: {train_ds.num_classes} {train_ds.labels}")
            print(f"{'='*70}")

            if test_ds.X.shape[0] == 0:
                print(f"  SKIP - no test data")
                continue

            n_features = train_ds.X.shape[1]
            n_classes = train_ds.num_classes
            X_tr, y_tr = train_ds.X, train_ds.y
            X_te, y_te = test_ds.X, test_ds.y
            X_target = X_te

            for arch_name in ARCHITECTURES:
              for cond_name in CONDITIONS:
                run_key = f"{ds_name}__{wl_pipe}__{arch_name}__{cond_name}"
                print(f"\n    {'='*60}")
                print(f"    {arch_name} | {cond_name} | {ds_name} | {pipe_name}")
                print(f"    {'='*60}")

                # Determine model kwargs from condition
                use_bn = False
                use_whitening = False

                seed_metrics_list = []

                for seed_idx, seed in enumerate(SEEDS):
                    set_global_seed(seed)
                    print(f"      [seed {seed}] ({seed_idx+1}/{n_seeds})")

                    try:
                        if arch_name == 'MLP':
                            model = make_mlp_model(
                                n_features, n_classes, config=model_size,
                                use_batch_norm=use_bn, use_whitening=use_whitening)
                        elif arch_name == 'Conv1D':
                            model = make_conv1d_model(
                                N_SUBCARRIERS, window_len, n_classes,
                                config=model_size, use_batch_norm=use_bn,
                                use_whitening=use_whitening)
                        elif arch_name == 'CNN_LSTM':
                            model = make_cnn_lstm_model(
                                N_SUBCARRIERS, window_len, n_classes,
                                config=model_size, use_batch_norm=use_bn,
                                use_whitening=use_whitening)
                        else:
                            continue
                    except Exception as e:
                        print(f"      MODEL CREATION ERROR: {e}")
                        traceback.print_exc()
                        continue

                    # Train
                    try:
                        trained_model, info = train_model(
                            model, X_tr, y_tr, X_target, X_te, y_te,
                            epochs=epochs, lr=lr, batch_size=64,
                            coral_weight=0.0,
                            use_conditional_coral=False,
                            confidence_threshold=0.8,
                            verbose=(verbose and seed_idx == 0),
                        )
                    except Exception as e:
                        print(f"      TRAIN ERROR: {e}")
                        traceback.print_exc()
                        continue

                    # Evaluate using shared metrics (with probabilities for ECE)
                    trained_model.eval()
                    t_infer = time.process_time()
                    with torch.no_grad():
                        logits = trained_model.predict(
                            torch.FloatTensor(X_te).to(device))
                    infer_time = time.process_time() - t_infer
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    preds = logits.argmax(dim=1).cpu().numpy()

                    metrics = _compute_all_metrics(
                        y_te, preds, y_prob=probs, n_classes=n_classes)
                    metrics['train_time_s'] = info['train_time_s']
                    metrics['infer_time_s'] = round(infer_time, 4)
                    metrics['data_load_time_s'] = data_load_time
                    metrics['window_len'] = window_len
                    metrics['seed'] = seed
                    metrics['fold'] = fold_idx

                    print(f"        Acc={metrics['accuracy']}  F1w={metrics['f1_weighted']}  "
                          f"Kappa={metrics['cohen_kappa']}  MCC={metrics['mcc']}  "
                          f"ECE={metrics.get('ece','N/A')}")

                    # Accumulate per run_key across folds
                    if run_key not in all_results:
                        all_results[run_key] = {'seeds': [], 'agg': None}
                    all_results[run_key]['seeds'].append(metrics)

        # After all folds for this pipeline, aggregate where not yet done
        for run_key in list(all_results.keys()):
            entry = all_results[run_key]
            if entry['agg'] is not None:
                continue
            if not entry['seeds']:
                continue
            agg = aggregate_seed_metrics(entry['seeds'])
            # run_key: dataset__wl{N}__rv_w{M}__arch__cond
            parts = run_key.split('__')
            agg['dataset'] = parts[0]
            agg['window_len'] = parts[1] if len(parts) > 1 else ''
            agg['pipeline'] = parts[2] if len(parts) > 2 else ''
            agg['architecture'] = parts[3] if len(parts) > 3 else ''
            agg['condition'] = parts[4] if len(parts) > 4 else ''
            entry['agg'] = agg
            print(f"  [{run_key}] "
                  f"Acc={agg.get('accuracy_mean','?')}  "
                  f"F1w={agg.get('f1_weighted_mean','?')}  "
                  f"Kappa={agg.get('cohen_kappa_mean','?')}")
            _save_dl_results()

    # ---- Final comparison table ----
    print(f"\n{'='*200}")
    split_desc = f"CV {n_folds or 'auto'} folds x {n_seeds} seeds" if cv_mode else f"{n_seeds} seed(s)"
    print(f"FINAL DL COMPARISON: {len(window_lens)} WLs x {len(var_windows)} VarWins x 4 Datasets x 3 Archs  ({split_desc})")
    print(f"{'='*200}")
    hdr = (f"{'Dataset':<25} {'WinLen':<10} {'VarWin':<10} {'Arch':<14} | "
           f"{'Acc':>8} {'F1w':>8} {'Kappa':>8} {'DataLoad(s)':>12} {'Train(s)':>10} {'Infer(s)':>10}")
    print(hdr)
    print("-" * 160)
    for key in sorted(all_results.keys()):
        a = all_results[key]['agg']
        s0 = all_results[key]['seeds'][0] if all_results[key]['seeds'] else {}
        acc = a.get('accuracy_mean', float('nan'))
        f1w = a.get('f1_weighted_mean', float('nan'))
        kap = a.get('cohen_kappa_mean', float('nan'))
        dl_t = s0.get('data_load_time_s', '')
        tr_t = s0.get('train_time_s', '')
        inf_t = s0.get('infer_time_s', '')
        print(f"{a['dataset']:<25} {a['window_len']:<10} {a['pipeline']:<10} "
              f"{a['architecture']:<14} | "
              f"{acc:>8.4f} {f1w:>8.4f} {kap:>8.4f} "
              f"{str(dl_t):>12} {str(tr_t):>10} {str(inf_t):>10}")

    # ---- Best per dataset+window_len+pipeline ----
    print(f"\n{'='*100}")
    print("BEST MODEL PER DATASET + WINDOW_LEN + VAR_WINDOW (by accuracy)")
    print(f"{'='*100}")
    ds_pipe_pairs = sorted(set((a['agg']['dataset'], a['agg']['window_len'], a['agg']['pipeline'])
                               for a in all_results.values()))
    for dn, wl, pn in ds_pipe_pairs:
        subset = {k: v for k, v in all_results.items()
                  if v['agg']['dataset'] == dn and v['agg']['window_len'] == wl
                  and v['agg']['pipeline'] == pn}
        if not subset:
            continue
        best_key = max(subset, key=lambda k: subset[k]['agg'].get('accuracy_mean', 0))
        ba = subset[best_key]['agg']
        print(f"  {dn:<25} {wl:<10} {pn:<10}: {ba['architecture']}  "
              f"Acc={ba.get('accuracy_mean',0):.4f}  "
              f"F1={ba.get('f1_weighted_mean',0):.4f}")

    print(f"\n{'='*80}")
    print("DL experiments completed!")
    print(f"{'='*80}")

    # ---- Final save ----
    _save_dl_results()
    print(f"\n[info] Final per-seed results: {os.path.abspath(csv_path)}")
    print(f"[info] Final aggregated results: {os.path.abspath(agg_csv_path)}")

    return all_results


if __name__ == '__main__':
    import sys, os, argparse
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='DL Experiments')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', 'data'),
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--window-lens', type=int, nargs='+', default=[500, 1000, 2000],
                        help='Window lengths to compare (default: 500 1000 2000)')
    parser.add_argument('--sr', type=int, default=150, help='Guaranteed sample rate')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'mid', 'large'])
    parser.add_argument('--n-seeds', type=int, default=1,
                        help='Number of random seeds for multi-run (default: 1)')
    parser.add_argument('--cv', action='store_true',
                        help='Use temporal forward-chaining cross-validation')
    parser.add_argument('--n-folds', type=int, default=None,
                        help='Number of CV folds (auto if not set)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--var-windows', type=int, nargs='+', default=[20, 200, 2000],
                        help='Rolling variance window sizes to compare (default: 20 200 2000)')
    args = parser.parse_args()

    run_dl_experiment(
        data_root=os.path.abspath(args.data_root),
        window_lens=args.window_lens,
        guaranteed_sr=args.sr,
        epochs=args.epochs,
        lr=args.lr,
        model_size=args.model_size,
        verbose=args.verbose,
        n_seeds=args.n_seeds,
        cv_mode=args.cv,
        n_folds=args.n_folds,
        var_windows=args.var_windows,
    )
