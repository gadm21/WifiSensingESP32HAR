import os
import glob
import copy
import time
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def make_ml_models():
    """Return 5 sklearn classifiers for ML experiments."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('et', et), ('xgb', xgb), ('svm', svm)],
        voting='soft', n_jobs=-1,
    )
    return [
        ('RandomForest', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        # ('ExtraTrees', ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ('XGBoost', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)),
        # ('SVM_RBF', SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
        # ('Ensemble', ensemble),
    ]


def run_ml_experiment(name, train_ds, test_ds=None, adapt=False):
    """Run all 5 ML models on a dataset.

    Parameters
    ----------
    name : str
        Experiment name for display.
    train_ds : TrainingDataset
    test_ds : TrainingDataset, optional
    adapt : bool
        If True and test_ds is provided, run CM-based adaptation after each
        model and store before/after comparison in results.
    """
    results = {}
    for model_name, model in make_ml_models():
        print(f"\n--- {model_name} ---")
        job = TrainingJob(
            model=model, train_dataset=train_ds, test_dataset=test_ds,
            test_size=0.2, batch_size=64, epochs=50, lr=1e-3
        )
        metrics = job.run()

        if adapt and test_ds is not None:
            print(f"\n  >> Running CM adaptation for {model_name}...")
            X_train, X_test = train_ds.X, test_ds.X
            y_train, y_test = train_ds.y, test_ds.y
            adaptation = evaluate_with_adaptation(
                model=job.model,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                label_map=train_ds.label_map,
                is_torch=job._is_torch,
                device=getattr(job, 'device', 'cpu'),
                batch_size=64,
            )
            metrics['adaptation'] = adaptation

        results[model_name] = metrics
    return results


# =============================================================================
# CSI Subcarrier Selection Mask (HT20 non-STBC, 64 subcarriers / 128 bytes)
# Based on ESP-CSI documentation: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html
# Reference: https://github.com/espressif/esp-csi/issues/114
# =============================================================================
# Boolean mask: True = keep subcarrier, False = exclude subcarrier
# Total 64 subcarriers (indices 0-63), selecting 52 valid LLTF subcarriers

CSI_SUBCARRIER_MASK = np.array([
    False,  # 0:  Null guard subcarrier (lower edge protection)
    False,  # 1:  Null guard subcarrier (lower edge protection)
    False,  # 2:  Null guard subcarrier (lower edge protection)
    False,  # 3:  Null guard subcarrier (lower edge protection)
    False,  # 4:  Null guard subcarrier (lower edge protection)
    False,  # 5:  Null guard subcarrier (lower edge protection)
    True,   # 6:  LLTF valid subcarrier (negative frequency, index -26)
    True,   # 7:  LLTF valid subcarrier (negative frequency, index -25)
    True,   # 8:  LLTF valid subcarrier (negative frequency, index -24)
    True,   # 9:  LLTF valid subcarrier (negative frequency, index -23)
    True,   # 10: LLTF valid subcarrier (negative frequency, index -22)
    True,   # 11: LLTF valid subcarrier (negative frequency, index -21)
    True,   # 12: LLTF valid subcarrier (negative frequency, index -20)
    True,   # 13: LLTF valid subcarrier (negative frequency, index -19)
    True,   # 14: LLTF valid subcarrier (negative frequency, index -18)
    True,   # 15: LLTF valid subcarrier (negative frequency, index -17)
    True,   # 16: LLTF valid subcarrier (negative frequency, index -16)
    True,   # 17: LLTF valid subcarrier (negative frequency, index -15)
    True,   # 18: LLTF valid subcarrier (negative frequency, index -14)
    True,   # 19: LLTF valid subcarrier (negative frequency, index -13)
    True,   # 20: LLTF valid subcarrier (negative frequency, index -12)
    True,   # 21: LLTF pilot subcarrier (index -11) - contains channel info
    True,   # 22: LLTF valid subcarrier (negative frequency, index -10)
    True,   # 23: LLTF valid subcarrier (negative frequency, index -9)
    True,   # 24: LLTF valid subcarrier (negative frequency, index -8)
    True,   # 25: LLTF pilot subcarrier (index -7) - contains channel info
    True,   # 26: LLTF valid subcarrier (negative frequency, index -6)
    True,   # 27: LLTF valid subcarrier (negative frequency, index -5)
    True,   # 28: LLTF valid subcarrier (negative frequency, index -4)
    True,   # 29: LLTF valid subcarrier (negative frequency, index -3)
    True,   # 30: LLTF valid subcarrier (negative frequency, index -2)
    True,   # 31: LLTF valid subcarrier (negative frequency, index -1)
    False,  # 32: DC subcarrier (center frequency, always null)
    True,   # 33: LLTF valid subcarrier (positive frequency, index +1)
    True,   # 34: LLTF valid subcarrier (positive frequency, index +2)
    True,   # 35: LLTF valid subcarrier (positive frequency, index +3)
    True,   # 36: LLTF valid subcarrier (positive frequency, index +4)
    True,   # 37: LLTF valid subcarrier (positive frequency, index +5)
    True,   # 38: LLTF valid subcarrier (positive frequency, index +6)
    True,   # 39: LLTF pilot subcarrier (index +7) - contains channel info
    True,   # 40: LLTF valid subcarrier (positive frequency, index +8)
    True,   # 41: LLTF valid subcarrier (positive frequency, index +9)
    True,   # 42: LLTF valid subcarrier (positive frequency, index +10)
    True,   # 43: LLTF pilot subcarrier (index +11) - contains channel info
    True,   # 44: LLTF valid subcarrier (positive frequency, index +12)
    True,   # 45: LLTF valid subcarrier (positive frequency, index +13)
    True,   # 46: LLTF valid subcarrier (positive frequency, index +14)
    True,   # 47: LLTF valid subcarrier (positive frequency, index +15)
    True,   # 48: LLTF valid subcarrier (positive frequency, index +16)
    True,   # 49: LLTF valid subcarrier (positive frequency, index +17)
    True,   # 50: LLTF valid subcarrier (positive frequency, index +18)
    True,   # 51: LLTF valid subcarrier (positive frequency, index +19)
    True,   # 52: LLTF valid subcarrier (positive frequency, index +20)
    True,   # 53: LLTF valid subcarrier (positive frequency, index +21)
    True,   # 54: LLTF valid subcarrier (positive frequency, index +22)
    True,   # 55: LLTF valid subcarrier (positive frequency, index +23)
    True,   # 56: LLTF valid subcarrier (positive frequency, index +24)
    True,   # 57: LLTF valid subcarrier (positive frequency, index +25)
    True,   # 58: LLTF valid subcarrier (positive frequency, index +26)
    False,  # 59: Null guard subcarrier (upper edge protection)
    False,  # 60: Null guard subcarrier (upper edge protection)
    False,  # 61: Null guard subcarrier (upper edge protection)
    False,  # 62: Null guard subcarrier (upper edge protection)
    False,  # 63: Null guard subcarrier (upper edge protection)
], dtype=bool)  # Total: 52 True values (valid subcarriers)


# =============================================================================
# ProcessingBlock base class
# =============================================================================
class ProcessingBlock(ABC):
    """Abstract base class for all CSI processing blocks.

    All blocks operate on a data dictionary, preserving and accumulating errors.
    Subclasses must implement `process(data)`.

    Parameters
    ----------
    verbose : bool
        If True, print detailed information about the processing step including
        input/output shapes, data statistics, and transformation details.
        Useful for debugging and understanding the data flow. Default False.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def _log(self, msg):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  [{self.__class__.__name__}] {msg}")

    @abstractmethod
    def process(self, data):
        """Process the data dictionary. Must be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, data):
        if isinstance(data, dict):
            if 'errors' not in data:
                data['errors'] = []
            try:
                return self.process(data)
            except Exception as e:
                data['errors'].append(f"{self.__class__.__name__}: {e}")
                return data
        return self.process(data)


# =============================================================================
# CSI Loader
# =============================================================================
class CSI_Loader(ProcessingBlock):
    """Loads CSI data from CSV, converts 128 I/Q bytes to mag, phase, real, imag.

    The raw 128-byte CSI payload is split into 64 (imag, real) pairs.
    From those, magnitude and phase are computed directly (no complex
    intermediate).  All four representations are resampled to equal
    time intervals at ``guaranteed_sr`` Hz and returned in the output
    dictionary.

    As a ProcessingBlock, it can be the first block in a pipeline.
    Accepts either a filepath string or a data dict with 'filepath' key.

    Output keys
    -----------
    mag : np.ndarray, float64, (N, 64)
    phase : np.ndarray, float64, (N, 64)
    real : np.ndarray, float64, (N, 64)
    imag : np.ndarray, float64, (N, 64)
    rssi, timestamp, total_lines, read_lines, errors, resampling_stats

    Parameters
    ----------
    verbose : bool
        If True, print loading progress. Default False.
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.filepath = None
        self.guaranteed_sr = 150

    def is_valid(self):
        return (self.filepath is not None
                and os.path.exists(self.filepath)
                and self.filepath.endswith('.csv'))

    def process(self, data):
        # Accept filepath as string or from data dict
        if isinstance(data, str):
            filepath = data
        elif isinstance(data, dict) and 'filepath' in data:
            filepath = data['filepath']
        else:
            raise ValueError(f"CSI_Loader expects filepath string or dict with 'filepath' key, got {type(data)}")

        errors = []
        total_lines = 0
        read_lines = 0
        self.filepath = filepath
        if not self.is_valid():
            raise ValueError(f"Invalid CSI file: {filepath}")

        df = pd.read_csv(filepath, header=0, on_bad_lines='skip', low_memory=False)
        total_lines = len(df)

        # Filter out rows that don't start with 'CSI_DATA' (these are log messages)
        valid_mask = df['type'].str.startswith('CSI_DATA', na=False)
        df = df[valid_mask]
        total_lines = len(df)

        all_rssi = df['rssi'].values
        raw_csi = df['data'].values
        timestamp = df['local_timestamp'].values

        # Convert timestamps to numeric, filtering out any non-numeric values
        timestamp_series = pd.to_numeric(pd.Series(timestamp), errors='coerce')
        timestamp = timestamp_series.fillna(0).values.astype(np.int64)

        # Parse each row into separate real / imag arrays (no complex)
        real_list, imag_list, valid_rssi = [], [], []
        for numline, line in enumerate(raw_csi):
            try:
                csi_row = [int(x) for x in line[1:-1].split(",")]
                if len(csi_row) != 128:
                    errors.append(f"Line {numline}: expected 128 values, got {len(csi_row)}")
                    continue
                imag_list.append(csi_row[0::2])  # even indices
                real_list.append(csi_row[1::2])  # odd indices
                valid_rssi.append(all_rssi[numline])
                read_lines += 1
            except Exception as e:
                errors.append(f"Line {numline}: {e}")

        real = np.array(real_list, dtype=np.float64)  # (N, 64)
        imag = np.array(imag_list, dtype=np.float64)  # (N, 64)
        mag = np.sqrt(real ** 2 + imag ** 2)           # (N, 64)
        phase = np.arctan2(imag, real)                 # (N, 64)
        rssi = np.array(valid_rssi, dtype=np.float64)
        ts = timestamp[np.arange(len(real))]           # align timestamps

        # Resample to equal intervals at guaranteed_sr Hz
        resampling_stats = {}
        samples_per_bin = np.array([], dtype=np.int64)
        if len(real) > 0:
            mag, phase, real, imag, rssi, ts, samples_per_bin, resampling_stats = (
                self._resample_equal_intervals(mag, phase, real, imag, rssi, ts)
            )

        self._log(f"Loaded {filepath}")
        self._log(f"Total lines: {total_lines}, Valid: {read_lines}, Skipped: {total_lines - read_lines}")
        self._log(f"Shape: {mag.shape}, dtype: {mag.dtype}")
        self._log(f"RSSI shape: {rssi.shape}, range: [{rssi.min():.1f}, {rssi.max():.1f}]")
        if errors:
            self._log(f"Errors ({len(errors)}): {errors[:3]}{'...' if len(errors) > 3 else ''}")

        return {
            'mag': mag,
            'phase': phase,
            'real': real,
            'imag': imag,
            'rssi': rssi,
            'timestamp': ts,
            'total_lines': total_lines,
            'read_lines': read_lines,
            'errors': errors,
            'resampling_stats': resampling_stats,
            'samples_per_bin': samples_per_bin,
        }

    def _resample_equal_intervals(self, mag, phase, real, imag, rssi, timestamps):
        """Resample all channels to equal intervals at guaranteed_sr Hz.

        Uses vectorized bin assignment (np.searchsorted) instead of a
        per-bin Python loop, so it scales to 200k+ samples without stalling.

        Samples falling in the same time-bin are averaged.
        Empty bins are filled via linear interpolation.

        Returns
        -------
        tuple
            (mag, phase, real, imag, rssi, timestamps, samples_per_bin, stats)
            samples_per_bin is an int array of length n_out.
        """
        n_orig = len(timestamps)
        zero_spb = np.array([], dtype=np.int64)
        empty_stats = {
            'original_samples': 0, 'resampled_samples': 0,
            'empty_slots': 0, 'empty_slots_pct': 0.0,
            'overlapping_samples': 0, 'actual_sampling_rate': 0.0,
            'target_sampling_rate': self.guaranteed_sr,
            'duration_sec': 0.0,
            'samples_per_bin_stats': {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0},
        }
        if n_orig == 0:
            return mag, phase, real, imag, rssi, timestamps, zero_spb, empty_stats

        # Convert ESP32 microsecond timestamps to seconds
        ts_sec = timestamps.astype(np.float64) / 1_000_000
        start, end = ts_sec[0], ts_sec[-1]
        duration = end - start

        n_out = int(np.ceil(duration * self.guaranteed_sr))
        if n_out < 2:
            self._log(f"Warning: Too short duration ({duration:.3f}s) for resampling")
            trivial = empty_stats.copy()
            trivial.update(original_samples=n_orig, resampled_samples=n_orig,
                           actual_sampling_rate=n_orig / max(duration, 1e-8),
                           duration_sec=duration,
                           samples_per_bin_stats={'mean': 1.0, 'std': 0.0, 'min': 1, 'max': 1})
            spb = np.ones(n_orig, dtype=np.int64)
            return mag, phase, real, imag, rssi, timestamps, spb, trivial

        target_t = start + np.arange(n_out) / self.guaranteed_sr
        dt = 1.0 / self.guaranteed_sr
        n_sc = mag.shape[1]

        # --- Vectorized bin assignment ---
        # Bin edges: each bin i covers [target_t[i] - dt/2, target_t[i] + dt/2)
        bin_edges = target_t - dt / 2
        # np.searchsorted gives the bin index for each sample
        bin_idx = np.searchsorted(bin_edges, ts_sec, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_out - 1)

        # Count samples per bin
        samples_per_bin = np.bincount(bin_idx, minlength=n_out).astype(np.int64)

        # Accumulate sums per bin for each channel, then divide by count
        channels = {'mag': mag, 'phase': phase, 'real': real, 'imag': imag}
        resampled = {}
        for k, src in channels.items():
            acc = np.zeros((n_out, n_sc), dtype=np.float64)
            np.add.at(acc, bin_idx, src)
            resampled[k] = acc

        rssi_acc = np.zeros(n_out, dtype=np.float64)
        np.add.at(rssi_acc, bin_idx, rssi)

        # Divide by count where non-zero
        populated = samples_per_bin > 0
        for k in channels:
            resampled[k][populated] /= samples_per_bin[populated, None]
        rssi_acc[populated] /= samples_per_bin[populated]

        # Identify empty / overpopulated bins
        empty_mask = samples_per_bin == 0
        empty_slots = np.where(empty_mask)[0]

        # Interpolate empty bins
        if len(empty_slots) > 0:
            valid_idx = np.where(populated)[0]
            if len(valid_idx) >= 2:
                for k in channels:
                    for sc in range(n_sc):
                        resampled[k][empty_slots, sc] = np.interp(
                            target_t[empty_slots], target_t[valid_idx],
                            resampled[k][valid_idx, sc])
                rssi_acc[empty_slots] = np.interp(
                    target_t[empty_slots], target_t[valid_idx], rssi_acc[valid_idx])
            elif len(valid_idx) == 1:
                for k in channels:
                    resampled[k][empty_slots] = resampled[k][valid_idx[0]]
                rssi_acc[empty_slots] = rssi_acc[valid_idx[0]]
            else:
                self._log("Warning: No valid samples for interpolation")

        n_empty = int(empty_mask.sum())
        overlapping = int((samples_per_bin[samples_per_bin > 1] - 1).sum())
        empty_pct = 100 * n_empty / n_out if n_out > 0 else 0.0

        if n_empty and self.verbose:
            self._log(f"Empty slots: {n_empty}/{n_out} ({empty_pct:.1f}%)")

        resampled_ts = (target_t * 1_000_000).astype(np.int64)

        stats = {
            'original_samples': n_orig,
            'resampled_samples': n_out,
            'empty_slots': n_empty,
            'empty_slots_pct': empty_pct,
            'overlapping_samples': overlapping,
            'actual_sampling_rate': n_orig / max(duration, 1e-8),
            'target_sampling_rate': self.guaranteed_sr,
            'duration_sec': duration,
            'samples_per_bin_stats': {
                'mean': float(samples_per_bin.mean()),
                'std': float(samples_per_bin.std()),
                'min': int(samples_per_bin.min()),
                'max': int(samples_per_bin.max()),
            },
        }

        self._log(f"Resampled: {n_orig} -> {n_out} samples @ {self.guaranteed_sr} Hz")
        self._log(f"Empty slots: {n_empty} ({empty_pct:.1f}%), Overlapping: {overlapping}")

        return (resampled['mag'], resampled['phase'], resampled['real'],
                resampled['imag'], rssi_acc, resampled_ts, samples_per_bin, stats)
# Processing Blocks
# =============================================================================
class FeatureSelector(ProcessingBlock):
    """Applies a boolean mask to exclude subcarriers/features.

    Parameters
    ----------
    mask : np.ndarray or None
        Boolean mask array. True = keep, False = exclude. Default CSI_SUBCARRIER_MASK.
    keys : list[str]
        Data dict keys to filter. Default ['mag', 'phase', 'real', 'imag'].
    verbose : bool
        If True, print mask statistics (total features, kept features, excluded),
        input/output shapes, and which subcarrier indices are kept. Default False.
    """

    def __init__(self, mask=None, keys=None, verbose=False):
        super().__init__(verbose)
        self.mask = mask if mask is not None else CSI_SUBCARRIER_MASK
        self.keys = keys if keys is not None else ['mag', 'phase', 'real', 'imag']
        if not isinstance(self.mask, np.ndarray) or self.mask.dtype != bool:
            raise TypeError(f"mask must be boolean np.ndarray, got {type(self.mask)}")

    def process(self, data):
        if isinstance(data, dict):
            for key in self.keys:
                arr = data[key]
                if not isinstance(arr, np.ndarray):
                    raise TypeError(f"data['{key}'] must be np.ndarray, got {type(arr)}")
                if arr.ndim != 2 or arr.shape[1] != len(self.mask):
                    raise ValueError(f"expected 2D array with {len(self.mask)} cols, got shape {arr.shape}")
                data[key] = arr[:, self.mask]
            self._log(f"Mask: {len(self.mask)} total, {self.mask.sum()} kept, {(~self.mask).sum()} excluded")
            self._log(f"Applied to keys: {self.keys}")
            return data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"expected np.ndarray, got {type(data)}")
        return data[:, self.mask]


class CsiSanitizer(ProcessingBlock):
    """Phase sanitization via sparse delay-domain reconstruction (SHARP).

    Derives OFDM subcarrier indices from ``CSI_SUBCARRIER_MASK``, builds a
    frequency vector f_k = idx_k * Δf, constructs the delay dictionary
    T(f_k, t_p) = exp(-j 2π f_k t_p)  [Eq. (6)], and for every packet solves
    a Lasso inverse problem [Eq. (8)] to obtain sparse path coefficients r.
    The dominant path is used as phase reference to cancel hardware offsets
    [Eq. (14)-(15)], producing sanitized CFR Ĥ.

    This block expects **complex** CSI that has already been filtered by
    ``FeatureSelector`` (i.e. 52 kept subcarriers matching the True positions
    in ``CSI_SUBCARRIER_MASK``).

    Parameters
    ----------
    key : str
        Data-dict key holding the complex CSI array. Default ``'csi'``.
    delta_f : float
        OFDM subcarrier spacing in Hz. Default 312 500 (HT20).
    delta_t : float
        Delay-grid step in seconds. Default 1e-7.
    t_min : float
        Minimum delay bound in seconds. Default -3e-7.
    t_max : float
        Maximum delay bound in seconds. Default 5e-7.
    subcarrier_stride : int
        Use every *n*-th subcarrier in the Lasso solver for speed. Default 2.
    lasso_alpha : float
        L1 regularisation strength passed to ``sklearn.linear_model.Lasso``.
        Default 1e-4.
    mask : np.ndarray or None
        Boolean mask used to derive subcarrier indices.  Default
        ``CSI_SUBCARRIER_MASK``.
    verbose : bool
        If True, log shapes and timing. Default False.
    """

    def __init__(self, key='csi', delta_f=312_500.0, delta_t=1e-7,
                 t_min=-3e-7, t_max=5e-7, subcarrier_stride=2,
                 lasso_alpha=1e-4, mask=None, verbose=False):
        super().__init__(verbose)
        self.key = key
        self.delta_f = float(delta_f)
        self.delta_t = float(delta_t)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.subcarrier_stride = int(subcarrier_stride)
        self.lasso_alpha = float(lasso_alpha)

        _mask = mask if mask is not None else CSI_SUBCARRIER_MASK
        # Derive OFDM subcarrier indices: position p in the 64-pt buffer maps
        # to subcarrier index (p - 32).  Kept positions are where mask is True.
        positions = np.where(_mask)[0]                       # e.g. [6..31, 33..58]
        self._sc_indices = (positions - 32).astype(np.int64)  # [-26..-1, +1..+26]
        self._freq_vec = self._sc_indices.astype(np.float64) * self.delta_f

    # ------------------------------------------------------------------
    # Internal helpers (inlined so the class has no external dependencies
    # beyond numpy / scipy / sklearn which are already in requirements)
    # ------------------------------------------------------------------
    def _build_T_matrix(self, freq_vec_hz):
        """Delay dictionary T  [Eq. (6)]:  T_{k,p} = exp(-j 2π f_k t_p)."""
        delay_grid = np.arange(self.t_min, self.t_max + self.delta_t * 0.5,
                               self.delta_t)
        # T shape: (K, P) where K = len(freq_vec_hz), P = len(delay_grid)
        T = np.exp(-1j * 2 * np.pi
                   * freq_vec_hz[:, None] * delay_grid[None, :])
        return T, delay_grid

    def _solve_lasso(self, h, T, select_sc):
        """Solve  min ||h - T r||_2^2 + α||r||_1  via real-valued expansion.

        Uses ``sklearn.linear_model.Lasso`` (Eqs. 9-11).
        """
        T_sel = T[select_sc, :]          # (K_sel, P)
        h_sel = h[select_sc]             # (K_sel,)

        # Real-valued expansion for complex Lasso (Eqs. 9-11):
        #   h = T r  with r = r_re + j*r_im
        #   [Re(T), -Im(T)] [r_re]   [Re(h)]
        #   [Im(T),  Re(T)] [r_im] = [Im(h)]
        K_sel, P = T_sel.shape
        A_re = np.vstack([T_sel.real, T_sel.imag])     # (2K_sel, P)
        A_im = np.vstack([-T_sel.imag, T_sel.real])    # (2K_sel, P)
        A_full = np.hstack([A_re, A_im])               # (2K_sel, 2P)
        b_full = np.concatenate([h_sel.real, h_sel.imag])  # (2K_sel,)

        model = Lasso(alpha=self.lasso_alpha, fit_intercept=False,
                      max_iter=5000, tol=1e-6)
        model.fit(A_full, b_full)
        x = model.coef_                                # (2P,)
        r = x[:P] + 1j * x[P:]                         # (P,) complex
        return r

    def _sanitize(self, H):
        """Run full phase sanitization on H (N_pkts, K) complex."""
        T_matrix, _ = self._build_T_matrix(self._freq_vec)
        select_sc = np.arange(0, len(self._freq_vec), self.subcarrier_stride)

        H_sanitized = np.zeros_like(H, dtype=np.complex128)

        for n_idx in range(H.shape[0]):
            h = H[n_idx, :]
            r = self._solve_lasso(h, T_matrix, select_sc)

            p_star = int(np.argmax(np.abs(r)))
            Tr = T_matrix * r                                    # (K, P)
            ref_col = (T_matrix[:, p_star] * r[p_star]).reshape(-1, 1)
            Trr = Tr * np.conj(ref_col)
            H_sanitized[n_idx, :] = np.sum(Trr, axis=1)

        return H_sanitized

    # ------------------------------------------------------------------
    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or not np.iscomplexobj(arr):
                raise TypeError(
                    f"data['{self.key}'] must be complex np.ndarray, "
                    f"got dtype {getattr(arr, 'dtype', type(arr))}")
            K = arr.shape[1]
            if K != len(self._sc_indices):
                raise ValueError(
                    f"Expected {len(self._sc_indices)} subcarriers after "
                    f"FeatureSelector, got {K}")
            self._log(f"Input: {arr.shape}, dtype={arr.dtype}")
            self._log(f"Subcarrier indices: {self._sc_indices[[0, -1]]} "
                      f"({len(self._sc_indices)} tones)")
            self._log(f"Delay grid: [{self.t_min:.1e}, {self.t_max:.1e}] "
                      f"step {self.delta_t:.1e}")
            H_san = self._sanitize(arr)
            self._log(f"Output: {H_san.shape}, dtype={H_san.dtype}")
            data[self.key] = H_san
            return data

        if not isinstance(data, np.ndarray) or not np.iscomplexobj(data):
            raise TypeError("expected complex np.ndarray")
        return self._sanitize(data)


class AmplitudeExtractor(ProcessingBlock):
    """Extracts amplitude |z| from complex CSI. Input: complex, Output: float64.

    Parameters
    ----------
    key : str
        Data dict key containing complex CSI. Default 'csi'.
    output_key : str
        Key to store amplitude result. Default 'amplitude'.
    verbose : bool
        If True, print input complex array shape/dtype, output amplitude shape,
        and amplitude statistics (min, max, mean, std). Default False.
    """

    def __init__(self, key='csi', output_key='amplitude', verbose=False):
        super().__init__(verbose)
        self.key = key
        self.output_key = output_key

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or not np.iscomplexobj(arr):
                raise TypeError(f"data['{self.key}'] must be complex np.ndarray, got dtype {getattr(arr, 'dtype', type(arr))}")
            amp = np.abs(arr).astype(np.float64)
            self._log(f"Input: {arr.shape} {arr.dtype} -> Output: {amp.shape} {amp.dtype}")
            self._log(f"Amplitude range: [{amp.min():.4f}, {amp.max():.4f}], mean={amp.mean():.4f}, std={amp.std():.4f}")
            data[self.output_key] = amp
            return data
        if not isinstance(data, np.ndarray) or not np.iscomplexobj(data):
            raise TypeError(f"expected complex np.ndarray")
        return np.abs(data).astype(np.float64)


class PhaseExtractor(ProcessingBlock):
    """Extracts phase angle from complex CSI. Input: complex, Output: float64 radians.

    Parameters
    ----------
    key : str
        Data dict key containing complex CSI. Default 'csi'.
    output_key : str
        Key to store phase result. Default 'phase'.
    unwrap : bool
        If True, unwrap phase discontinuities along subcarrier axis. Default False.
    verbose : bool
        If True, print input shape, output shape, phase range in radians,
        and whether unwrapping was applied. Default False.
    """

    def __init__(self, key='csi', output_key='phase', unwrap=False, verbose=False):
        super().__init__(verbose)
        self.key = key
        self.output_key = output_key
        self.unwrap = unwrap

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or not np.iscomplexobj(arr):
                raise TypeError(f"data['{self.key}'] must be complex np.ndarray, got dtype {getattr(arr, 'dtype', type(arr))}")
            phase = np.angle(arr).astype(np.float64)
            if self.unwrap:
                phase = np.unwrap(phase, axis=1)
            self._log(f"Input: {arr.shape} -> Output: {phase.shape}")
            self._log(f"Phase range: [{phase.min():.4f}, {phase.max():.4f}] rad, unwrap={self.unwrap}")
            data[self.output_key] = phase
            return data
        if not isinstance(data, np.ndarray) or not np.iscomplexobj(data):
            raise TypeError(f"expected complex np.ndarray")
        phase = np.angle(data).astype(np.float64)
        if self.unwrap:
            phase = np.unwrap(phase, axis=1)
        return phase


class WindowTransformer(ProcessingBlock):
    """Windows (n_samples, features) -> (n_windows, win_len, features) or (n_windows, win_len*features).

    Parameters
    ----------
    window_length : int
        Number of samples per window.
    key : str
        Data dict key to window. Default 'amplitude'.
    mode : str
        'sequential' keeps 3D shape, 'flattened' reshapes to 2D. Default 'sequential'.
    stride : int or None
        Step between windows. Default equals window_length (non-overlapping).
    verbose : bool
        If True, print input samples, window count, output shape, overlap percentage,
        and samples discarded due to incomplete final window. Default False.
    """

    def __init__(self, window_length, key='amplitude', mode='sequential', stride=None, verbose=False):
        super().__init__(verbose)
        if not isinstance(window_length, int) or window_length <= 0:
            raise ValueError(f"window_length must be positive int, got {window_length}")
        if mode not in ('sequential', 'flattened'):
            raise ValueError(f"mode must be 'sequential' or 'flattened', got '{mode}'")
        self.window_length = window_length
        self.key = key
        self.mode = mode
        self.stride = stride if stride is not None else window_length
        if not isinstance(self.stride, int) or self.stride <= 0:
            raise ValueError(f"stride must be positive int, got {self.stride}")

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                raise ValueError(f"data['{self.key}'] must be 2D np.ndarray, got {getattr(arr, 'shape', type(arr))}")
            data[self.key] = self._window(arr)
            return data
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError(f"expected 2D np.ndarray")
        return self._window(data)

    def _window(self, arr):
        n_samples, n_features = arr.shape
        n_windows = (n_samples - self.window_length) // self.stride + 1
        if n_windows <= 0:
            raise ValueError(f"Not enough samples ({n_samples}) for window_length={self.window_length}")
        windows = np.array([arr[i * self.stride : i * self.stride + self.window_length] for i in range(n_windows)])
        samples_used = (n_windows - 1) * self.stride + self.window_length
        samples_discarded = n_samples - samples_used
        overlap_pct = (1 - self.stride / self.window_length) * 100 if self.window_length > 0 else 0
        self._log(f"Input: ({n_samples}, {n_features}), window_length={self.window_length}, stride={self.stride}")
        self._log(f"Windows: {n_windows}, overlap: {overlap_pct:.1f}%, discarded: {samples_discarded} samples")
        if self.mode == 'flattened':
            windows = windows.reshape(n_windows, -1)
        self._log(f"Output shape: {windows.shape}, mode='{self.mode}'")
        return windows


class FFTTransformer(ProcessingBlock):
    """Applies FFT along time axis. Works on 2D or 3D arrays.

    Parameters
    ----------
    key : str
        Data dict key to transform. Default 'amplitude'.
    mode : str
        Output mode: 'complex', 'magnitude', or 'db'. Default 'magnitude'.
    real_only : bool
        If True, use rfft (real FFT) for efficiency. Default True.
    axis : int
        Axis along which to compute FFT. Default -2 (time axis).
    flatten : bool
        If True and output is 3D, flatten to 2D (n_windows, freq_bins*features).
        Default False.
    verbose : bool
        If True, print input/output shapes, frequency bin count, mode used,
        and output value range. Default False.
    """

    def __init__(self, key='amplitude', mode='magnitude', real_only=True, axis=-2, flatten=False, verbose=False):
        super().__init__(verbose)
        if mode not in ('complex', 'magnitude', 'db'):
            raise ValueError(f"mode must be 'complex', 'magnitude', or 'db', got '{mode}'")
        self.key = key
        self.mode = mode
        self.real_only = real_only
        self.axis = axis
        self.flatten = flatten

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or arr.ndim not in (2, 3):
                raise ValueError(f"data['{self.key}'] must be 2D or 3D np.ndarray")
            data[self.key] = self._fft(arr)
            return data
        if not isinstance(data, np.ndarray) or data.ndim not in (2, 3):
            raise ValueError(f"expected 2D or 3D np.ndarray")
        return self._fft(data)

    def _fft(self, arr):
        squeezed = False
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
            squeezed = True
        if self.real_only:
            fft_out = np.fft.rfft(arr, axis=self.axis)
        else:
            fft_out = np.fft.fft(arr, axis=self.axis)
        if self.mode == 'complex':
            result = fft_out
        elif self.mode == 'magnitude':
            result = np.abs(fft_out)
        elif self.mode == 'db':
            result = 20 * np.log10(np.abs(fft_out) + 1e-9)
        if squeezed:
            result = result.squeeze(axis=0)
        if self.flatten and result.ndim == 3:
            n = result.shape[0]
            result = result.reshape(n, -1)
        self._log(f"FFT: {arr.shape} -> {result.shape}")
        return result


class STFTTransformer(ProcessingBlock):
    """Applies Short-Time Fourier Transform along the time axis.

    Converts a 2D array (n_samples, n_features) into a 3D spectrogram
    (n_frames, n_freq_bins, n_features) or flattened 2D output.

    Parameters
    ----------
    key : str
        Data dict key to transform. Default 'amplitude'.
    nperseg : int
        Length of each STFT segment (window size). Default 64.
    noverlap : int or None
        Number of overlapping samples between segments. Default nperseg // 2.
    window : str
        Window function name (passed to scipy.signal.stft). Default 'hann'.
    mode : str
        Output mode: 'magnitude', 'power', 'complex', or 'db'. Default 'magnitude'.
    output_key : str or None
        If set, store result under this key instead of overwriting input key.
    verbose : bool
        If True, print input shape, output shape, and STFT parameters. Default False.
    """

    def __init__(self, key='amplitude', nperseg=64, noverlap=None,
                 window='hann', mode='magnitude', output_key=None, verbose=False):
        super().__init__(verbose)
        if mode not in ('complex', 'magnitude', 'power', 'db'):
            raise ValueError(f"mode must be 'complex', 'magnitude', 'power', or 'db', got '{mode}'")
        self.key = key
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.window = window
        self.mode = mode
        self.output_key = output_key if output_key is not None else key

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                raise ValueError(f"data['{self.key}'] must be 2D np.ndarray, got {getattr(arr, 'shape', type(arr))}")
            result = self._stft(arr)
            self._log(f"Input: {arr.shape} -> Output: {result.shape}")
            self._log(f"STFT params: nperseg={self.nperseg}, noverlap={self.noverlap}, window='{self.window}', mode='{self.mode}'")
            self._log(f"Output: (n_frames, n_freq_bins, n_features) = {result.shape}")
            data[self.output_key] = result
            return data
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError(f"expected 2D np.ndarray, got {getattr(data, 'shape', type(data))}")
        return self._stft(data)

    def _stft(self, arr):
        from scipy.signal import stft
        n_samples, n_features = arr.shape
        spectrograms = []
        for col in range(n_features):
            f, t, Zxx = stft(arr[:, col], fs=1.0, window=self.window,
                             nperseg=self.nperseg, noverlap=self.noverlap)
            if self.mode == 'complex':
                spectrograms.append(Zxx)
            elif self.mode == 'magnitude':
                spectrograms.append(np.abs(Zxx))
            elif self.mode == 'power':
                spectrograms.append(np.abs(Zxx) ** 2)
            elif self.mode == 'db':
                spectrograms.append(20 * np.log10(np.abs(Zxx) + 1e-9))
        # Stack: (n_freq_bins, n_time_frames, n_features)
        result = np.stack(spectrograms, axis=-1)
        # Transpose to (n_time_frames, n_freq_bins, n_features)
        result = result.transpose(1, 0, 2)
        return result



class Augmentor(ProcessingBlock):
    """Applies data augmentation to CSI amplitude/phase arrays.

    Augmentation types (all independently toggleable):
    - gaussian_noise: Add Gaussian noise to simulate interference.
    - amplitude_scaling: Random per-sample scaling for distance variations.
    - time_warp: Stretch/compress sequences for temporal drifts.

    Parameters
    ----------
    key : str
        Data dict key to augment. Default 'amplitude'.
    output_key : str or None
        If set, store result under this key instead of overwriting input key.
        This allows keeping original data alongside augmented data.
    gaussian_noise : bool
        Enable Gaussian noise injection. Default True.
    noise_std : float or tuple
        Noise standard deviation. If tuple (lo, hi), sampled uniformly per sample.
        Default (0.1, 0.5).
    amplitude_scaling : bool
        Enable random amplitude scaling. Default True.
    scale_range : tuple
        (min_scale, max_scale) multiplier range. Default (0.8, 1.2).
    time_warp : bool
        Enable time-warping (stretch/compress along time axis). Default True.
    warp_range : tuple
        (min_factor, max_factor) as fractions of original length.
        Default (0.9, 1.1) for ±10%.
    seed : int or None
        Random seed for reproducibility. Default None.
    verbose : bool
        If True, print augmentation parameters and output shape. Default False.
    """

    def __init__(self, key='amplitude', output_key=None, gaussian_noise=True, noise_std=(0.1, 0.5),
                 amplitude_scaling=True, scale_range=(0.8, 1.2),
                 time_warp=True, warp_range=(0.9, 1.1), seed=None, verbose=False):
        self.key = key
        self.output_key = output_key if output_key is not None else key
        self.gaussian_noise = gaussian_noise
        self.noise_std = noise_std
        self.amplitude_scaling = amplitude_scaling
        self.scale_range = scale_range
        self.time_warp = time_warp
        self.warp_range = warp_range
        self.rng = np.random.RandomState(seed)
        super().__init__(verbose)

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"data['{self.key}'] must be np.ndarray, got {type(arr)}")
            aug = self._augment(arr)
            self._log(f"Input: {arr.shape} -> Output: {aug.shape}")
            self._log(f"Augmentations: noise={self.gaussian_noise}, scale={self.amplitude_scaling}, warp={self.time_warp}")
            if self.gaussian_noise:
                self._log(f"  Noise std: {self.noise_std}")
            if self.amplitude_scaling:
                self._log(f"  Scale range: {self.scale_range}")
            data[self.output_key] = aug
            return data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"expected np.ndarray, got {type(data)}")
        return self._augment(data)

    def _augment(self, arr):
        arr = arr.copy()
        if self.gaussian_noise:
            arr = self._add_noise(arr)
        if self.amplitude_scaling:
            arr = self._scale(arr)
        if self.time_warp:
            arr = self._warp(arr)
        return arr

    def _add_noise(self, arr):
        if isinstance(self.noise_std, (tuple, list)):
            std = self.rng.uniform(self.noise_std[0], self.noise_std[1])
        else:
            std = self.noise_std
        noise = self.rng.normal(0, std, size=arr.shape)
        return arr + noise

    def _scale(self, arr):
        lo, hi = self.scale_range
        if arr.ndim == 2:
            scales = self.rng.uniform(lo, hi, size=(arr.shape[0], 1))
        elif arr.ndim == 3:
            scales = self.rng.uniform(lo, hi, size=(arr.shape[0], 1, 1))
        else:
            scales = self.rng.uniform(lo, hi)
        return arr * scales

    def _warp(self, arr):
        from scipy.interpolate import interp1d
        if arr.ndim == 2:
            n_samples, n_features = arr.shape
            factor = self.rng.uniform(self.warp_range[0], self.warp_range[1])
            new_len = max(2, int(round(n_samples * factor)))
            x_old = np.linspace(0, 1, n_samples)
            x_new = np.linspace(0, 1, new_len)
            f = interp1d(x_old, arr, axis=0, kind='linear', fill_value='extrapolate')
            return f(x_new)
        elif arr.ndim == 3:
            n_windows, win_len, n_features = arr.shape
            warped = []
            for i in range(n_windows):
                factor = self.rng.uniform(self.warp_range[0], self.warp_range[1])
                new_len = max(2, int(round(win_len * factor)))
                x_old = np.linspace(0, 1, win_len)
                x_new = np.linspace(0, 1, new_len)
                f = interp1d(x_old, arr[i], axis=0, kind='linear', fill_value='extrapolate')
                warped_win = f(x_new)
                x_back = np.linspace(0, 1, new_len)
                x_orig = np.linspace(0, 1, win_len)
                f_back = interp1d(x_back, warped_win, axis=0, kind='linear', fill_value='extrapolate')
                warped.append(f_back(x_orig))
            return np.array(warped)
        return arr


class Normalizer(ProcessingBlock):
    """
    Z-score normalizes per subcarrier/feature column (mean=0, std=1).

    Handles amplitude scaling shifts from distance/environment changes.
    Computes statistics along the time axis (axis=0 for 2D, per-window for 3D).

    Parameters
    ----------
    key : str
        Data dict key to normalize. Default 'amplitude'.
    output_key : str or None
        If set, store result under this key instead of overwriting input key.
    eps : float
        Small constant added to std to avoid division by zero. Default 1e-8.
    verbose : bool
        If True, print input shape, output shape, and normalization statistics. Default False.
    """

    def __init__(self, key='amplitude', output_key=None, eps=1e-8, verbose=False):
        super().__init__(verbose)
        self.key = key
        self.output_key = output_key if output_key is not None else key
        self.eps = eps

    def process(self, data):
        if isinstance(data, dict):
            arr = data[self.key]
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"data['{self.key}'] must be np.ndarray, got {type(arr)}")
            norm = self._normalize(arr)
            self._log(f"Input: {arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")
            self._log(f"Output: {norm.shape}, range=[{norm.min():.4f}, {norm.max():.4f}]")
            self._log(f"Post-norm mean={norm.mean():.6f}, std={norm.std():.6f}")
            data[self.output_key] = norm
            return data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"expected np.ndarray, got {type(data)}")
        return self._normalize(data)

    def _normalize(self, arr):
        if arr.ndim == 2:
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True)
            return (arr - mean) / (std + self.eps)
        elif arr.ndim == 3:
            mean = arr.mean(axis=1, keepdims=True)
            std = arr.std(axis=1, keepdims=True)
            return (arr - mean) / (std + self.eps)
        return arr


class ConcatBlock(ProcessingBlock):
    """Concatenates multiple keys from one or more data dicts.

    Supports three modes:
    - Single dict: concatenates values of multiple keys along feature axis.
    - List of dicts, axis=-1 (feature concat): takes one key from each dict
      and concatenates along the feature axis. All dicts must have the same
      number of samples (rows).
    - List of dicts, axis=0 (sample concat): takes one key from each dict
      and concatenates along the sample axis (row-wise). All dicts must have
      the same number of features (columns). This is the mode to use when
      joining original + augmented data.

    Parameters
    ----------
    keys : list of str
        Keys to concatenate from data dict(s).
    output_key : str
        Key to store concatenated result. Default 'features'.
    axis : int
        Concatenation axis. -1 for feature concat, 0 for sample concat. Default -1.
    verbose : bool
        If True, print keys being concatenated, individual array shapes,
        output shape, and axis used. Default False.
    """

    def __init__(self, keys, output_key='features', axis=-1, verbose=False):
        super().__init__(verbose)
        if not isinstance(keys, (list, tuple)) or len(keys) < 1:
            raise ValueError("keys must be a non-empty list of strings")
        self.keys = keys
        self.output_key = output_key
        self.axis = axis

    def process(self, data):
        # Case 1: list of dicts — take one key from each and concat
        if isinstance(data, (list, tuple)):
            arrays = []
            errors = []
            for i, d in enumerate(data):
                if not isinstance(d, dict):
                    raise TypeError(f"element {i} must be dict, got {type(d)}")
                errors.extend(d.get('errors', []))
                key = self.keys[i] if i < len(self.keys) else self.keys[0]
                if key not in d:
                    raise KeyError(f"key '{key}' not found in dict {i}")
                arrays.append(d[key])

            # Flatten 3D to 2D if needed before concat
            flat = []
            for a in arrays:
                if a.ndim == 1:
                    flat.append(a.reshape(-1, 1))
                elif a.ndim == 3:
                    flat.append(a.reshape(a.shape[0], -1))
                else:
                    flat.append(a)

            if self.axis == 0:
                # Sample-wise concat: column counts must match
                n_cols = flat[0].shape[1]
                for i, a in enumerate(flat):
                    if a.shape[1] != n_cols:
                        raise ValueError(
                            f"column mismatch: array 0 has {n_cols} cols, "
                            f"array {i} has {a.shape[1]}")
            else:
                # Feature-wise concat: row counts must match
                n_rows = flat[0].shape[0]
                for i, a in enumerate(flat):
                    if a.shape[0] != n_rows:
                        raise ValueError(
                            f"row mismatch: array 0 has {n_rows} rows, "
                            f"array {i} has {a.shape[0]}")

            result = data[0].copy() if isinstance(data[0], dict) else {}
            result[self.output_key] = np.concatenate(flat, axis=self.axis)
            result['errors'] = errors
            self._log(f"Keys: {self.keys}, axis={self.axis}")
            self._log(f"Input shapes: {[a.shape for a in flat]}")
            self._log(f"Output: '{self.output_key}' shape={result[self.output_key].shape}")
            return result

        # Case 2: single dict — concat multiple keys from it
        if isinstance(data, dict):
            arrays = []
            for key in self.keys:
                if key not in data:
                    raise KeyError(f"key '{key}' not found in data")
                arr = data[key]
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    elif arr.ndim == 3:
                        arr = arr.reshape(arr.shape[0], -1)
                    arrays.append(arr)
                else:
                    raise TypeError(f"data['{key}'] must be np.ndarray, got {type(arr)}")

            n_rows = arrays[0].shape[0]
            for i, a in enumerate(arrays):
                if a.shape[0] != n_rows:
                    raise ValueError(f"row mismatch: '{self.keys[0]}' has {n_rows} rows, '{self.keys[i]}' has {a.shape[0]}")

            data[self.output_key] = np.concatenate(arrays, axis=self.axis)
            self._log(f"Keys: {self.keys}, axis={self.axis}")
            self._log(f"Input shapes: {[a.shape for a in arrays]}")
            self._log(f"Output: '{self.output_key}' shape={data[self.output_key].shape}")
            return data

        raise TypeError(f"expected dict or list of dicts, got {type(data)}")


# =============================================================================
# Pipeline
# =============================================================================
class Pipeline:
    """Chains ProcessingBlocks. First block should be CSI_Loader.
    
    Called with a filepath, passes it through all blocks sequentially.
    """

    def __init__(self, blocks):
        self.blocks = blocks
        for b in blocks:
            if not isinstance(b, ProcessingBlock):
                raise TypeError(f"All blocks must be ProcessingBlock, got {type(b)}")
        if len(blocks) == 0 or not isinstance(blocks[0], CSI_Loader):
            raise TypeError("First block must be CSI_Loader")

    def __call__(self, filepath):
        data = filepath  # Start with filepath, CSI_Loader will handle it
        for block in self.blocks:
            data = block(data)
        return data


# =============================================================================
# DatasetFile
# =============================================================================
class DatasetFile:
    """Represents a single CSI file with labels and a processing pipeline.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    pipeline : Pipeline
        Processing pipeline to apply.
    labels : list of str
        Labels for this file (e.g. ['drink', 'activity']).
        First label is primary unless primary_label is set.
    primary_label : str, optional
        Override which label is primary.
    """

    def __init__(self, filepath, pipeline, labels, primary_label=None):
        self.filepath = filepath
        self.pipeline = pipeline
        self.labels = labels if isinstance(labels, list) else [labels]
        self.primary_label = primary_label if primary_label else self.labels[0]
        self._data = None

    def load(self):
        """Apply pipeline and cache result."""
        self._data = self.pipeline(self.filepath)
        return self._data

    @property
    def data(self):
        if self._data is None:
            self.load()
        return self._data


# =============================================================================
# TrainingDataset
# =============================================================================
METADATA_FILENAME = 'dataset_metadata.json'

class TrainingDataset:
    """Aggregates multiple DatasetFile objects into X, y arrays for training.

    Parameters
    ----------
    dataset_files : list of DatasetFile
    feature_key : str
        Key in the processed data dict to use as features. Default 'features'.
    label_map : dict, optional
        Maps label strings to integers. Auto-generated if None.
    balance : bool
        If True, trim oversampled classes to match the smallest class. Default False.
    """

    def __init__(self, dataset_files, feature_key='features', label_map=None, balance=False):
        self.dataset_files = dataset_files
        self.feature_key = feature_key
        self.label_map = label_map
        self.balance = balance
        self._X = None
        self._y = None
        # Metadata fields (populated by from_metadata)
        self._name = None
        self._labels = None
        self._description = ''
        self._environment = ''
        self._task = ''

    def build(self):
        """Load all files and build X, y arrays."""
        all_X = []
        all_y = []
        all_labels = set()

        for df in self.dataset_files:
            all_labels.add(df.primary_label)

        if self.label_map is None:
            self.label_map = {label: i for i, label in enumerate(sorted(all_labels))}

        for df in self.dataset_files:
            data = df.data
            if self.feature_key not in data:
                raise KeyError(f"feature_key '{self.feature_key}' not found in data for {df.filepath}")
            X = data[self.feature_key]
            # Keep 3D sequential data as-is for transformer models
            # if X.ndim == 3:
            #     X = X.reshape(X.shape[0], -1)
            y_val = self.label_map[df.primary_label]
            y = np.full(X.shape[0], y_val, dtype=np.int64)
            all_X.append(X)
            all_y.append(y)

        self._X = np.concatenate(all_X, axis=0)
        self._y = np.concatenate(all_y, axis=0)

        if self.balance:
            self._balance_classes()

        return self._X, self._y

    def _balance_classes(self):
        """Trim oversampled classes to match the smallest class."""
        classes, counts = np.unique(self._y, return_counts=True)
        min_count = counts.min()
        balanced_idx = []
        rng = np.random.RandomState(42)
        for cls in classes:
            cls_idx = np.where(self._y == cls)[0]
            if len(cls_idx) > min_count:
                cls_idx = rng.choice(cls_idx, size=min_count, replace=False)
            balanced_idx.append(cls_idx)
        balanced_idx = np.concatenate(balanced_idx)
        balanced_idx.sort()
        self._X = self._X[balanced_idx]
        self._y = self._y[balanced_idx]

    @property
    def X(self):
        if self._X is None:
            self.build()
        return self._X

    @property
    def y(self):
        if self._y is None:
            self.build()
        return self._y

    @property
    def num_classes(self):
        return len(self.label_map) if self.label_map else 0

    @property
    def input_size(self):
        return self.X.shape[1] if self._X is not None else 0

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """Index into the dataset. Returns (x, y) tuple."""
        return self.X[idx], self.y[idx]

    @property
    def name(self):
        return self._name or ''

    @property
    def labels(self):
        """Sorted list of label strings."""
        if self._labels is not None:
            return self._labels
        if self.label_map:
            return sorted(self.label_map.keys())
        return []

    @property
    def train_X(self):
        """Alias for X (for API compat with train/test split workflow)."""
        return self.X

    @property
    def train_y(self):
        """Alias for y."""
        return self.y

    def get_fewshot(self, n_per_class=10):
        """Get few-shot samples from the BEGINNING of the dataset.

        Parameters
        ----------
        n_per_class : int
            Number of samples per class. Default 10.

        Returns
        -------
        X_fs, y_fs : np.ndarray
        """
        X, y = self.X, self.y
        idxs = []
        for cls in sorted(np.unique(y)):
            cls_idx = np.where(y == cls)[0]
            k = min(n_per_class, len(cls_idx))
            if k > 0:
                idxs.append(cls_idx[:k])
        if not idxs:
            return np.empty((0,) + X.shape[1:]), np.empty((0,), dtype=np.int64)
        idxs = np.concatenate(idxs)
        return X[idxs], y[idxs]

    def get_torch_dataset(self):
        """Returns a PyTorch TensorDataset."""
        import torch
        from torch.utils.data import TensorDataset
        X_t = torch.FloatTensor(self.X)
        y_t = torch.LongTensor(self.y)
        return TensorDataset(X_t, y_t)

    @staticmethod
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
        var = np.clip(mean_sq - means ** 2, 0, None)
        return var

    @staticmethod
    def _window_array_static(arr, window_len, stride, mode):
        """Window a 2D array into (n_windows, ...) -- static helper."""
        n_samples, n_features = arr.shape
        n_windows = (n_samples - window_len) // stride + 1
        if n_windows <= 0:
            return None
        windows = np.array([
            arr[i * stride: i * stride + window_len]
            for i in range(n_windows)
        ])
        if mode == 'flattened':
            windows = windows.reshape(n_windows, -1)
        return windows

    @classmethod
    def from_metadata(cls, root_dir, pipeline_name='amplitude', window_len=100,
                      guaranteed_sr=100, mode='flattened', stride=None,
                      var_window=20, balance=False, verbose=False):
        """Load a dataset from a directory containing dataset_metadata.json.

        Reads the metadata, builds appropriate processing pipelines, creates
        train/test splits, and returns a (train_ds, test_ds) pair.

        Parameters
        ----------
        root_dir : str
            Path to the dataset folder (must contain dataset_metadata.json).
        pipeline_name : str
            'amplitude', 'amplitude_phase', 'amplitude_sanitized', or
            'rolling_variance'.
        window_len : int
            Window length. Default 100.
        guaranteed_sr : int
            Resampling rate. Default 100.
        mode : str
            'flattened' or 'sequential'. Default 'flattened'.
        stride : int or None
            Window stride. None = window_len (non-overlapping).
        var_window : int
            Rolling variance window size. Default 20.
        balance : bool
            Balance classes. Default False.
        verbose : bool

        Returns
        -------
        train_ds, test_ds : TrainingDataset
            Two datasets. test_ds may have 0 samples if no test files.
        """
        import json

        root_dir = os.path.abspath(root_dir)
        meta_path = os.path.join(root_dir, METADATA_FILENAME)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        ds_name = metadata['name']
        labels_list = sorted(metadata['labels'])
        label_map = {lbl: i for i, lbl in enumerate(labels_list)}
        train_pct = metadata.get('train_pct', 0.8)
        stride = stride if stride is not None else window_len

        loader = CSI_Loader(verbose=verbose)
        loader.guaranteed_sr = guaranteed_sr
        selector = FeatureSelector(mask=CSI_SUBCARRIER_MASK, verbose=verbose)

        print(f"\n[TrainingDataset] Loading '{ds_name}' ({len(metadata['files'])} files, "
              f"pipeline={pipeline_name}, window={window_len}, mode={mode})")

        train_files_data = []
        test_files_data = []

        for entry in metadata['files']:
            fpath = os.path.join(root_dir, entry['path'])
            label = entry['label']
            split = entry.get('split', 'percentage')

            if label not in label_map:
                print(f"  SKIP {entry['path']} -- label '{label}' not in labels")
                continue

            try:
                data = loader.process(fpath)
                data = selector.process(data)
            except Exception as e:
                print(f"  ERROR {entry['path']}: {e}")
                continue

            mag = data['mag']
            phase = data['phase']

            if pipeline_name == 'amplitude':
                X = cls._window_array_static(mag, window_len, stride, mode)

            elif pipeline_name == 'amplitude_phase':
                mag_w = cls._window_array_static(mag, window_len, stride, mode)
                phase_w = cls._window_array_static(phase, window_len, stride, mode)
                if mag_w is None or phase_w is None:
                    print(f"  SKIP {entry['path']} -- too few samples")
                    continue
                if mode == 'flattened':
                    X = np.concatenate([mag_w, phase_w], axis=1)
                else:
                    X = np.concatenate([mag_w, phase_w], axis=2)

            elif pipeline_name == 'amplitude_sanitized':
                csi_complex = data['real'] + 1j * data['imag']
                sanitizer = CsiSanitizer(verbose=verbose)
                csi_san = sanitizer._sanitize(csi_complex)
                san_phase = np.angle(csi_san).astype(np.float64)
                mag_w = cls._window_array_static(mag, window_len, stride, mode)
                phase_w = cls._window_array_static(san_phase, window_len, stride, mode)
                if mag_w is None or phase_w is None:
                    print(f"  SKIP {entry['path']} -- too few samples")
                    continue
                if mode == 'flattened':
                    X = np.concatenate([mag_w, phase_w], axis=1)
                else:
                    X = np.concatenate([mag_w, phase_w], axis=2)

            elif pipeline_name == 'rolling_variance':
                rv = cls._rolling_variance(mag, var_window)
                X = cls._window_array_static(rv, window_len, stride, mode)

            else:
                raise ValueError(f"Unknown pipeline: {pipeline_name}")

            if X is None or len(X) == 0:
                print(f"  SKIP {entry['path']} -- no windows produced")
                continue

            y_val = label_map[label]

            if split == 'train':
                train_files_data.append((X, label, y_val))
                print(f"  TRAIN {entry['path']:30s}  label={label:8s}  windows={X.shape[0]}")
            elif split == 'test':
                test_files_data.append((X, label, y_val))
                print(f"  TEST  {entry['path']:30s}  label={label:8s}  windows={X.shape[0]}")
            elif split == 'percentage':
                n = X.shape[0]
                n_train = max(1, int(n * train_pct))
                train_files_data.append((X[:n_train], label, y_val))
                if n > n_train:
                    test_files_data.append((X[n_train:], label, y_val))
                print(f"  SPLIT {entry['path']:30s}  label={label:8s}  "
                      f"train={n_train}  test={n - n_train}")

        if not train_files_data:
            raise RuntimeError(f"No training data loaded for '{ds_name}'")

        train_X = np.concatenate([d[0] for d in train_files_data], axis=0)
        train_y = np.concatenate([np.full(d[0].shape[0], d[2], dtype=np.int64)
                                  for d in train_files_data], axis=0)

        train_ds = cls.__new__(cls)
        train_ds.dataset_files = []
        train_ds.feature_key = 'mag'
        train_ds.label_map = label_map
        train_ds.balance = balance
        train_ds._X = train_X
        train_ds._y = train_y
        train_ds._name = ds_name
        train_ds._labels = labels_list
        train_ds._description = metadata.get('description', '')
        train_ds._environment = metadata.get('environment', '')
        train_ds._task = metadata.get('task', '')

        if balance:
            train_ds._balance_classes()

        if test_files_data:
            test_X = np.concatenate([d[0] for d in test_files_data], axis=0)
            test_y = np.concatenate([np.full(d[0].shape[0], d[2], dtype=np.int64)
                                     for d in test_files_data], axis=0)
        else:
            test_X = np.empty((0,) + train_X.shape[1:], dtype=train_X.dtype)
            test_y = np.empty((0,), dtype=np.int64)

        test_ds = cls.__new__(cls)
        test_ds.dataset_files = []
        test_ds.feature_key = 'mag'
        test_ds.label_map = label_map
        test_ds.balance = False
        test_ds._X = test_X
        test_ds._y = test_y
        test_ds._name = ds_name
        test_ds._labels = labels_list
        test_ds._description = metadata.get('description', '')
        test_ds._environment = metadata.get('environment', '')
        test_ds._task = metadata.get('task', '')

        print(f"[TrainingDataset] '{ds_name}' built: "
              f"train={train_X.shape}, test={test_X.shape}, "
              f"classes={len(labels_list)} {labels_list}")

        return train_ds, test_ds


def load_all_datasets(data_root, window_len=100, guaranteed_sr=100,
                      pipeline_name='amplitude', mode='flattened',
                      stride=None, var_window=20, verbose=False):
    """Load all 4 built-in datasets from data_root using TrainingDataset.

    Parameters
    ----------
    data_root : str
        Root folder containing the 4 dataset subfolders.

    Returns
    -------
    dict : {dataset_name: (train_ds, test_ds)}
    """
    DATASET_DIRS = [
        'home_har_data',
        'home_occupation_data',
        'office_har_data',
        'office_localization_data',
    ]

    datasets = {}
    for dname in DATASET_DIRS:
        dpath = os.path.join(data_root, dname)
        meta_path = os.path.join(dpath, METADATA_FILENAME)
        if not os.path.isfile(meta_path):
            print(f"[warn] Skipping {dname} -- no {METADATA_FILENAME}")
            continue
        train_ds, test_ds = TrainingDataset.from_metadata(
            root_dir=dpath,
            pipeline_name=pipeline_name,
            window_len=window_len,
            guaranteed_sr=guaranteed_sr,
            mode=mode,
            stride=stride,
            var_window=var_window,
            verbose=verbose,
        )
        datasets[train_ds.name] = (train_ds, test_ds)

    return datasets


# FederatedPartitioner
class FederatedPartitioner:
    """Partitions a TrainingDataset using Dirichlet distribution (pure numpy).

    Creates non-IID federated data splits without external dependencies.

    Parameters
    ----------
    dataset : TrainingDataset
        The dataset to partition.
    num_partitions : int
        Number of federated partitions (clients).
    alpha : float
        Dirichlet concentration parameter. Lower = more heterogeneous.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, dataset, num_partitions, alpha=0.5, seed=42):
        self.dataset = dataset
        self.num_partitions = num_partitions
        self.alpha = alpha

        rng = np.random.RandomState(seed)
        X, y = dataset.X, dataset.y
        classes = np.unique(y)
        n_classes = len(classes)

        # Dirichlet partitioning: for each class, sample proportions across clients
        partition_indices = [[] for _ in range(num_partitions)]
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet([alpha] * num_partitions)
            # Convert proportions to counts
            counts = (proportions * len(cls_idx)).astype(int)
            # Distribute remainder
            remainder = len(cls_idx) - counts.sum()
            for i in range(remainder):
                counts[i % num_partitions] += 1
            # Assign indices
            start = 0
            for pid in range(num_partitions):
                partition_indices[pid].append(cls_idx[start:start + counts[pid]])
                start += counts[pid]

        self._partition_indices = [np.concatenate(idxs) for idxs in partition_indices]

    def load_partition(self, partition_id):
        """Load a partition as a TrainingDataset-compatible object.

        Parameters
        ----------
        partition_id : int
            Partition index (0 to num_partitions-1).

        Returns
        -------
        TrainingDataset
            A new TrainingDataset containing only the partition's data.
        """
        idx = self._partition_indices[partition_id]
        part_ds = TrainingDataset.__new__(TrainingDataset)
        part_ds.dataset_files = []
        part_ds.feature_key = self.dataset.feature_key
        part_ds.label_map = self.dataset.label_map
        part_ds.balance = False
        part_ds._X = self.dataset.X[idx]
        part_ds._y = self.dataset.y[idx]
        return part_ds


# =============================================================================
# TrainingJob
# =============================================================================
class TrainingJob:
    """Trains a PyTorch nn.Module or sklearn model on a TrainingDataset.

    Accepts either a single dataset (split internally via test_size) or
    separate train_dataset and test_dataset.

    Parameters
    ----------
    model : torch.nn.Module or sklearn estimator
    train_dataset : TrainingDataset
        Training data. Required.
    test_dataset : TrainingDataset, optional
        Separate test data. If provided, test_size is ignored.
    test_size : float
        Fraction of train_dataset to hold out for testing. Default 0.2.
        Ignored when test_dataset is provided.
    batch_size : int
        Batch size for PyTorch training. Default 64.
    epochs : int
        Number of epochs for PyTorch training. Default 50.
    lr : float
        Learning rate for PyTorch training. Default 1e-3.
    device : str
        'cuda' or 'cpu'. Default auto-detect.
    posttraining_pipeline : list of callable, optional
        A list of processing blocks (callables) applied to X_test before
        testing evaluation, after training is complete. Each callable receives
        a 2D np.ndarray and returns a 2D np.ndarray. Default None.
    """
    def __init__(self, model, train_dataset, test_dataset=None, test_size=0.2,
                 batch_size=64, epochs=50, lr=1e-3, device=None,
                 posttraining_pipeline=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.posttraining_pipeline = posttraining_pipeline or []
        self.metrics = {}
        self._is_torch = self._check_torch(model)
        if device is None:
            if self._is_torch:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = 'cpu'
        else:
            self.device = device

    @staticmethod
    def _check_torch(model):
        try:
            import torch.nn as nn
            return isinstance(model, nn.Module)
        except ImportError:
            return False

    def _prepare_data(self):
        """Build train/test splits from datasets.

        Time-series aware: for each class, the last test_size fraction of
        samples (in original order) is used for testing. No shuffling or
        stratified random splitting — later data is always test data to
        avoid information leakage.
        """
        if self.test_dataset is not None:
            X_train, y_train = self.train_dataset.X, self.train_dataset.y
            X_test, y_test = self.test_dataset.X, self.test_dataset.y
        else:
            X, y = self.train_dataset.X, self.train_dataset.y
            train_idx = []
            test_idx = []
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0]
                n_test = max(1, int(len(cls_idx) * self.test_size))
                split_point = len(cls_idx) - n_test
                train_idx.append(cls_idx[:split_point])
                test_idx.append(cls_idx[split_point:])
            train_idx = np.concatenate(train_idx)
            test_idx = np.concatenate(test_idx)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
        return X_train, X_test, y_train, y_test

    def _apply_posttraining(self, X_test):
        """Apply posttraining pipeline blocks to X_test."""
        for block in self.posttraining_pipeline:
            X_test = block(X_test)
        return X_test

    def run(self):
        """Execute training, apply posttraining pipeline, then evaluate."""
        X_train, X_test, y_train, y_test = self._prepare_data()

        n_total = X_train.shape[0] + X_test.shape[0]
        n_features = X_train.shape[1]
        n_classes = self.train_dataset.num_classes
        print(f"Dataset: {n_total} samples, {n_features} features, {n_classes} classes")
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        print(f"Label map: {self.train_dataset.label_map}")

        # --- Phase 1: Train ---
        if self._is_torch:
            train_result = self._fit_torch(X_train, y_train)
        else:
            train_result = self._fit_sklearn(X_train, y_train)

        # --- Phase 2: Posttraining pipeline on test data ---
        if self.posttraining_pipeline:
            print(f"Applying posttraining pipeline ({len(self.posttraining_pipeline)} blocks) to test data...")
            X_test = self._apply_posttraining(X_test)
            print(f"Post-pipeline test shape: {X_test.shape}")

        # --- Phase 3: Evaluate ---
        if self._is_torch:
            self.metrics = self._evaluate_torch(X_train, y_train, X_test, y_test, train_result)
        else:
            self.metrics = self._evaluate_sklearn(X_train, y_train, X_test, y_test, train_result)

        return self.metrics

    # ----- sklearn -----
    def _fit_sklearn(self, X_train, y_train):
        print(f"\nTraining sklearn model: {type(self.model).__name__}")
        t0 = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - t0
        return {'train_time_s': round(train_time, 2)}

    def _evaluate_sklearn(self, X_train, y_train, X_test, y_test, train_result):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        avg = 'weighted' if self.train_dataset.num_classes > 2 else 'binary'
        metrics = {
            'model_type': 'sklearn',
            'model_name': type(self.model).__name__,
            'train_time_s': train_result['train_time_s'],
            'train_accuracy': round(accuracy_score(y_train, y_pred_train), 4),
            'test_accuracy': round(accuracy_score(y_test, y_pred_test), 4),
            'test_precision': round(precision_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
            'test_recall': round(recall_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
            'test_f1': round(f1_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
        }
        self._print_metrics(metrics)
        return metrics

    # ----- PyTorch -----
    def _fit_torch(self, X_train, y_train):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self.device)
        self.model = self.model.to(device)

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"\nTraining PyTorch model on {device} for {self.epochs} epochs...")
        history = {'train_loss': [], 'train_acc': []}

        t0 = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            train_loss = running_loss / total
            train_acc = correct / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if (epoch + 1) % max(1, self.epochs // 10) == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        train_time = time.time() - t0
        return {'train_time_s': round(train_time, 2), 'history': history}

    def _evaluate_torch(self, X_train, y_train, X_test, y_test, train_result):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        device = torch.device(self.device)
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(y_batch.numpy())

        y_pred = np.array(all_preds)
        y_true = np.array(all_true)
        avg = 'weighted' if self.train_dataset.num_classes > 2 else 'binary'

        history = train_result.get('history', {})
        metrics = {
            'model_type': 'pytorch',
            'model_name': type(self.model).__name__,
            'train_time_s': train_result['train_time_s'],
            'epochs': self.epochs,
            'train_accuracy': round(history['train_acc'][-1], 4) if history.get('train_acc') else None,
            'test_accuracy': round(accuracy_score(y_true, y_pred), 4),
            'test_precision': round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
            'test_recall': round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
            'test_f1': round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'history': history,
        }
        self._print_metrics(metrics)
        return metrics

    @staticmethod
    def _print_metrics(m):
        print(f"\n{'='*50}")
        print(f"Results: {m['model_name']} ({m['model_type']})")
        print(f"{'='*50}")
        print(f"  Train time:     {m['train_time_s']}s")
        print(f"  Train accuracy: {m['train_accuracy']}")
        print(f"  Test accuracy:  {m['test_accuracy']}")
        print(f"  Test precision: {m['test_precision']}")
        print(f"  Test recall:    {m['test_recall']}")
        print(f"  Test F1:        {m['test_f1']}")
        print(f"  Confusion matrix:")
        for row in m['confusion_matrix']:
            print(f"    {row}")


# =============================================================================
# Confusion Matrix Adaptation (domain drift correction)
# =============================================================================
class ConfusionMatrixAdapter:
    """Corrects model predictions using the inverse of the train-set confusion matrix.

    When a model is trained on domain A and tested on domain B, systematic
    misclassifications appear as off-diagonal mass in the confusion matrix.
    This adapter:
      1. Row-normalises the train CM to get P(predicted | true).
      2. Computes the pseudo-inverse of that matrix.
      3. Row-normalises the inverse so each row sums to 1.
    At inference the model's probability vector is left-multiplied by this
    correction matrix, shifting mass back toward the true class.

    Parameters
    ----------
    cm : array-like, shape (n_classes, n_classes)
        Confusion matrix from the *training* evaluation (rows = true, cols = predicted).
    eps : float
        Small constant to avoid division by zero. Default 1e-8.
    """

    def __init__(self, cm, eps=1e-8):
        cm = np.array(cm, dtype=np.float64)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(f"cm must be square, got shape {cm.shape}")
        self.n_classes = cm.shape[0]
        self.raw_cm = cm.copy()
        self.eps = eps
        self.correction = self._build_correction(cm)

    def _build_correction(self, cm):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / (row_sums + self.eps)
        cm_inv = np.linalg.pinv(cm_norm)
        cm_inv = np.clip(cm_inv, 0, None)
        inv_row_sums = cm_inv.sum(axis=1, keepdims=True)
        correction = cm_inv / (inv_row_sums + self.eps)
        return correction

    def adapt_probs(self, probs):
        """Apply correction to probability matrix (n_samples, n_classes)."""
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        adapted = probs @ self.correction.T
        adapted = np.clip(adapted, 0, None)
        row_sums = adapted.sum(axis=1, keepdims=True)
        adapted = adapted / (row_sums + self.eps)
        return adapted

    def adapt_predictions(self, probs):
        """Return hard class predictions after adaptation."""
        adapted = self.adapt_probs(probs)
        return np.argmax(adapted, axis=1)

    def summary(self):
        """Print the correction matrix."""
        print(f"\n  [CMAdapter] Raw CM:\n{self.raw_cm}")
        print(f"  [CMAdapter] Correction matrix:\n{np.round(self.correction, 4)}")


def evaluate_with_adaptation(model, X_train, y_train, X_test, y_test,
                             label_map=None, is_torch=False, device='cpu',
                             batch_size=64):
    """Evaluate a trained model before and after CM-based adaptation.

    1. Predict on train set -> build confusion matrix -> build adapter.
    2. Predict on test set (probabilities).
    3. Compute metrics from raw predictions (before).
    4. Apply adapter to test probabilities -> compute metrics (after).
    5. Print side-by-side comparison.

    Parameters
    ----------
    model : sklearn estimator or torch.nn.Module
        Already-trained model.
    X_train, y_train : np.ndarray
        Training data (used to build the CM).
    X_test, y_test : np.ndarray
        Test data to evaluate.
    label_map : dict, optional
        {label_str: int} mapping for display.
    is_torch : bool
        True if model is a PyTorch nn.Module.
    device : str
        PyTorch device. Ignored for sklearn.
    batch_size : int
        Batch size for PyTorch inference.

    Returns
    -------
    dict with keys 'before', 'after', 'adapter'.
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix)

    def _get_probs_and_preds(model, X, is_torch, device, batch_size):
        if is_torch:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            model.eval()
            ds = TensorDataset(torch.FloatTensor(X))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            all_probs = []
            with torch.no_grad():
                for (X_batch,) in loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs.cpu().numpy())
            probs = np.concatenate(all_probs, axis=0)
        else:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
            else:
                preds = model.predict(X)
                n_classes = len(np.unique(y_train))
                probs = np.zeros((len(preds), n_classes))
                probs[np.arange(len(preds)), preds] = 1.0
        preds = np.argmax(probs, axis=1)
        return probs, preds

    def _compute_metrics(y_true, y_pred, n_classes):
        avg = 'weighted' if n_classes > 2 else 'binary'
        return {
            'accuracy':  round(accuracy_score(y_true, y_pred), 4),
            'precision': round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
            'recall':    round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
            'f1':        round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))

    # --- Step 1: train-set predictions -> CM -> adapter ---
    train_probs, train_preds = _get_probs_and_preds(model, X_train, is_torch, device, batch_size)
    train_cm = confusion_matrix(y_train, train_preds)
    adapter = ConfusionMatrixAdapter(train_cm)
    adapter.summary()

    # --- Step 2: test-set predictions (raw) ---
    test_probs, test_preds_raw = _get_probs_and_preds(model, X_test, is_torch, device, batch_size)
    before = _compute_metrics(y_test, test_preds_raw, n_classes)

    # --- Step 3: adapted test predictions ---
    test_preds_adapted = adapter.adapt_predictions(test_probs)
    after = _compute_metrics(y_test, test_preds_adapted, n_classes)

    # --- Step 4: print comparison ---
    print(f"\n{'='*60}")
    print(f"  CM-ADAPTATION COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<15} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'-'*45}")
    for key in ['accuracy', 'precision', 'recall', 'f1']:
        b, a = before[key], after[key]
        delta = a - b
        sign = '+' if delta >= 0 else ''
        print(f"  {key:<15} {b:>10.4f} {a:>10.4f} {sign}{delta:>9.4f}")

    if label_map:
        inv_map = {v: k for k, v in label_map.items()}
    else:
        inv_map = {i: str(i) for i in range(n_classes)}

    print(f"\n  Confusion Matrix BEFORE adaptation:")
    for row in before['confusion_matrix']:
        print(f"    {row}")
    print(f"\n  Confusion Matrix AFTER adaptation:")
    for row in after['confusion_matrix']:
        print(f"    {row}")
    print(f"{'='*60}")

    return {'before': before, 'after': after, 'adapter': adapter}

# =============================================================================
def load_csi_datasets(train_dirs, test_dirs, window_len, verbose=False):
    """Load and return train/test TrainingDatasets from CSI CSV files."""
    pipeline = Pipeline([
        CSI_Loader(verbose=verbose),
        FeatureSelector(verbose=verbose),
        # CsiSanitizer(verbose=verbose),
        # ConcatBlock(keys=['mag', 'rssi'], output_key='features', axis=-1, verbose=verbose),
        # Normalizer(key='features', output_key='features', verbose=verbose),
        # Augmentor(key='features', output_key='aug', gaussian_noise=True, noise_std=(0.1, 0.3),
        #           amplitude_scaling=True, scale_range=(0.85, 1.15), time_warp=False, seed=42, verbose=verbose),
        # ConcatBlock(keys=['features', 'aug'], output_key='concat', axis=0, verbose=verbose),
        WindowTransformer(window_length=window_len, key='mag', mode='sequential', verbose=verbose, stride=window_len//3),
        # FFTTransformer(key='features', mode='magnitude', real_only=True, axis=-2, flatten=True, verbose=verbose),
    ])

    test_pipeline = Pipeline([
        CSI_Loader(verbose=verbose),
        FeatureSelector(verbose=verbose),
        # CsiSanitizer(verbose=verbose),
        # ConcatBlock(keys=['mag', 'rssi'], output_key='features', axis=-1, verbose=verbose),
        # Normalizer(key='features', output_key='features', verbose=verbose),
        # ConcatBlock(keys=['features', 'mag'], output_key='concat', axis=-1, verbose=verbose),
        WindowTransformer(window_length=window_len, key='mag', mode='sequential', verbose=verbose, stride=window_len//3),
        # FFTTransformer(key='features', mode='magnitude', real_only=True, axis=-2, flatten=True, verbose=verbose),
    ])

    # labels = ['empty', 'smoke', 'watch', 'work', 'sleep']
    labels = ['drink', 'eat', 'empty', 'smoke', 'watch', 'work', 'sleep']
    # labels = ['drink', 'eat', 'empty', 'smoke', 'watch', 'work']

    # Collect train files: for each dir × label, glob for CSVs starting with that label
    train_files = []  # list of (label, filepath)
    for train_dir in train_dirs:
        for label in labels:
            for f in sorted(glob.glob(f'{train_dir}/{label}*.csv')):
                train_files.append((label, f))
    
    # Collect test files: same flexible search
    test_files = []  # list of (label, filepath)
    for test_dir in test_dirs:
        for label in labels:
            for f in sorted(glob.glob(f'{test_dir}/{label}*.csv')):
                test_files.append((label, f))
    
    ds_files = [DatasetFile(p, pipeline, [l]) for l, p in train_files]
    train_ds = TrainingDataset(ds_files, feature_key='mag', balance=True)
    train_ds.build()
    
    test_ds_files = [DatasetFile(p, test_pipeline, [l]) for l, p in test_files]
    test_ds = TrainingDataset(test_ds_files, feature_key='mag', label_map=train_ds.label_map, balance=True)
    test_ds.build()

    return train_ds, test_ds


# =============================================================================
# ML Experiments: 4 pipelines × 4 datasets × 4 models
# =============================================================================
def run_ml_pipeline_experiment(data_root, window_len=100, guaranteed_sr=100,
                               var_window=20, verbose=False):
    """Run comprehensive ML experiment across all datasets and pipelines.

    Compares 4 ML models on 4 feature pipelines across 4 datasets.
    Reports accuracy, F1, precision, recall, train time, and resource usage.

    Parameters
    ----------
    data_root : str
        Root folder containing the 4 dataset subfolders.
    window_len : int
        Window length. Default 100.
    guaranteed_sr : int
        Resampling rate. Default 100.
    var_window : int
        Rolling variance window. Default 20.
    verbose : bool
        Verbose CSI loading. Default False.

    Returns
    -------
    dict : nested {dataset__pipeline: {model: metrics}}
    """
    import traceback
    import psutil

    PIPELINES = ['amplitude', 'amplitude_phase', 'amplitude_sanitized', 'rolling_variance']

    all_results = {}

    for pipe_name in PIPELINES:
        print(f"\n{'#'*80}")
        print(f"# PIPELINE: {pipe_name}")
        print(f"{'#'*80}")

        try:
            datasets = load_all_datasets(
                data_root, window_len=window_len, guaranteed_sr=guaranteed_sr,
                pipeline_name=pipe_name, mode='flattened', stride=None,
                var_window=var_window, verbose=verbose,
            )
        except Exception as e:
            print(f"  ERROR loading datasets for pipeline '{pipe_name}': {e}")
            traceback.print_exc()
            continue

        for ds_name, (train_ds, test_ds) in datasets.items():
            key = f"{ds_name}__{pipe_name}"
            print(f"\n{'='*70}")
            print(f"  Dataset: {ds_name}  |  Pipeline: {pipe_name}")
            print(f"  Train: {train_ds.X.shape}  Test: {test_ds.X.shape}  "
                  f"Classes: {train_ds.num_classes} {train_ds.labels}")
            print(f"{'='*70}")

            if test_ds.X.shape[0] == 0:
                print(f"  SKIP — no test data")
                continue

            models = make_ml_models()
            run_results = {}

            for model_name, model in models:
                print(f"\n  --- {model_name} ---")
                proc = psutil.Process(os.getpid())
                mem_before = proc.memory_info().rss / 1024 / 1024

                t0 = time.process_time()
                try:
                    model.fit(train_ds.X, train_ds.y)
                except Exception as e:
                    print(f"    FIT ERROR: {e}")
                    continue
                train_time = time.process_time() - t0

                mem_after = proc.memory_info().rss / 1024 / 1024

                t1 = time.process_time()
                y_pred = model.predict(test_ds.X)
                infer_time = time.process_time() - t1

                from sklearn.metrics import (
                    accuracy_score, f1_score, precision_score,
                    recall_score, confusion_matrix, cohen_kappa_score,
                    matthews_corrcoef, balanced_accuracy_score,
                )

                acc = round(accuracy_score(test_ds.y, y_pred), 4)
                bal_acc = round(balanced_accuracy_score(test_ds.y, y_pred), 4)
                f1_w = round(f1_score(test_ds.y, y_pred, average='weighted', zero_division=0), 4)
                f1_mac = round(f1_score(test_ds.y, y_pred, average='macro', zero_division=0), 4)
                prec = round(precision_score(test_ds.y, y_pred, average='weighted', zero_division=0), 4)
                rec = round(recall_score(test_ds.y, y_pred, average='weighted', zero_division=0), 4)
                kappa = round(cohen_kappa_score(test_ds.y, y_pred), 4)
                mcc = round(matthews_corrcoef(test_ds.y, y_pred), 4)
                cm = confusion_matrix(test_ds.y, y_pred).tolist()

                metrics = {
                    'accuracy': acc,
                    'balanced_accuracy': bal_acc,
                    'f1_weighted': f1_w,
                    'f1_macro': f1_mac,
                    'precision_weighted': prec,
                    'recall_weighted': rec,
                    'cohen_kappa': kappa,
                    'mcc': mcc,
                    'confusion_matrix': cm,
                    'train_time_s': round(train_time, 3),
                    'inference_time_s': round(infer_time, 3),
                    'memory_delta_mb': round(mem_after - mem_before, 1),
                    'train_samples': train_ds.X.shape[0],
                    'test_samples': test_ds.X.shape[0],
                    'n_features': train_ds.X.shape[1],
                }

                print(f"    Acc={acc}  BalAcc={bal_acc}  F1w={f1_w}  "
                      f"Kappa={kappa}  MCC={mcc}  "
                      f"Train={train_time:.2f}s  Infer={infer_time:.3f}s  "
                      f"Mem={mem_after - mem_before:+.1f}MB")

                run_results[model_name] = metrics

            all_results[key] = run_results

    # ---- Final comparison table ----
    print(f"\n{'='*160}")
    print("FINAL ML COMPARISON: 4 Pipelines x 4 Datasets x 4 Models")
    print(f"{'='*160}")
    hdr = (f"{'Dataset':<25} {'Pipeline':<22} {'Model':<18} | "
           f"{'Acc':>6} {'BalAcc':>6} {'F1w':>6} {'F1mac':>6} "
           f"{'Prec':>6} {'Rec':>6} {'Kappa':>6} {'MCC':>6} | "
           f"{'Train':>7} {'Infer':>7} {'Mem':>6}")
    print(hdr)
    print("-" * 160)
    for key, run_res in all_results.items():
        parts = key.split('__')
        ds_name, pipe = parts[0], parts[1]
        for mname, m in run_res.items():
            print(f"{ds_name:<25} {pipe:<22} {mname:<18} | "
                  f"{m['accuracy']:>6.4f} {m['balanced_accuracy']:>6.4f} "
                  f"{m['f1_weighted']:>6.4f} {m['f1_macro']:>6.4f} "
                  f"{m['precision_weighted']:>6.4f} {m['recall_weighted']:>6.4f} "
                  f"{m['cohen_kappa']:>6.4f} {m['mcc']:>6.4f} | "
                  f"{m['train_time_s']:>6.2f}s {m['inference_time_s']:>6.3f}s "
                  f"{m['memory_delta_mb']:>+5.1f}M")
        print("-" * 160)

    # ---- Save results to CSV ----
    import csv
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'ml_pipeline_results.csv')
    fieldnames = ['dataset', 'pipeline', 'model',
                  'accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro',
                  'precision_weighted', 'recall_weighted', 'cohen_kappa', 'mcc',
                  'train_time_s', 'inference_time_s', 'memory_delta_mb',
                  'train_samples', 'test_samples', 'n_features']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, run_res in all_results.items():
            parts = key.split('__')
            ds_name, pipe = parts[0], parts[1]
            for model_name, m in run_res.items():
                row = {'dataset': ds_name, 'pipeline': pipe, 'model': model_name}
                row.update({k: m[k] for k in fieldnames if k in m})
                writer.writerow(row)
    print(f"\n[info] Results saved to {os.path.abspath(csv_path)}")

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ML Pipeline Experiments')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', '..', 'wifi_sensing_data'),
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--window', type=int, default=300, help='Window length')
    parser.add_argument('--sr', type=int, default=150, help='Guaranteed sample rate')
    parser.add_argument('--var-window', type=int, default=20, help='Rolling variance window')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    run_ml_pipeline_experiment(
        data_root=os.path.abspath(args.data_root),
        window_len=args.window,
        guaranteed_sr=args.sr,
        var_window=args.var_window,
        verbose=args.verbose,
    )