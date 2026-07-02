"""
Physics-informed full-range (0.1-100 uM) voltammogram data augmentation.

Object-oriented port of `full_range_data_augmentation.ipynb` (sections 0-9:
calibration loading, PCHIP Ip-concentration spline, Long & Winefordner noise
model, the 4-ingredient augmentation pipeline, log-uniform stratified
sampling and feature vectorization). The ablation study and the SNR audit
(notebook sections 10-11) are evaluation/research code, not generation code,
and are intentionally left out of this module - they stay in the notebook.

Every tunable knob lives on `AugmentationConfig`, so a Streamlit frontend
can bind widgets directly to its fields and pass it into
`FullRangeDataAugmentor.generate()`.

Typical usage
-------------
>>> augmentor = FullRangeDataAugmentor.from_excel(
...     'datasets/Standard calibration in culture media_extended.xlsx',
...     concentrations_uM=CONCENTRATIONS)
>>> dataset = augmentor.generate(AugmentationConfig(n_total_synthetic=300, low_conc_boost=2.0))
>>> dataset.to_raw_dataframe().head()
>>> dataset.save_raw_csv('raw/raw_signals_augmented.csv')
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.stats import linregress

from peak import Peak
from voltammogram_signal import Signal

GLOBAL_RANDOM_SEED = 42

DEFAULT_ANCHOR_CONCENTRATIONS_uM = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 25.0, 50.0, 100.0)


# Section 1 - Calibration data (real anchor curves)
class CalibrationDataset:
    """Real calibration replicates plus the mean baseline-subtracted anchor curves."""

    def __init__(self, potential_grid_V: np.ndarray, blank_baseline_uA: np.ndarray,
                 raw_signal_matrix_uA: np.ndarray, concentrations_uM: Sequence[float],
                 anchor_concentrations_uM: Sequence[float] = DEFAULT_ANCHOR_CONCENTRATIONS_uM):
        self.potential_grid_V = np.asarray(potential_grid_V, dtype=float)
        self.blank_baseline_uA = np.asarray(blank_baseline_uA, dtype=float)
        self.raw_signal_matrix_uA = np.asarray(raw_signal_matrix_uA, dtype=float)
        self.concentrations_uM = list(concentrations_uM)
        self.anchor_concentrations_uM = list(anchor_concentrations_uM)

        self.base_curves_ = {
            c: self._compute_mean_baseline_subtracted_signal(c)
            for c in self.anchor_concentrations_uM
        }

    @classmethod
    def from_excel(cls, path: str, concentrations_uM: Sequence[float],
                    sheet_name: str = 'Raw data',
                    anchor_concentrations_uM: Sequence[float] = DEFAULT_ANCHOR_CONCENTRATIONS_uM):
        """Load the raw calibration sheet (potential | blank | one column per replicate)."""
        df = pd.read_excel(path, sheet_name=sheet_name)
        potential_grid_V = df.iloc[1:, 0].values.astype(float)
        blank_baseline_uA = df.iloc[1:, 1].values.astype(float)
        raw_signal_matrix_uA = df.iloc[1:, 2:].values.astype(float)
        return cls(potential_grid_V, blank_baseline_uA, raw_signal_matrix_uA,
                   concentrations_uM, anchor_concentrations_uM)

    def _compute_mean_baseline_subtracted_signal(self, target_concentration_uM: float) -> np.ndarray:
        """Mean, across replicates, of the baseline-subtracted signal at one concentration."""
        column_indices = [i for i, c in enumerate(self.concentrations_uM) if c == target_concentration_uM]
        if not column_indices:
            raise ValueError(f'No replicates found for anchor concentration {target_concentration_uM} uM')
        replicates = [self.raw_signal_matrix_uA[:, i] - self.blank_baseline_uA for i in column_indices]
        return np.mean(replicates, axis=0)

    def base_curve(self, concentration_uM: float) -> np.ndarray:
        return self.base_curves_[concentration_uM]

    def n_replicates(self, concentration_uM: float) -> int:
        return sum(1 for c in self.concentrations_uM if c == concentration_uM)

    def peak_current_table(self, savgol_window: int = 5, savgol_polyorder: int = 3) -> pd.DataFrame:
        """Smoothed Ip/Ep at every anchor concentration (used to fit the PCHIP spline)."""
        rows = []
        for c in self.anchor_concentrations_uM:
            pk = Peak(self.potential_grid_V, savgol_filter(self.base_curves_[c], savgol_window, savgol_polyorder))
            rows.append(dict(concentration_uM=c, Ip=float(pk.Ip), Ep=float(pk.Ep), n_replicates=self.n_replicates(c)))
        return pd.DataFrame(rows)

    def real_signals_baseline_subtracted(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, y): every real replicate as a row, baseline already subtracted."""
        X = self.raw_signal_matrix_uA.T - self.blank_baseline_uA
        y = np.array(self.concentrations_uM, dtype=float)
        return X, y


# Section 2 - PCHIP Ip-concentration spline
class PeakCurrentSpline:
    """Monotone cubic Ip(c) interpolant (Fritsch & Carlson 1980) - no overshoot, captures saturation."""

    def __init__(self, calibration: CalibrationDataset, savgol_window: int = 5, savgol_polyorder: int = 3):
        table = calibration.peak_current_table(savgol_window, savgol_polyorder)
        self.concentration_array = table['concentration_uM'].to_numpy()
        self.peak_current_array = table['Ip'].to_numpy()
        self._spline = PchipInterpolator(self.concentration_array, self.peak_current_array)

    def __call__(self, concentration_uM: float) -> float:
        return float(self._spline(concentration_uM))

    def dense_grid(self, n_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
        c = np.linspace(self.concentration_array.min(), self.concentration_array.max(), n_points)
        return c, self._spline(c)


# Section 3 - Long & Winefordner empirical noise model
class LongWinefordnerNoiseModel:
    """sigma(c)^2 = sigma_abs^2 + (RSD * Ip(c))^2, with a hard SNR floor.

    Fit by linear regression of per-anchor replicate variance vs Ip^2
    (Long & Winefordner 1983, Anal. Chem. 55(7):712A-724A).
    """

    def __init__(self, sigma_abs_uA: float, rsd_relative: float, r_squared: float,
                 ip_spline: PeakCurrentSpline, snr_floor: float = 3.0):
        self.sigma_abs_uA = sigma_abs_uA
        self.rsd_relative = rsd_relative
        self.r_squared = r_squared
        self.ip_spline = ip_spline
        self.snr_floor = snr_floor

    @classmethod
    def fit(cls, calibration: CalibrationDataset, ip_spline: PeakCurrentSpline,
            snr_floor: float = 3.0, savgol_window: int = 5, savgol_polyorder: int = 3,
            min_replicates: int = 2) -> 'LongWinefordnerNoiseModel':
        means, stds = [], []
        for anchor_c in calibration.anchor_concentrations_uM:
            idxs = [i for i, c in enumerate(calibration.concentrations_uM) if c == anchor_c]
            if len(idxs) < min_replicates:
                continue
            ip_replicates = []
            for idx in idxs:
                I = calibration.raw_signal_matrix_uA[:, idx] - calibration.blank_baseline_uA
                pk = Peak(calibration.potential_grid_V, savgol_filter(I, savgol_window, savgol_polyorder))
                ip_replicates.append(pk.Ip)
            means.append(np.mean(ip_replicates))
            stds.append(np.std(ip_replicates, ddof=1))
        means_arr, stds_arr = np.array(means), np.array(stds)
        slope, intercept, r, *_ = linregress(means_arr ** 2, stds_arr ** 2)
        sigma_abs = float(np.sqrt(max(intercept, 0.0)))
        rsd = float(np.sqrt(max(slope, 0.0)))
        return cls(sigma_abs, rsd, float(r ** 2), ip_spline, snr_floor=snr_floor)

    def sigma(self, concentration_uM: float) -> float:
        """L&W sigma at `concentration_uM`, clipped so peak-SNR never drops below `snr_floor`."""
        Ip_c = self.ip_spline(concentration_uM)
        sigma = np.sqrt(self.sigma_abs_uA ** 2 + (self.rsd_relative * Ip_c) ** 2)
        sigma_max = Ip_c / self.snr_floor
        return float(min(sigma, sigma_max))


# Section 4 - Tunable configuration (to bind Streamlit widgets to)
@dataclass
class AugmentationConfig:
    """Every tunable parameter of the augmentation pipeline."""

    # Peak window (used by the noise model and the SNR audit)
    E_peak_start: float = -0.55
    E_peak_end: float = -0.25

    # Interpolation between anchor curves
    use_pchip: bool = True

    # Instrument noise
    use_lw_noise: bool = True
    noise_sigma_const_uA: float = 0.001  # used only when use_lw_noise=False
    snr_floor: float = 3.0

    # Concentration-scaled polynomial baseline distortion
    enable_baseline: bool = True
    baseline_amp_max_uA: float = 0.05
    baseline_scale_c_ref_uM: float = 5.0

    # Horizontal potential drift (electrode conditioning)
    enable_drift: bool = True
    potential_drift_sigma_low_V: float = 0.005
    potential_drift_sigma_high_V: float = 0.010

    # Stratified concentration sampling
    n_total_synthetic: int = 300
    stratified: bool = True  # False -> flat per-segment allocation (no log weighting)
    low_conc_threshold_uM: float = 2.5
    low_conc_boost: float = 2.0

    # Reproducibility (reseeds the global numpy RNG at the start of generate())
    rng_seed: int = GLOBAL_RANDOM_SEED


# Section 5 - Single-signal augmentation pipeline
class SignalAugmentor:
    """Applies the 4 augmentation ingredients: PCHIP blend, L&W noise, baseline bump, drift."""

    def __init__(self, calibration: CalibrationDataset, ip_spline: PeakCurrentSpline,
                 noise_model: LongWinefordnerNoiseModel, config: AugmentationConfig):
        self.calibration = calibration
        self.ip_spline = ip_spline
        self.noise_model = noise_model
        self.config = config
        self._anchors = calibration.anchor_concentrations_uM

    def get_base_pair(self, c_target: float):
        """Return (c_low, I_low, c_high, I_high) bracketing c_target."""
        c_target = float(np.clip(c_target, self._anchors[0], self._anchors[-1]))
        for i in range(len(self._anchors) - 1):
            c_lo, c_hi = self._anchors[i], self._anchors[i + 1]
            if c_lo <= c_target <= c_hi:
                return c_lo, self.calibration.base_curve(c_lo), c_hi, self.calibration.base_curve(c_hi)
        return (self._anchors[-2], self.calibration.base_curve(self._anchors[-2]),
                self._anchors[-1], self.calibration.base_curve(self._anchors[-1]))

    def interpolate_base_curve(self, c_target: float, c_low: float, I_low: np.ndarray,
                                c_high: float, I_high: np.ndarray) -> np.ndarray:
        """Blend the two bracketing anchor curves, weighted by a PCHIP-corrected alpha."""
        if c_high == c_low:
            return I_low.copy()
        if self.config.use_pchip:
            Ip_low, Ip_high, Ip_target = self.ip_spline(c_low), self.ip_spline(c_high), self.ip_spline(c_target)
            denom = Ip_high - Ip_low
            alpha = (Ip_target - Ip_low) / denom if abs(denom) > 1e-9 else (c_target - c_low) / (c_high - c_low)
        else:
            alpha = (c_target - c_low) / (c_high - c_low)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return (1.0 - alpha) * I_low + alpha * I_high

    def apply_position_dependent_noise(self, signal: np.ndarray, c_target: float) -> np.ndarray:
        """sigma_LW(c) inside the peak window, sigma_abs (instrument floor) outside it."""
        cfg = self.config
        E = self.calibration.potential_grid_V
        sigma_peak = self.noise_model.sigma(c_target) if cfg.use_lw_noise else cfg.noise_sigma_const_uA
        if sigma_peak <= 0:
            return signal
        sigma_floor = self.noise_model.sigma_abs_uA if cfg.use_lw_noise else cfg.noise_sigma_const_uA
        peak_mask = (E >= cfg.E_peak_start) & (E <= cfg.E_peak_end)
        noise = np.zeros_like(signal)
        noise[peak_mask] = np.random.normal(0.0, sigma_peak, int(peak_mask.sum()))
        noise[~peak_mask] = np.random.normal(0.0, sigma_floor, int((~peak_mask).sum()))
        return signal + noise

    def apply_polynomial_baseline_distortion(self, signal: np.ndarray, c_target: float) -> np.ndarray:
        """Smooth even-symmetric bump; amplitude scales with concentration via tanh(c / c_ref)."""
        cfg = self.config
        E = self.calibration.potential_grid_V
        scale = float(np.tanh(c_target / cfg.baseline_scale_c_ref_uM))
        amp = np.random.uniform(-cfg.baseline_amp_max_uA, cfg.baseline_amp_max_uA) * scale
        E_norm = (E - E.mean()) / (E.max() - E.min())
        distortion = amp * (E_norm ** 2 - E_norm ** 4)
        return signal + distortion

    def apply_horizontal_potential_drift(self, signal: np.ndarray) -> np.ndarray:
        """Shift the signal by delta_E ~ N(0, sigma^2), sigma ~ U(sigma_low, sigma_high)."""
        cfg = self.config
        E = self.calibration.potential_grid_V
        sigma = np.random.uniform(cfg.potential_drift_sigma_low_V, cfg.potential_drift_sigma_high_V)
        delta_E = np.random.normal(0.0, sigma)
        return np.interp(E, E + delta_E, signal, left=signal[0], right=signal[-1])

    def generate_signal(self, c_target: float) -> np.ndarray:
        """Full pipeline: PCHIP interpolation -> L&W noise -> scaled baseline -> drift."""
        c_lo, I_lo, c_hi, I_hi = self.get_base_pair(c_target)
        I = self.interpolate_base_curve(c_target, c_lo, I_lo, c_hi, I_hi)
        I = self.apply_position_dependent_noise(I, c_target)
        if self.config.enable_baseline:
            I = self.apply_polynomial_baseline_distortion(I, c_target)
        if self.config.enable_drift:
            I = self.apply_horizontal_potential_drift(I)
        return I


# Section 6 - Stratified concentration sampling
class StratifiedConcentrationSampler:
    """Log-uniform per-segment allocation with optional low-concentration oversampling."""

    def __init__(self, anchor_concentrations_uM: Sequence[float], config: AugmentationConfig):
        self.anchor_concentrations_uM = list(anchor_concentrations_uM)
        self.config = config

    def segment_counts(self) -> np.ndarray:
        cfg = self.config
        anchors = self.anchor_concentrations_uM
        if not cfg.stratified:
            return self._uniform_allocation(anchors, cfg.n_total_synthetic)

        weights = []
        for i in range(len(anchors) - 1):
            c_lo, c_hi = anchors[i], anchors[i + 1]
            weight = np.log(c_hi / c_lo)  # equal coverage per log-decade
            if (c_lo + c_hi) / 2.0 < cfg.low_conc_threshold_uM:
                weight *= cfg.low_conc_boost
            weights.append(weight)

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        counts = np.round(weights * cfg.n_total_synthetic).astype(int)
        diff = cfg.n_total_synthetic - counts.sum()
        if diff > 0:
            counts[np.argmax(weights)] += diff
        elif diff < 0:
            counts[np.argmin(weights[counts > 0])] -= abs(diff)
        return counts

    @staticmethod
    def _uniform_allocation(anchors: Sequence[float], n_total: int) -> np.ndarray:
        n_seg = len(anchors) - 1
        counts = np.full(n_seg, n_total // n_seg, dtype=int)
        counts[:n_total - counts.sum()] += 1
        return counts

    def sample_concentrations(self) -> np.ndarray:
        """Draw c ~ exp(U(ln c_lo, ln c_hi)) within every segment (equal density per decade)."""
        anchors = self.anchor_concentrations_uM
        counts = self.segment_counts()
        targets = []
        for i, n_seg in enumerate(counts):
            if n_seg == 0:
                continue
            c_lo, c_hi = anchors[i], anchors[i + 1]
            log_targets = np.random.uniform(np.log(c_lo), np.log(c_hi), size=n_seg)
            targets.append(np.exp(log_targets))
        return np.concatenate(targets) if targets else np.array([])


# Section 7 - Generated dataset container
@dataclass
class AugmentedDataset:
    """Sorted (concentration, signal) pairs produced by `FullRangeDataAugmentor.generate()`."""

    potential_grid_V: np.ndarray
    concentrations_uM: np.ndarray
    signals_uA: np.ndarray  # shape (n_signals, n_points)

    def __len__(self) -> int:
        return len(self.concentrations_uM)

    def to_raw_dataframe(self) -> pd.DataFrame:
        """Wide format matching raw/raw_signals_*.csv: concentration, I_0..I_{n-1}."""
        cols = [f'I_{k}' for k in range(self.signals_uA.shape[1])]
        df = pd.DataFrame(self.signals_uA, columns=cols)
        df.insert(0, 'concentration', self.concentrations_uM)
        return df

    def save_raw_csv(self, path: str) -> str:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.to_raw_dataframe().to_csv(path, index=False)
        return path


# Section 8 - Feature vectorization (core / extended / experimental suites)
FEATURE_COLUMNS_BY_SUITE = {
    'core': ['peak_current', 'peak_potential', 'peak_AUC', 'peak_FWHM'],
    'extended': ['peak_current', 'peak_potential', 'peak_AUC', 'peak_FWHM',
                 'pca1_comp', 'first_derivative_max', 'second_derivative_min'],
    'experimental': ['peak_current', 'peak_potential', 'peak_AUC', 'peak_FWHM',
                      'pca1_comp', 'first_derivative_max', 'second_derivative_min',
                      'left_slope', 'right_slope', 'asymetry', 'peak_sharpness',
                      'peak_compactness', 'current_variance', 'peak_skewness',
                      'peak_kurtosis', 'tchebichef_curve_moments', 'mean_peak',
                      'signal_entropy', 'spectral_entropy', 'fft_power',
                      'pca2_comp', 'pca3_comp', 'wavelet_energy'],
}


class FeatureVectorizer:
    """Wraps `Signal` to build tidy feature DataFrames for real and synthetic batches."""

    def __init__(self, potential_grid_V: np.ndarray, blank_baseline_uA: np.ndarray):
        self.potential_grid_V = np.asarray(potential_grid_V, dtype=float)
        self.blank_baseline_uA = np.asarray(blank_baseline_uA, dtype=float)
        Signal.set_common_potential_E(self.potential_grid_V)

    @staticmethod
    def vectorize(sig_obj: Signal, suite: str = 'core') -> list:
        vec = []
        if suite in ('core', 'extended', 'experimental'):
            vec += [sig_obj.get_peak_current_value(), sig_obj.get_peak_potential_value(),
                    sig_obj.get_peak_auc(), sig_obj.get_peak_fwhm()]
        if suite in ('extended', 'experimental'):
            vec += [sig_obj.get_pca1_comp(), sig_obj.get_first_derivative_max(),
                    sig_obj.get_second_derivative_min()]
        if suite == 'experimental':
            vec += [
                sig_obj.get_left_slope(), sig_obj.get_right_slope(),
                sig_obj.get_asymetry(), sig_obj.get_peak_sharpness(),
                sig_obj.get_peak_compactness(), sig_obj.get_current_variance(),
                sig_obj.get_peak_skewness(), sig_obj.get_peak_kurtosis(),
                sig_obj.get_tchebichef_curve_moments(), sig_obj.get_mean_peak(),
                sig_obj.get_signal_entropy(), sig_obj.get_spectral_entropy(),
                sig_obj.get_fft_power(),
                sig_obj.get_pca2_comp(), sig_obj.get_pca3_comp(),
                sig_obj.get_wavelet_energy(),
            ]
        return vec

    def real_feature_dataframe(self, calibration: CalibrationDataset, suite: str = 'core') -> pd.DataFrame:
        Signal.set_common_baseline_I(self.blank_baseline_uA)
        rows, cols_names = [], FEATURE_COLUMNS_BY_SUITE[suite] + ['concentration']
        for col_idx in range(calibration.raw_signal_matrix_uA.shape[1]):
            try:
                sig = Signal(calibration.raw_signal_matrix_uA[:, col_idx])
                rows.append(self.vectorize(sig, suite) + [calibration.concentrations_uM[col_idx]])
            except Exception:
                continue
        return pd.DataFrame(rows, columns=cols_names)

    def synthetic_feature_dataframe(self, dataset: AugmentedDataset, suite: str = 'core') -> pd.DataFrame:
        Signal.set_common_baseline_I(np.array([]))  # baseline already removed in the anchor curves
        rows, cols_names = [], FEATURE_COLUMNS_BY_SUITE[suite] + ['concentration']
        try:
            for I, c in zip(dataset.signals_uA, dataset.concentrations_uM):
                try:
                    sig = Signal(I)
                    rows.append(self.vectorize(sig, suite) + [float(c)])
                except Exception:
                    continue
        finally:
            Signal.set_common_baseline_I(self.blank_baseline_uA)  # restore for subsequent real-batch calls
        return pd.DataFrame(rows, columns=cols_names)


# Section 9 - Top-level facade (the Streamlit entry point)
class FullRangeDataAugmentor:
    """Wires calibration data, the Ip spline and the noise model into one `generate()` call.

    Build once per calibration dataset; call `generate()` with a fresh
    `AugmentationConfig` for every user-triggered regeneration.
    """

    def __init__(self, calibration: CalibrationDataset, snr_floor: float = 3.0):
        self.calibration = calibration
        self.ip_spline = PeakCurrentSpline(calibration)
        self.noise_model = LongWinefordnerNoiseModel.fit(calibration, self.ip_spline, snr_floor=snr_floor)

    @classmethod
    def from_excel(cls, path: str, concentrations_uM: Sequence[float], **kwargs) -> 'FullRangeDataAugmentor':
        calibration = CalibrationDataset.from_excel(path, concentrations_uM=concentrations_uM)
        return cls(calibration, **kwargs)

    def generate(self, config: Optional[AugmentationConfig] = None) -> AugmentedDataset:
        config = config or AugmentationConfig()
        self.noise_model.snr_floor = config.snr_floor
        np.random.seed(config.rng_seed)

        sampler = StratifiedConcentrationSampler(self.calibration.anchor_concentrations_uM, config)
        signal_augmentor = SignalAugmentor(self.calibration, self.ip_spline, self.noise_model, config)

        concentrations = sampler.sample_concentrations()
        signals = [signal_augmentor.generate_signal(c) for c in concentrations]

        order = np.argsort(concentrations)
        concentrations = concentrations[order]
        signals = np.array([signals[i] for i in order])

        return AugmentedDataset(self.calibration.potential_grid_V, concentrations, signals)

    def feature_vectorizer(self) -> FeatureVectorizer:
        return FeatureVectorizer(self.calibration.potential_grid_V, self.calibration.blank_baseline_uA)
