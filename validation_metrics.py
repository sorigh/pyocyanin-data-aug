from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import ot  # Python Optimal Transport
    _OT_AVAILABLE = True
except ImportError:
    _OT_AVAILABLE = False
    warnings.warn("POT (ot) not installed - SWD metric will be skipped.", ImportWarning)

# Module-level constants (match notebook defaults)
FEATURE_KEYS    = ['Ip', 'Ep', 'FWHM', 'AUC', 'skewness', 'asymmetry']
SHAPE_KEYS      = ['Ep', 'FWHM', 'skewness', 'asymmetry']   # Ip/AUC excluded from PFF/DS
DS_FEATURE_KEYS = ['Ep', 'FWHM', 'skewness', 'asymmetry']


# Section 1 - Core signal utilities
def extract_peak_features(
    E: np.ndarray,
    I: np.ndarray,
    E_win_start: float = -0.55,
    E_win_end:   float = -0.25,
) -> Optional[dict]:
    """
    Extract scalar peak features from a single linear voltammogram signal.

    Parameters
    ----------
    E            : potential array (V), shape (n_points,)
    I            : current array (µA), shape (n_points,)
    E_win_start  : left boundary of the peak search window (V)
    E_win_end    : right boundary of the peak search window (V)

    Returns
    -------
    dict with keys: Ip, Ep, FWHM, AUC, skewness, asymmetry, left_slope,
                    right_slope, peak_window
    Returns None if the peak window contains fewer than 3 points.

    Notes
    -----
    FIX-9: I_weights = I_peak - I_peak.min() ensures non-negative weights
    for the weighted-moment skewness calculation. A variance floor of 1e-8 V²
    prevents NaN propagation for very narrow windows.
    """
    mask = (E >= E_win_start) & (E <= E_win_end)
    idxs = np.where(mask)[0]
    if len(idxs) < 3:
        return None

    start_idx, end_idx = idxs[0], idxs[-1]
    features: dict = {}

    # Peak apex
    Ip_idx = np.argmax(I[start_idx:end_idx]) + start_idx
    Ip     = I[Ip_idx]
    Ep     = E[Ip_idx]

    # Refined peak boundaries (local minima search)
    left_idx, Imin_l = Ip_idx, Ip
    for i in range(Ip_idx, start_idx - 1, -1):
        if I[i] < Imin_l:
            Imin_l, left_idx = I[i], i

    right_idx, Imin_r = Ip_idx, Ip
    for i in range(Ip_idx, end_idx + 1):
        if I[i] < Imin_r:
            Imin_r, right_idx = I[i], i

    I_peak = I[left_idx:right_idx + 1]
    E_peak = E[left_idx:right_idx + 1]

    features['Ip']          = float(Ip)
    features['Ep']          = float(Ep)
    features['peak_window'] = I_peak

    # FWHM
    half_max = Ip * 0.5
    above = np.where(I_peak >= half_max)[0]
    features['FWHM'] = float(E_peak[above[-1]] - E_peak[above[0]]) if len(above) >= 2 else np.nan

    # AUC (trapezoidal integration)
    features['AUC'] = float(np.trapezoid(I_peak, E_peak))

    # Skewness - FIX-9: shift weights to non-negative
    I_weights = I_peak - I_peak.min()
    total_I   = np.sum(I_weights)
    if total_I > 1e-12:
        mu    = np.sum(E_peak * I_weights) / total_I
        var   = np.sum(I_weights * (E_peak - mu) ** 2) / total_I
        sigma = np.sqrt(max(var, 1e-8))
        features['skewness'] = float(
            np.sum(I_weights * (E_peak - mu) ** 3) / (total_I * sigma ** 3)
        )
    else:
        features['skewness'] = np.nan

    # Slope asymmetry
    try:
        alpha = 0.1
        thr   = alpha * Ip
        l_above = np.where(I[left_idx:Ip_idx] >= thr)[0]
        r_above = np.where(I[Ip_idx:right_idx] >= thr)[0]
        if len(l_above) >= 2 and len(r_above) >= 2:
            l_start = left_idx + l_above[0]
            E_l = E[l_start:Ip_idx];     I_l = I[l_start:Ip_idx]
            E_r = E[Ip_idx:Ip_idx + r_above[-1]]; I_r = I[Ip_idx:Ip_idx + r_above[-1]]
            if len(E_l) >= 2 and len(E_r) >= 2:
                left_slope  = np.polyfit(E_l, I_l, 1)[0]
                right_slope = np.polyfit(E_r, I_r, 1)[0]
                features['left_slope']  = float(left_slope)
                features['right_slope'] = float(right_slope)
                features['asymmetry']   = float(abs(left_slope) / (abs(right_slope) + 1e-12))
            else:
                features['left_slope'] = features['right_slope'] = features['asymmetry'] = np.nan
        else:
            features['left_slope'] = features['right_slope'] = features['asymmetry'] = np.nan
    except Exception:
        features['left_slope'] = features['right_slope'] = features['asymmetry'] = np.nan

    return features


def build_feature_matrix(
    X: np.ndarray,
    E: np.ndarray,
    E_win_start: float = -0.55,
    E_win_end:   float = -0.25,
) -> pd.DataFrame:
    """
    Extract peak features for every signal in X.

    Returns a DataFrame with columns = FEATURE_KEYS.
    Infinite values are replaced with NaN.
    """
    rows = []
    for sig in X:
        f = extract_peak_features(E, sig, E_win_start, E_win_end)
        if f is not None:
            rows.append({k: f[k] for k in FEATURE_KEYS})
    df = pd.DataFrame(rows)
    return df.replace([np.inf, -np.inf], np.nan)



# Section 2 - Concentration-stratification helpers
def assign_nearest_log_class(
    y_arr: np.ndarray,
    real_classes: np.ndarray,
) -> np.ndarray:
    """
    Assign each synthetic concentration to the nearest real calibration class
    on a logarithmic scale (FIX-2: handles log-uniform synthetic sampling).

    Parameters:
    y_arr        : array of synthetic concentrations (µM)
    real_classes : array of real anchor concentrations (µM)

    Returns:
    Array of same shape as y_arr, dtype=float, containing assigned classes.
    """
    classes  = np.asarray(real_classes, dtype=float)
    log_cls  = np.log(classes)
    assigned = np.empty(len(y_arr), dtype=float)
    for i, c in enumerate(y_arr):
        assigned[i] = classes[np.argmin(np.abs(log_cls - np.log(max(float(c), 1e-12))))]
    return assigned


def build_class_normalised_matrices(
    X_real:  np.ndarray,
    y_real:  np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    E:       np.ndarray,
    E_win_start: float = -0.55,
    E_win_end:   float = -0.25,
) -> tuple[dict, dict, dict]:
    """
    Build per-class Ip-normalised peak-window matrices (FIX-5).

    Returns
    X_r_norm   : {conc: (n_real, n_peak_pts)}  real signals / class-mean-Ip
    X_s_norm   : {conc: (n_synth, n_peak_pts)} synth signals / same class-mean-Ip
    ip_by_class: {conc: float}  class-mean Ip of real signals
    """
    peak_mask      = (E >= E_win_start) & (E <= E_win_end)
    real_classes   = np.unique(y_real)
    y_synth_binned = assign_nearest_log_class(y_synth, real_classes)

    X_r_norm:    dict = {}
    X_s_norm:    dict = {}
    ip_by_class: dict = {}

    for c in sorted(real_classes):
        mask_r = y_real == c
        mask_s = y_synth_binned == c
        if not mask_r.any() or not mask_s.any():
            continue
        Xr_pk = X_real[mask_r][:, peak_mask]
        Xs_pk = X_synth[mask_s][:, peak_mask]
        ip_mean = Xr_pk.max(axis=1).mean()
        ip_by_class[c]  = ip_mean
        X_r_norm[c]     = Xr_pk / (ip_mean + 1e-10)
        X_s_norm[c]     = Xs_pk / (ip_mean + 1e-10)

    return X_r_norm, X_s_norm, ip_by_class


def normalize_rowwise(X: np.ndarray) -> np.ndarray:
    """Min-max normalisation per row (used for ACF / PCA / t-SNE)."""
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    return (X - mins) / (maxs - mins + 1e-10)



# Section 3 - Tier 1 statistical metrics
def compute_jsd(
    p_hist: np.ndarray, 
    q_hist: np.ndarray, 
    eps: float = 1e-10
    ) -> float:
    """
    Jensen-Shannon Divergence (base-2) between two empirical histograms.

    JSD(P||Q) = 0.5·KL(P||M) + 0.5·KL(Q||M),  M = (P + Q) / 2
    Returns a value in [0, 1].
    """
    p = np.array(p_hist, dtype=float) + eps
    q = np.array(q_hist, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def compute_per_class_jsd(
    X_real:  np.ndarray,
    y_real:  np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    E:       np.ndarray,
    E_win_start: float = -0.55,
    E_win_end:   float = -0.25,
    n_bins:  int = 30,
) -> dict:
    """
    Per-concentration-class JSD on Ip-normalised peak-window current values.

    FIX-A: bin edges are defined on the REAL distribution support only;
    synthetic values are clipped to that range to avoid artefact JSD inflation.

    Returns
    dict {conc: jsd_value}  - NaN for classes with no synthetic bin.
    """
    peak_mask      = (E >= E_win_start) & (E <= E_win_end)
    real_classes   = np.unique(y_real)
    y_synth_binned = assign_nearest_log_class(y_synth, real_classes)
    jsd_per_class  = {}

    for conc in real_classes:
        Xr_win = X_real[y_real == conc][:, peak_mask]
        Xs_win = X_synth[y_synth_binned == conc][:, peak_mask]
        if len(Xs_win) == 0:
            jsd_per_class[conc] = np.nan
            continue
        ip_mean    = Xr_win.max(axis=1).mean()
        Xr_n       = Xr_win / (ip_mean + 1e-10)
        Xs_n       = Xs_win / (ip_mean + 1e-10)
        I_real     = Xr_n.ravel()
        I_synth    = Xs_n.ravel()
        n_bins_cls = int(np.clip(n_bins, 8, max(8, len(I_real) // 8)))
        r_min, r_max = I_real.min(), I_real.max()
        bins = np.linspace(r_min, r_max, n_bins_cls + 1)
        p_hist, _ = np.histogram(I_real, bins=bins)
        q_hist, _ = np.histogram(np.clip(I_synth, r_min, r_max), bins=bins)
        jsd_per_class[conc] = compute_jsd(p_hist, q_hist)

    return jsd_per_class


def compute_mmd_squared(
    X_real_norm:  np.ndarray,
    X_synth_norm: np.ndarray,
    gamma:        Optional[float] = None,
    max_n:        int = 200,
) -> tuple[float, float]:
    """
    Unbiased MMD² with RBF kernel (median heuristic gamma by default).

    Returns
    (mmd2, gamma_used)
    """
    X = X_real_norm.astype(np.float32)
    Y = X_synth_norm.astype(np.float32)
    rng = np.random.default_rng(42)
    if len(X) > max_n: X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n: Y = Y[rng.choice(len(Y), max_n, replace=False)]
    n, m = len(X), len(Y)
    if gamma is None:
        XY    = np.vstack([X, Y])
        dists = cdist(XY, XY, 'euclidean')
        gamma = 1.0 / (2.0 * np.median(dists[dists > 0]) ** 2 + 1e-10)
    K_xx = rbf_kernel(X, X, gamma=gamma); np.fill_diagonal(K_xx, 0)
    K_yy = rbf_kernel(Y, Y, gamma=gamma); np.fill_diagonal(K_yy, 0)
    K_xy = rbf_kernel(X, Y, gamma=gamma)
    mmd2 = K_xx.sum() / (n * (n - 1)) - 2 * K_xy.mean() + K_yy.sum() / (m * (m - 1))
    return float(mmd2), float(gamma)


def compute_swd(
    X_real_norm:  np.ndarray,
    X_synth_norm: np.ndarray,
    n_projections: int = 500,
    seed:         int = 42,
    max_n:        int = 300,
) -> float:
    """
    Sliced Wasserstein Distance (SWD) - Bonneel et al. (2015).

    Requires the POT library. Returns NaN if unavailable.
    """
    if not _OT_AVAILABLE:
        return float('nan')
    X = X_real_norm.astype(np.float64)
    Y = X_synth_norm.astype(np.float64)
    rng = np.random.default_rng(seed)
    if len(X) > max_n: X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n: Y = Y[rng.choice(len(Y), max_n, replace=False)]
    return float(ot.sliced_wasserstein_distance(X, Y, n_projections=n_projections, seed=seed))


def compute_stratified_ks(
    X_r_by_class: dict,
    X_s_by_class: dict,
) -> tuple[dict, float, float]:
    """
    Concentration-stratified point-wise KS test (FIX-5).

    For each class, runs KS at every voltage step in the normalised peak window
    and reports the fraction of steps where p > 0.05.

    Returns
    per_class : {c: {'frac_pass', 'ks_stats', 'p_values'}}
    mean_frac : mean pass-fraction across classes
    min_frac  : worst-class pass-fraction
    """
    per_class = {}
    for c, Xr_n in sorted(X_r_by_class.items()):
        Xs_n = X_s_by_class.get(c)
        if Xs_n is None or len(Xs_n) < 2 or len(Xr_n) < 2:
            continue
        n_pts    = Xr_n.shape[1]
        ks_stats = np.zeros(n_pts)
        p_values = np.zeros(n_pts)
        for j in range(n_pts):
            stat, pval    = ks_2samp(Xr_n[:, j], Xs_n[:, j])
            ks_stats[j]   = stat
            p_values[j]   = pval
        per_class[c] = {
            'frac_pass': float(np.mean(p_values > 0.05)),
            'ks_stats':  ks_stats,
            'p_values':  p_values,
        }
    fracs     = [v['frac_pass'] for v in per_class.values()]
    mean_frac = float(np.mean(fracs)) if fracs else 0.0
    min_frac  = float(np.min(fracs))  if fracs else 0.0
    return per_class, mean_frac, min_frac


def compute_stratified_swd(
    X_r_by_class:  dict,
    X_s_by_class:  dict,
    n_projections: int = 300,
    seed:          int = 42,
) -> tuple[dict, float, float]:
    """
    Concentration-stratified SWD (FIX-5).

    Returns
    per_class_swd, mean_swd, max_swd
    """
    per_class_swd = {}
    for c, Xr_n in sorted(X_r_by_class.items()):
        Xs_n = X_s_by_class.get(c)
        if Xs_n is None or len(Xs_n) < 2 or len(Xr_n) < 2:
            continue
        per_class_swd[c] = compute_swd(Xr_n, Xs_n, n_projections=n_projections, seed=seed)
    vals     = list(per_class_swd.values())
    mean_swd = float(np.nanmean(vals)) if vals else float('nan')
    max_swd  = float(np.nanmax(vals))  if vals else float('nan')
    return per_class_swd, mean_swd, max_swd


def _estimate_wasserstein_null_floor(
    big_pool: np.ndarray, 
    n_small: int, 
    n_big: int, 
    n_boot: int, 
    seed: int
) -> float:
    """
    Estimate the null floor of Wasserstein distance via bootstrap resampling.
    Reasoning:
    When comparing a small real dataset (e.g. 2 points) to a much larger 
    synthetic dataset (e.g. 40 points), the linear interpolation of the small sample
    has a over simplified "geometry" compared to the larger sample. (a line compared to a curve).
     
    This causes an artificially high Wasserstein distance, even if both sets 
    describe the same baisc distribution.
    
    To correct this, I repeatedly resample (extract a small sample from the big dataset,
    interpolate it onto the larger grid, and measure the distance to the full distribution (full dataset)) 
    and take the median of the resulting distances.
    """
    if n_big < 8 or n_boot <= 0:
        return 0.0

    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_boot)
    big_grid = np.linspace(0, 1, n_big)
    small_grid = np.linspace(0, 1, n_small)

    for i in range(n_boot):
        # Extract a bootstrap sample of size n_small from the big_pool
        resample = np.sort(rng.choice(big_pool, size=n_small, replace=True))
        
        # Interpolate the resampled points onto the larger grid
        interp_b = np.interp(big_grid, small_grid, resample)
        
        # Measure the distance (which would be 0 in ideal conditions, but isn't in practice)
        boot_vals[i] = np.mean(np.abs(np.sort(big_pool) - interp_b))

    return float(np.median(boot_vals))

# Section 4 - Tier 2 physics-anchored metrics
def compute_normalised_wasserstein(
    p_arr: np.ndarray,
    q_arr: np.ndarray,
    seed: int = 42,
    ) -> float:
    """
    Normalised Wasserstein-1 distance (replaces histogram Hellinger;
    interpolate the SMALLER sample onto the LARGER sample's quantile
    grid, regardless of argument order. The old code always built the grid
    from p_arr, which degenerated to a 2-point min/max comparison whenever
    the first-passed array (often n_real=2-6) was the smaller one.

    Valid even at n = 3 (exact sort-based W1 vs. Hellinger which needs n ≥ 20).
    Normalised by pooled standard deviation -> dimensionless, comparable across
    feature scales.
    """
    debias_n_boot = 500
    p_sorted = np.sort(p_arr.astype(float))
    q_sorted = np.sort(q_arr.astype(float))
    n_p, n_q = len(p_sorted), len(q_sorted)
    if n_p != n_q:
        if n_p >= n_q:
            big, big_pts = p_sorted, np.linspace(0, 1, n_p)
            small, small_pts = q_sorted, np.linspace(0, 1, n_q)
        else:
            big, big_pts = q_sorted, np.linspace(0, 1, n_q)
            small, small_pts = p_sorted, np.linspace(0, 1, n_p)
        small_interp = np.interp(big_pts, small_pts, small)
        w1_raw = float(np.mean(np.abs(big - small_interp)))

        n_small, n_big = min(n_p, n_q), max(n_p, n_q)
        big_pool = q_sorted if n_q >= n_p else p_sorted
        
        null_floor = _estimate_wasserstein_null_floor(big_pool, n_small, n_big, debias_n_boot, seed)
        w1 = max(0.0, w1_raw - null_floor)
    else:
        w1 = float(np.mean(np.abs(p_sorted - q_sorted)))
    pooled_std = float(np.std(np.concatenate([p_arr, q_arr]), ddof=1))
    return w1 / (pooled_std + 1e-12)


def compute_pff_stratified(
    feat_real_df:  pd.DataFrame,
    y_real:        np.ndarray,
    feat_synth_df: pd.DataFrame,
    y_synth:       np.ndarray,
    shape_keys:    Optional[list] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Concentration-stratified Peak Feature Fidelity (PFF).

    Tests {Ep, FWHM, skewness, asymmetry} using per-class KS test.
    Ip and AUC excluded (FIX-10/FIX-I: carry concentration label).

    Returns
    -------
    pff_df      : per-feature-per-class test results DataFrame
    class_pass  : {conc: True/False/None}  - None = skipped (too few samples)
    """
    if shape_keys is None:
        shape_keys = SHAPE_KEYS

    real_classes     = np.unique(y_real)
    feat_r           = feat_real_df.reset_index(drop=True)
    feat_s           = feat_synth_df.reset_index(drop=True)
    y_r              = y_real[:len(feat_r)]
    y_s              = y_synth[:len(feat_s)]
    y_s_binned       = assign_nearest_log_class(y_s, real_classes)

    rows        = []
    class_pass  = {}

    for conc in sorted(real_classes):
        mask_r = np.where(y_r == conc)[0]
        mask_s = np.where(y_s_binned == conc)[0]
        if len(mask_r) < 2 or len(mask_s) < 2:
            class_pass[conc] = None
            continue
        class_passes = []
        for feat in shape_keys:
            r_raw = feat_r.iloc[mask_r][feat].replace([np.inf, -np.inf], np.nan).dropna().values
            s_raw = feat_s.iloc[mask_s][feat].replace([np.inf, -np.inf], np.nan).dropna().values
            if len(r_raw) < 2 or len(s_raw) < 2:
                rows.append({'feature': feat, 'class_uM': conc,
                             'KS_stat': np.nan, 'p_value': np.nan,
                             'Wasserstein': np.nan, 'KS_pass': False})
                class_passes.append(False)
                continue
            ks_stat, p_val = ks_2samp(r_raw, s_raw)
            wass           = compute_normalised_wasserstein(r_raw, s_raw)
            rows.append({'feature': feat, 'class_uM': conc,
                         'KS_stat': round(ks_stat, 5), 'p_value': round(p_val, 5),
                         'Wasserstein': round(wass, 5), 'KS_pass': p_val > 0.05})
            class_passes.append(p_val > 0.05)
        class_pass[conc] = all(class_passes) if class_passes else False

    return pd.DataFrame(rows), class_pass


def compute_randles_sevcik_conformance(
    feat_real_df:  pd.DataFrame,
    feat_synth_df: pd.DataFrame,
    y_real:        np.ndarray,
    y_synth:       np.ndarray,
) -> dict:
    """
    Validates Randles-Ševčík Ip ∝ C linearity for synthetic signals.

    Fits β via regression through the origin on real {C, Ip} pairs, then
    compares synthetic residuals to real residuals via a two-sample KS test.

    Pass criteria (FIX-3): R²_synth ≥ 0.90 AND KS p > 0.05 AND residual ratio < 1.5.
    """
    Ip_r = feat_real_df['Ip'].values
    Ip_s = feat_synth_df['Ip'].values
    C_r  = y_real
    C_s  = y_synth[:len(Ip_s)]

    mask_r = ~np.isnan(Ip_r); mask_s = ~np.isnan(Ip_s)
    Ip_r = Ip_r[mask_r]; C_r = C_r[mask_r]
    Ip_s = Ip_s[mask_s]; C_s = C_s[mask_s]

    beta      = np.dot(C_r, Ip_r) / np.dot(C_r, C_r)
    Ip_r_pred = beta * C_r
    Ip_s_pred = beta * C_s

    R2_real  = 1 - np.sum((Ip_r - Ip_r_pred) ** 2) / (np.sum((Ip_r - np.mean(Ip_r)) ** 2) + 1e-12)
    R2_synth = 1 - np.sum((Ip_s - Ip_s_pred) ** 2) / (np.sum((Ip_s - np.mean(Ip_s)) ** 2) + 1e-12)

    eps_real  = Ip_r - Ip_r_pred
    eps_synth = Ip_s - Ip_s_pred
    ks_stat, ks_pval  = ks_2samp(eps_real, eps_synth)
    mean_res_ratio    = np.mean(np.abs(eps_synth)) / (np.mean(np.abs(eps_real)) + 1e-12)

    log_C_r = np.log(C_r + 1e-9); log_C_s = np.log(C_s + 1e-9)
    beta_log, intercept_log = np.polyfit(log_C_r, Ip_r, 1)
    Ip_s_log_pred = beta_log * log_C_s + intercept_log
    R2_synth_log  = 1 - np.sum((Ip_s - Ip_s_log_pred) ** 2) / (np.sum((Ip_s - np.mean(Ip_s)) ** 2) + 1e-12)

    rs_pass = (R2_synth >= 0.90) and (ks_pval > 0.05) and (mean_res_ratio < 1.5)
    return {
        'beta': beta, 'R2_real': R2_real, 'R2_synth': R2_synth,
        'R2_synth_log': R2_synth_log, 'mean_residual_ratio': mean_res_ratio,
        'ks_stat': ks_stat, 'ks_pvalue': ks_pval, 'rs_pass': rs_pass,
        'eps_real': eps_real, 'eps_synth': eps_synth,
        'C_r': C_r, 'C_s': C_s, 'Ip_r': Ip_r, 'Ip_s': Ip_s,
    }


def compute_acf_vectors(
    X: np.ndarray, 
    max_lag: int = 20
    ) -> np.ndarray:
    """
    Sample autocorrelation functions r(τ) for each signal in X.

    r(τ) = [Σ_{t} (I_t - Ī)(I_{t+τ} - Ī)] / [Σ_t (I_t - Ī)²]

    Returns shape (n_signals, max_lag).
    """
    n, T = X.shape
    acfs = np.zeros((n, max_lag))
    for i in range(n):
        sig   = X[i] - X[i].mean()
        denom = np.dot(sig, sig)
        if denom == 0:
            continue
        for lag in range(1, max_lag + 1):
            acfs[i, lag - 1] = np.dot(sig[:T - lag], sig[lag:]) / denom
    return acfs


# Section 5 - Tier 3 utility metrics
def build_feature_matrix_array(
    feat_df: pd.DataFrame,
    y_arr:   np.ndarray,
    keys:    list = FEATURE_KEYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned (X_features, y) arrays with NaN rows removed."""
    df = feat_df[keys].copy()
    df['y'] = y_arr[:len(df)]
    df = df.dropna()
    return df[keys].values, df['y'].values


def run_tstr_trtr(
    X_feat_real:  np.ndarray,
    y_feat_real:  np.ndarray,
    X_feat_synth: np.ndarray,
    y_feat_synth: np.ndarray,
    alphas:       tuple = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict:
    """
    TSTR / TRTR in log-concentration space with 5-fold CV TRTR baseline.

    Pass criteria (FIX-11): log-RMSE ratio ≤ 1.20, MAPE ratio ≤ 1.30,
    log-R² ratio ≥ 0.80.
    """
    log_y_real  = np.log(np.clip(y_feat_real,  1e-6, None))
    log_y_synth = np.log(np.clip(y_feat_synth, 1e-6, None))

    sc_r  = StandardScaler()
    X_r_sc = sc_r.fit_transform(X_feat_real)

    kf = KFold(n_splits=min(5, len(X_feat_real) // 3), shuffle=True, random_state=42)
    y_kf_log  = cross_val_predict(RidgeCV(alphas=alphas), X_r_sc, log_y_real, cv=kf)
    y_kf_lin  = np.exp(y_kf_log)
    R2_trtr_kf   = r2_score(log_y_real, y_kf_log)
    rmse_trtr_kf = float(np.sqrt(mean_squared_error(log_y_real, y_kf_log)))
    mape_trtr    = float(np.mean(np.abs((y_feat_real - y_kf_lin) / (y_feat_real + 1e-9)))) * 100

    loo = LeaveOneOut()
    y_loo_log     = cross_val_predict(RidgeCV(alphas=alphas), X_r_sc, log_y_real, cv=loo)
    rmse_trtr_loo = float(np.sqrt(mean_squared_error(log_y_real, y_loo_log)))

    sc_s       = StandardScaler()
    X_s_sc     = sc_s.fit_transform(X_feat_synth)
    model_tstr = RidgeCV(alphas=alphas, cv=min(5, len(X_feat_synth))).fit(X_s_sc, log_y_synth)

    X_r_on_s     = sc_s.transform(X_feat_real)
    y_tstr_log   = model_tstr.predict(X_r_on_s)
    y_tstr_lin   = np.exp(y_tstr_log)
    R2_tstr_log  = r2_score(log_y_real, y_tstr_log)
    rmse_tstr_log = float(np.sqrt(mean_squared_error(log_y_real, y_tstr_log)))
    mape_tstr     = float(np.mean(np.abs((y_feat_real - y_tstr_lin) / (y_feat_real + 1e-9)))) * 100

    log_rmse_ratio = rmse_tstr_log / (rmse_trtr_kf + 1e-12)
    mape_ratio     = mape_tstr     / (mape_trtr    + 1e-12)
    r2_ratio_log   = R2_tstr_log   / (R2_trtr_kf   + 1e-12)
    tstr_pass      = (log_rmse_ratio <= 1.20) and (mape_ratio <= 1.30) and (r2_ratio_log >= 0.80)

    return {
        'R2_trtr_kf': R2_trtr_kf, 'rmse_trtr_kf': rmse_trtr_kf,
        'rmse_trtr_loo': rmse_trtr_loo, 'R2_tstr_log': R2_tstr_log,
        'rmse_tstr_log': rmse_tstr_log, 'rmse_tstr_lin': float(np.sqrt(mean_squared_error(y_feat_real, y_tstr_lin))),
        'rmse_trtr_lin': float(np.sqrt(mean_squared_error(y_feat_real, y_kf_lin))),
        'log_rmse_ratio': log_rmse_ratio, 'mape_ratio': mape_ratio,
        'r2_ratio_log': r2_ratio_log, 'mape_trtr': mape_trtr, 'mape_tstr': mape_tstr,
        'y_tstr_log': y_tstr_log, 'y_tstr_lin': y_tstr_lin,
        'y_kf_log': y_kf_log,    'y_kf_lin':   y_kf_lin,
        'log_y_real': log_y_real, 'tstr_pass': tstr_pass,
        'efficiency_ratio': log_rmse_ratio,
    }


def run_trts(
    X_feat_real:  np.ndarray,
    y_feat_real:  np.ndarray,
    X_feat_synth: np.ndarray,
    y_feat_synth: np.ndarray,
    alphas:       tuple = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict:
    """
    TRTS (Train on Real, Test on Synthetic).

    Detects out-of-distribution synthetic signals.
    Pass: TRTS_ratio ≤ 1.50 AND R2_trts ≥ 0.70.
    """
    sc_    = StandardScaler()
    X_r_sc = sc_.fit_transform(X_feat_real)
    X_s_sc = sc_.transform(X_feat_synth)
    model  = RidgeCV(alphas=alphas, cv=min(5, len(X_feat_real))).fit(X_r_sc, y_feat_real)
    y_trts = model.predict(X_s_sc)
    y_trtr = model.predict(X_r_sc)

    R2_trts   = r2_score(y_feat_synth, y_trts)
    rmse_trts = float(np.sqrt(mean_squared_error(y_feat_synth, y_trts)))
    rmse_trtr = float(np.sqrt(mean_squared_error(y_feat_real, y_trtr)))
    trts_ratio = rmse_trts / (rmse_trtr + 1e-12)

    return {
        'R2_trts': R2_trts, 'rmse_trts': rmse_trts,
        'trts_ratio': trts_ratio, 'y_trts': y_trts,
        'trts_pass': (trts_ratio <= 1.50) and (R2_trts >= 0.70),
    }


def run_augmentation_gain(
    X_feat_real:  np.ndarray,
    y_feat_real:  np.ndarray,
    X_feat_synth: np.ndarray,
    y_feat_synth: np.ndarray,
    ratios:       tuple = (0, 1, 2, 5, 10, 20),
    alphas:       tuple = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict:
    """
    Log-scale augmentation gain analysis with LOO evaluation (FIX-12 + FIX-C).

    Returns
    dict {ratio: {'log_rmse': float, 'r2_log': float}}
    """
    log_y_real  = np.log(np.clip(y_feat_real,  1e-6, None))
    log_y_synth = np.log(np.clip(y_feat_synth, 1e-6, None))
    N_real  = len(X_feat_real)
    loo     = LeaveOneOut()
    results = {}

    for ratio in ratios:
        loo_preds = np.zeros(N_real)
        for train_idx, test_idx in loo.split(X_feat_real):
            X_r_tr = X_feat_real[train_idx]
            y_r_tr = log_y_real[train_idx]
            if ratio == 0:
                X_train = X_r_tr; y_train = y_r_tr
            else:
                n_synth = min(ratio * len(train_idx), len(X_feat_synth))
                rng_    = np.random.RandomState(42 + int(train_idx[0]))
                idx_s   = rng_.choice(len(X_feat_synth), n_synth, replace=False)
                X_train = np.vstack([X_r_tr, X_feat_synth[idx_s]])
                y_train = np.concatenate([y_r_tr, log_y_synth[idx_s]])
            sc_     = StandardScaler()
            X_tr_sc = sc_.fit_transform(X_train)
            X_te_sc = sc_.transform(X_feat_real[test_idx])
            model   = RidgeCV(alphas=alphas, cv=min(5, max(3, len(X_train))))
            model.fit(X_tr_sc, y_train)
            loo_preds[test_idx] = model.predict(X_te_sc)
        results[ratio] = {
            'log_rmse': float(np.sqrt(mean_squared_error(log_y_real, loo_preds))),
            'r2_log':   float(r2_score(log_y_real, loo_preds)),
        }
    return results


# Section 6 - Tier 4 discriminative score
def compute_discriminative_score_stratified(
    feat_real_df:  pd.DataFrame,
    y_real:        np.ndarray,
    feat_synth_df: pd.DataFrame,
    y_synth:       np.ndarray,
    n_estimators:  int = 100,
    seed:          int = 42,
    min_n_real:    int = 3,
) -> dict:
    """
    Concentration-stratified Discriminative Score (FIX-8 + FIX-D).

    DS = 1 - |accuracy - 0.5|.  DS near 0.5 = indistinguishable (ideal).
    Bootstrap averaging for classes with n_real < 8 (FIX-D).
    Pass threshold: mean_DS ≥ 0.70 (TimeGAN standard, Yoon et al. 2019).
    Features: {Ep, FWHM, skewness, asymmetry} only (Ip/AUC excluded - FIX-8).
    """
    real_classes   = np.unique(y_real)
    y_s_binned     = assign_nearest_log_class(y_synth, real_classes)
    feat_r         = feat_real_df.reset_index(drop=True)
    feat_s         = feat_synth_df.reset_index(drop=True)

    ds_per_class   = {}
    acc_per_class  = {}
    fi_per_class   = {}
    rng            = np.random.default_rng(seed)
    DS_THR         = 0.70

    for conc in sorted(real_classes):
        mask_r = np.where(y_real == conc)[0]
        mask_s = np.where(y_s_binned == conc)[0]
        if len(mask_r) < min_n_real or len(mask_s) < 2:
            ds_per_class[conc] = acc_per_class[conc] = None
            continue

        Xr = feat_r.loc[feat_r.index[mask_r], DS_FEATURE_KEYS].fillna(0).values
        Xs_full = feat_s.loc[feat_s.index[mask_s], DS_FEATURE_KEYS].fillna(0).values
        n_r   = len(Xr)
        n_s   = min(n_r * 5, len(Xs_full))
        Xs    = Xs_full[rng.choice(len(Xs_full), n_s, replace=False)]

        use_bootstrap = n_r < 8
        if use_bootstrap:
            N_BOOT = 20
            accs   = []
            for bround in range(N_BOOT):
                b_seed = seed + bround + int(conc * 100)
                brng   = np.random.default_rng(b_seed)
                idx_r_b = brng.choice(n_r, n_r, replace=True)
                idx_s_b = brng.choice(len(Xs), n_r, replace=True)
                X_b = np.vstack([Xr[idx_r_b], Xs[idx_s_b]])
                y_b = np.array([1] * n_r + [0] * n_r)
                n_half = max(1, len(X_b) // 2)
                perm   = brng.permutation(len(X_b))
                X_tr = X_b[perm[:n_half]]; y_tr = y_b[perm[:n_half]]
                X_te = X_b[perm[n_half:]]; y_te = y_b[perm[n_half:]]
                if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
                    continue
                sc_b  = StandardScaler()
                clf_b = RandomForestClassifier(50, random_state=b_seed, class_weight='balanced')
                clf_b.fit(sc_b.fit_transform(X_tr), y_tr)
                accs.append(clf_b.score(sc_b.transform(X_te), y_te))
            mean_acc = float(np.mean(accs)) if accs else 0.5
            X_all = np.vstack([Xr, Xs])
            y_all = np.array([1] * n_r + [0] * len(Xs))
            sc_fi = StandardScaler()
            clf_fi = RandomForestClassifier(n_estimators, random_state=seed, class_weight='balanced')
            clf_fi.fit(sc_fi.fit_transform(X_all), y_all)
            fi_per_class[conc] = dict(zip(DS_FEATURE_KEYS, clf_fi.feature_importances_))
        else:
            X_all = np.vstack([Xr, Xs])
            y_all = np.array([1] * n_r + [0] * len(Xs))
            if len(X_all) < 6:
                ds_per_class[conc] = acc_per_class[conc] = None
                continue
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_all, y_all, test_size=0.30, random_state=seed, stratify=y_all)
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_all, y_all, test_size=0.30, random_state=seed)
            sc_  = StandardScaler()
            clf_ = RandomForestClassifier(n_estimators, random_state=seed, class_weight='balanced')
            clf_.fit(sc_.fit_transform(X_tr), y_tr)
            mean_acc = clf_.score(sc_.transform(X_te), y_te)
            fi_per_class[conc] = dict(zip(DS_FEATURE_KEYS, clf_.feature_importances_))

        ds_val = 1.0 - abs(mean_acc - 0.5)
        ds_per_class[conc]  = float(ds_val)
        acc_per_class[conc] = float(mean_acc)

    valid_ds = [v for v in ds_per_class.values() if v is not None]
    mean_ds  = float(np.mean(valid_ds)) if valid_ds else float('nan')

    # Global pooled DS (diagnostic only)
    n_r_all  = len(feat_r)
    n_s_g    = min(n_r_all * 5, len(feat_s))
    idx_sg   = rng.choice(len(feat_s), n_s_g, replace=False)
    Xr_g     = feat_r[DS_FEATURE_KEYS].fillna(0).values
    Xs_g     = feat_s.iloc[idx_sg][DS_FEATURE_KEYS].fillna(0).values
    X_all_g  = np.vstack([Xr_g, Xs_g])
    y_all_g  = np.array([1] * len(Xr_g) + [0] * len(Xs_g))
    try:
        Xtr_g, Xte_g, ytr_g, yte_g = train_test_split(
            X_all_g, y_all_g, test_size=0.30, random_state=seed, stratify=y_all_g)
    except Exception:
        Xtr_g, Xte_g, ytr_g, yte_g = train_test_split(
            X_all_g, y_all_g, test_size=0.30, random_state=seed)
    sc_g  = StandardScaler()
    clf_g = RandomForestClassifier(n_estimators, random_state=seed, class_weight='balanced')
    clf_g.fit(sc_g.fit_transform(Xtr_g), ytr_g)
    acc_g = clf_g.score(sc_g.transform(Xte_g), yte_g)

    return {
        'ds_per_class':    ds_per_class,
        'acc_per_class':   acc_per_class,
        'fi_per_class':    fi_per_class,
        'mean_ds':         mean_ds,
        'ds_pass':         bool(mean_ds >= DS_THR) if not np.isnan(mean_ds) else False,
        'ds_global_shape': 1.0 - abs(acc_g - 0.5),
        'acc_global':      acc_g,
        'ds_pass_threshold': DS_THR,
    }


# Section 7 - CUSTOM METRICS 
#
# Both metrics output a single float in [0.0, 1.0].
# They are penalty-based: start from 1 (perfect) and subtract.
# They are interpretable as percentages (0 % = nothing in common,
# 100 % = indistinguishable from the reference).

#Probability Density Function
class PDFOverlapScore:
    """
    PDF Overlap Score - Probability Distribution Intersection over Union.

    Mathematical definition:
    Given two continuous distributions P (real) and Q (synthetic), the PDF
    overlap is defined by the *Bhattacharyya overlap coefficient*:

        PDF_overlap(P, Q) = ∫ min(p(x), q(x)) dx

    which is the "intersection area" under both density curves.  Using a
    kernel-density or histogram approximation on n bins:

        PDF_overlap ≈ Σ_k  min(p_k, q_k) · Δx

    Properties
    • Range: [0, 1]
    • Perfect match: PDF_overlap = 1.0
    • Disjoint distribution: PDF_overlap = 0.0
    • Symmetric: overlap(P, Q) = overlap(Q, P)
    • Penalty interpretation: 1 - PDF_overlap gives the non-overlapping
      fraction.


    Usage
    >>> scorer = PDFOverlapScore(n_bins=50)
    >>> scores = scorer.per_class(X_real, y_real, X_synth, y_synth, E)
    >>> print(scores)  # {conc: overlap_score}
    """

    def __init__(self, n_bins: int = 50, E_win_start: float = -0.55, E_win_end: float = -0.25):
        self.n_bins      = n_bins
        self.E_win_start = E_win_start
        self.E_win_end   = E_win_end

    def _overlap(self, p_values: np.ndarray, q_values: np.ndarray) -> float:
        """
        Compute the PDF overlap (Bhattacharyya integral) via histogram approximation.
        Bin edges are defined on the REAL support (FIX-A convention) to prevent
        artefact inflation from extrapolated synthetic samples.
        """
        n_b = int(np.clip(self.n_bins, 10, max(10, len(p_values) // 2)))
        r_min, r_max = float(p_values.min()), float(p_values.max())
        if r_max <= r_min:
            return 1.0  # degenerate: point mass, treat as identical
        bins = np.linspace(r_min, r_max, n_b + 1)
        delta_x  = (r_max - r_min) / n_b
        p_hist, _ = np.histogram(p_values, bins=bins, density=True)
        q_hist, _ = np.histogram(np.clip(q_values, r_min, r_max), bins=bins, density=True)
        # Normalise so each integrates to 1
        p_norm = p_hist / (p_hist.sum() + 1e-12)
        q_norm = q_hist / (q_hist.sum() + 1e-12)
        return float(np.sum(np.minimum(p_norm, q_norm)))

    def per_class(
        self,
        X_real:  np.ndarray,
        y_real:  np.ndarray,
        X_synth: np.ndarray,
        y_synth: np.ndarray,
        E:       np.ndarray,
    ) -> dict:
        """
        Per-concentration-class PDF overlap on Ip-normalised peak-window currents.

        Returns
        dict {conc: overlap_score ∈ [0, 1]}
        The mean across classes is the scalar 'pdf_overlap' summary statistic.
        """
        peak_mask  = (E >= self.E_win_start) & (E <= self.E_win_end)
        real_classes = np.unique(y_real)
        y_s_binned  = assign_nearest_log_class(y_synth, real_classes)
        overlaps = {}

        for conc in sorted(real_classes):
            Xr_win = X_real[y_real == conc][:, peak_mask]
            Xs_win = X_synth[y_s_binned == conc][:, peak_mask]
            if len(Xs_win) == 0:
                overlaps[conc] = np.nan
                continue
            ip_mean = Xr_win.max(axis=1).mean()
            I_real  = (Xr_win / (ip_mean + 1e-10)).ravel()
            I_synth = (Xs_win / (ip_mean + 1e-10)).ravel()
            overlaps[conc] = self._overlap(I_real, I_synth)

        return overlaps

    def mean(self, overlaps: dict) -> float:
        """Mean PDF overlap across evaluated classes (ignores NaN)."""
        vals = [v for v in overlaps.values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')


class VoltammogramFidelityIndex:
    """
    VoltammogramFidelityIndex (VFI) - Penalty-based composite similarity score.

    Mathematical definition
    VFI is constructed by starting from a perfect score and subtracting
    normalised penalties derived from the metrics already computed in Tiers 1–2.
    It is *similar to difflib.SequenceMatcher.ratio(), which is
    defined as:

        SM.ratio() = 1 - edit_cost / max_possible_edit_cost

    Here the 'edit cost' is replaced with a weighted sum of distributional divergences:

        VFI = 1  −  w1·P_JSD  −  w2·P_SWD  −  w3·P_ACF  −  w4·P_feat

    where each penalty Pᵢ ∈ [0, 1] is a normalised version of one of the
    Tier-1/2 metrics:

        P_JSD  = clip(mean_JSD  / JSD_max_possible, 0, 1)
                 JSD_max = log₂(2) = 1 by definition.

        P_SWD  = clip(mean_SWD  / SWD_scale, 0, 1)
                 SWD_scale = 1.0  (empirical: SWD ≈ 0 for identical, ≥ 1 for random).

        P_ACF  = clip(delta_ACF / ACF_scale, 0, 1)
                 ACF_scale = 1.0  (L2-norm of ACF vectors; typical range 0–0.5).

        P_feat = clip(mean_norm_W1 / W1_scale, 0, 1)
                 W1_scale = 1.0  (normalised Wasserstein on feature distributions).

    Weights w1..w4 default to 0.25, 0.25, 0.25, 0.25.
    Weights sum to 1 so the total penalty is also in [0, 1] and VFI ∈ [0, 1].

    Why it is similar to the SequenceMatcher analogy
    - difflib starts at 1.0 and subtracts the fraction of unmatched characters.
    - VFI starts at 1.0 and subtracts the weighted fraction of distributional
      mismatch (JSD divergence + shape distance + noise texture drift + feature drift).
    - The output is a single interpretable percentage: 0.85 means "85 % fidelity
      to the reference distribution" - as intuitive as a string similarity ratio.

    Interpretation guide
    VFI ≥ 0.90 : Excellent - synthetic batch nearly indistinguishable from real
    VFI ∈ [0.75, 0.90) : Good - minor distributional mismatch
    VFI ∈ [0.60, 0.75) : Marginal - tune augmentation parameters
    VFI  < 0.60 : Poor - significant mismatch; batch should not be used

    Usage
    -----
    >>> vfi_calc = VoltammogramFidelityIndex(w_jsd=0.40, w_swd=0.25, w_acf=0.15, w_feat=0.20)
    >>> vfi = vfi_calc.compute(
    ...     mean_jsd=0.08, mean_swd=0.12, delta_acf=0.04,
    ...     mean_feat_w1=0.15, verbose=True
    ... )
    >>> print(f"VFI = {vfi:.4f}")
    """

    def __init__(
        self,
        w_jsd:    float = 0.25,
        w_swd:    float = 0.25,
        w_acf:    float = 0.25,
        w_feat:   float = 0.25,
        jsd_scale: float = 1.0,
        swd_scale: float = 1.0,
        acf_scale: float = 1.0,
        w1_scale:  float = 1.0,
    ):
        assert abs(w_jsd + w_swd + w_acf + w_feat - 1.0) < 1e-9, \
            "Weights must sum to 1.0"
        self.w_jsd     = w_jsd
        self.w_swd     = w_swd
        self.w_acf     = w_acf
        self.w_feat    = w_feat
        self.jsd_scale = jsd_scale
        self.swd_scale = swd_scale
        self.acf_scale = acf_scale
        self.w1_scale  = w1_scale

    def compute(
        self,
        mean_jsd:     float,
        mean_swd:     float,
        delta_acf:    float,
        mean_feat_w1: float,
        verbose:      bool = False,
    ) -> float:
        """
        Compute the VoltammogramFidelityIndex from pre-computed sub-metrics.

        Parameters
        mean_jsd     : mean per-class JSD (Tier 1)
        mean_swd     : mean per-class SWD (Tier 1)
        delta_acf    : L2-norm of (mean_ACF_real − mean_ACF_synth) (Tier 2)
        mean_feat_w1 : mean normalised Wasserstein-1 across shape features (Tier 2 PFF)
        verbose      : if True, prints a breakdown of each penalty component

        Returns
        VFI ∈ [0.0, 1.0]
        """
        p_jsd  = float(np.clip(mean_jsd / self.jsd_scale, 0.0, 1.0))
        p_swd  = float(np.clip(mean_swd / self.swd_scale, 0.0, 1.0))
        p_acf  = float(np.clip(delta_acf / self.acf_scale, 0.0, 1.0))
        p_feat = float(np.clip(mean_feat_w1 / self.w1_scale,  0.0, 1.0))

        total_penalty = (
            self.w_jsd  * p_jsd  +
            self.w_swd  * p_swd  +
            self.w_acf  * p_acf  +
            self.w_feat * p_feat
        )
        vfi = float(np.clip(1.0 - total_penalty, 0.0, 1.0))

        if verbose:
            print("VoltammogramFidelityIndex breakdown:")
            print(f"  P_JSD  = {p_jsd:.4f}  (mean_JSD={mean_jsd:.4f}, weight={self.w_jsd})")
            print(f"  P_SWD  = {p_swd:.4f}  (mean_SWD={mean_swd:.4f}, weight={self.w_swd})")
            print(f"  P_ACF  = {p_acf:.4f}  (delta_ACF={delta_acf:.4f}, weight={self.w_acf})")
            print(f"  P_feat = {p_feat:.4f}  (mean_W1={mean_feat_w1:.4f}, weight={self.w_feat})")
            print(f"  Total penalty = {total_penalty:.4f}")
            print(f"  -- VFI = {vfi:.4f}  ({_vfi_label(vfi)}) --")
        return vfi

    @staticmethod
    def from_gate_results(results: dict, **kwargs) -> float:
        """
        Convenience method: compute VFI directly from a ValidationGate.run() dict.

        Example
        >>> results = gate.run(X_synth, y_synth)
        >>> vfi = VoltammogramFidelityIndex.from_gate_results(results, verbose=True)
        """
        calc = VoltammogramFidelityIndex(**{k: v for k, v in kwargs.items()
                                            if k in ('w_jsd', 'w_swd', 'w_acf', 'w_feat',
                                                      'jsd_scale', 'swd_scale', 'acf_scale', 'w1_scale')})
        return calc.compute(
            mean_jsd     = results.get('mean_jsd', 0.0),
            mean_swd     = results.get('swd_mean', 0.0),
            delta_acf    = results.get('delta_acf', 0.0),
            mean_feat_w1 = results.get('mean_feat_w1', 0.0),
            verbose      = kwargs.get('verbose', False),
        )


def _vfi_label(vfi: float) -> str:
    if vfi >= 0.90: return "Excellent"
    if vfi >= 0.75: return "Good"
    if vfi >= 0.60: return "Marginal"
    return "Poor"


# Section 8 - ValidationGate: orchestrating class
class ValidationGate:
    """
    4-Tier Validation Gate for physics-informed augmented voltammogram signals.

    Encapsulates all 4 tiers in a single importable class. Create one instance
    per real-data baseline, then call .run() for each synthetic batch you want
    to evaluate.

    Parameters
    E           : potential array (V), shape (n_points,)
    X_real      : real signal matrix (n_real, n_points)
    y_real      : real concentration labels (n_real,) in µM
    E_win_start : peak window left boundary (V), default -0.55
    E_win_end   : peak window right boundary (V), default -0.25
    max_acf_lag : maximum autocorrelation lag, default 20
    vfi_weights : optional dict with keys w_jsd, w_swd, w_acf, w_feat

    Examples
    >>> import numpy as np
    >>> from validation_metrics import ValidationGate
    >>> gate = ValidationGate(E, X_real, y_real)
    >>> results_batch1 = gate.run(X_synth1, y_synth1, verbose=True)
    >>> results_batch2 = gate.run(X_synth2, y_synth2, verbose=True)
    >>> # Compare VFI across batches
    >>> print(results_batch1['vfi'], results_batch2['vfi'])
    """

    def __init__(
        self,
        E:            np.ndarray,
        X_real:       np.ndarray,
        y_real:       np.ndarray,
        E_win_start:  float = -0.55,
        E_win_end:    float = -0.25,
        max_acf_lag:  int   = 20,
        vfi_weights:  Optional[dict] = None,
    ):
        self.E            = np.asarray(E, dtype=float)
        self.X_real       = np.asarray(X_real, dtype=float)
        self.y_real       = np.asarray(y_real, dtype=float)
        self.E_win_start  = E_win_start
        self.E_win_end    = E_win_end
        self.max_acf_lag  = max_acf_lag
        self.peak_mask    = (E >= E_win_start) & (E <= E_win_end)

        # Pre-compute real-side features (fixed across batches)
        self.feat_real = build_feature_matrix(X_real, E, E_win_start, E_win_end)

        # VFI calculator
        _weights = vfi_weights or {}
        self._vfi = VoltammogramFidelityIndex(**_weights)

        # PDF overlap scorer
        self._pdf = PDFOverlapScore(E_win_start=E_win_start, E_win_end=E_win_end)

    # API
    def run(
        self,
        X_synth:  np.ndarray,
        y_synth:  np.ndarray,
        verbose:  bool = True,
        run_tiers: Sequence[int] = (1, 2, 3, 4),
    ) -> dict:
        """
        Validate a batch of synthetic signals against the real baseline.

        Parameters
        X_synth   : synthetic signal matrix (n_synth, n_points)
        y_synth   : synthetic concentration labels (n_synth,) in µM
        verbose   : print summary tables if True
        run_tiers : which tiers to execute (subset of {1,2,3,4})

        Returns
        results dict with keys:
          tier1_pass, tier2_pass, tier3_pass, tier4_pass, final_pass
          vfi           : VoltammogramFidelityIndex (float ∈ [0,1])
          pdf_overlap   : per-class PDF overlap dict + 'mean' key
          + all intermediate metrics from each tier
        """
        X_synth = np.asarray(X_synth, dtype=float)
        y_synth = np.asarray(y_synth, dtype=float)

        feat_synth     = build_feature_matrix(X_synth, self.E, self.E_win_start, self.E_win_end)
        X_r_norm, X_s_norm, ip_by_class = build_class_normalised_matrices(
            self.X_real, self.y_real, X_synth, y_synth,
            self.E, self.E_win_start, self.E_win_end,
        )

        results: dict = {'feat_synth': feat_synth}

        # Tier 1 
        if 1 in run_tiers:
            results.update(self._run_tier1(X_synth, y_synth, X_r_norm, X_s_norm))

        # Tier 2 
        if 2 in run_tiers:
            results.update(self._run_tier2(X_synth, y_synth, feat_synth))

        # Tier 3 
        if 3 in run_tiers:
            results.update(self._run_tier3(feat_synth, y_synth))

        # Tier 4 
        if 4 in run_tiers:
            results.update(self._run_tier4(feat_synth, y_synth))

        # Final pass/fail
        results['final_pass'] = all([
            results.get('tier1_pass', True),
            results.get('tier2_pass', True),
            results.get('tier3_pass', True),
            results.get('tier4_pass', True),
        ])

        # Custom metrics
        pdf_overlaps = self._pdf.per_class(self.X_real, self.y_real, X_synth, y_synth, self.E)
        pdf_overlaps['mean'] = self._pdf.mean(pdf_overlaps)
        results['pdf_overlap'] = pdf_overlaps

        # Mean normalised W1 across PFF features (for VFI)
        if 'pff_df' in results and not results['pff_df'].empty:
            per_class_w1 = (
            results['pff_df']
            .dropna(subset=['Wasserstein'])
            .groupby('class_uM')['Wasserstein']
            .mean()
            )
            mean_feat_w1 = float(per_class_w1.median()) if len(per_class_w1) else 0.0
        else:
            mean_feat_w1 = 0.0
        results['mean_feat_w1'] = mean_feat_w1

        vfi = self._vfi.compute(
            mean_jsd     = results.get('mean_jsd',   0.0),
            mean_swd     = results.get('swd_mean',   0.0),
            delta_acf    = results.get('delta_acf',  0.0),
            mean_feat_w1 = mean_feat_w1,
            verbose      = False,
        )
        results['vfi'] = vfi

        if verbose:
            self._print_summary(results)

        return results

    @classmethod
    def from_csv(cls, path_potential_grid: str, path_signals_real: str, target_col: str = 'concentration'):
        """
        Create a ValidationGate instance from CSV files.
        """
        # Potential axis (E)
        # It is assumed that the CSV contains a single column of potentials (V) in ascending order.
        E = pd.read_csv(path_potential_grid).values.flatten()
        
        # Real signals and labels
        df_real = pd.read_csv(path_signals_real)
        y_real = df_real[target_col].values
        X_real = df_real.drop(columns=[target_col]).values
        
        # Create the ValidationGate instance
        return cls(E, X_real, y_real)

    def evaluate_csv(self, path_signals_augmented: str, target_col: str = 'concentration', **kwargs):
        """
        Run the validation gate on a synthetic batch read from a CSV file.
        """
        df_aug = pd.read_csv(path_signals_augmented)
        y_aug = df_aug[target_col].values
        X_aug = df_aug.drop(columns=[target_col]).values
        
        # Run the gate on the augmented data
        return self.run(X_aug, y_aug, **kwargs)


    # Tier implementations (private)
    def _run_tier1(self, X_synth, y_synth, X_r_norm, X_s_norm) -> dict:
        # KS (stratified)
        ks_per_class, ks_mean_frac, ks_min_frac = compute_stratified_ks(X_r_norm, X_s_norm)
        ks_pass = ks_mean_frac >= 0.90

        # JSD
        jsd_map  = compute_per_class_jsd(self.X_real, self.y_real, X_synth, y_synth, self.E,
                                          self.E_win_start, self.E_win_end)
        jsd_vals = [v for v in jsd_map.values() if not np.isnan(v)]
        mean_jsd = float(np.mean(jsd_vals)) if jsd_vals else np.nan
        max_jsd  = float(np.max(jsd_vals))  if jsd_vals else np.nan
        jsd_pass = (mean_jsd < 0.15) and (max_jsd < 0.25) if not np.isnan(mean_jsd) else True

        # MMD^2
        X_r_pool = np.vstack(list(X_r_norm.values()))
        X_s_pool = np.vstack(list(X_s_norm.values()))
        mmd2, gamma_used = compute_mmd_squared(X_r_pool, X_s_pool)
        mmd_pass = mmd2 < 0.05

        # SWD
        swd_per_class, swd_mean, swd_max = compute_stratified_swd(X_r_norm, X_s_norm)
        swd_pass = (swd_mean < 0.30) and (swd_max < 0.40) if not np.isnan(swd_mean) else True

        tier1_pass = ks_pass and jsd_pass and mmd_pass and swd_pass

        return {
            'tier1_pass':    tier1_pass,
            'ks_per_class':  ks_per_class,
            'ks_mean_frac':  ks_mean_frac,
            'ks_min_frac':   ks_min_frac,
            'ks_pass':       ks_pass,
            'jsd_map':       jsd_map,
            'mean_jsd':      mean_jsd,
            'max_jsd':       max_jsd,
            'jsd_pass':      jsd_pass,
            'mmd2':          mmd2,
            'gamma_used':    gamma_used,
            'mmd_pass':      mmd_pass,
            'swd_per_class': swd_per_class,
            'swd_mean':      swd_mean,
            'swd_max':       swd_max,
            'swd_pass':      swd_pass,
        }

    def _run_tier2(self, X_synth, y_synth, feat_synth) -> dict:
        # PFF
        pff_df, pff_class_pass = compute_pff_stratified(
            self.feat_real, self.y_real, feat_synth, y_synth)
        n_cls = sum(1 for v in pff_class_pass.values() if v is not None)
        n_ok  = sum(1 for v in pff_class_pass.values() if v is True)
        pff_frac = n_ok / n_cls if n_cls > 0 else 0.0
        pff_pass = pff_frac >= 0.70

        # Randles-Ševčík
        rs = compute_randles_sevcik_conformance(
            self.feat_real, feat_synth, self.y_real, y_synth)
        rs_pass = rs['rs_pass']

        # ACF
        MAX_LAG = self.max_acf_lag
        acf_real  = compute_acf_vectors(self.X_real, MAX_LAG)
        acf_synth = compute_acf_vectors(X_synth,     MAX_LAG)
        mu_r = acf_real.mean(axis=0); mu_s = acf_synth.mean(axis=0)
        delta_acf = float(np.linalg.norm(mu_r - mu_s))

        alpha_bonf = 0.05 / MAX_LAG
        lag_pvals  = np.array([ttest_ind(acf_real[:, l], acf_synth[:, l]).pvalue
                                for l in range(MAX_LAG)])
        n_sig_lags = int(np.sum(lag_pvals < alpha_bonf))

        acf_pass = (delta_acf < 0.10) and (n_sig_lags <= MAX_LAG // 10)
        tier2_pass = pff_pass and rs_pass and acf_pass

        return {
            'tier2_pass':    tier2_pass,
            'pff_df':        pff_df,
            'pff_class_pass': pff_class_pass,
            'pff_frac':      pff_frac,
            'pff_pass':      pff_pass,
            'rs':            rs,
            'rs_pass':       rs_pass,
            'delta_acf':     delta_acf,
            'n_sig_lags':    n_sig_lags,
            'acf_pass':      acf_pass,
        }

    def _run_tier3(self, feat_synth, y_synth) -> dict:
        X_r, y_r = build_feature_matrix_array(self.feat_real, self.y_real)
        X_s, y_s = build_feature_matrix_array(feat_synth,     y_synth)

        tstr_res = run_tstr_trtr(X_r, y_r, X_s, y_s)
        tstr_pass = tstr_res['tstr_pass']

        trts_res = run_trts(X_r, y_r, X_s, y_s)
        trts_pass = trts_res['trts_pass']

        gain_results = run_augmentation_gain(X_r, y_r, X_s, y_s)
        ratios_  = sorted(gain_results.keys())
        rmse_vals = [gain_results[r]['log_rmse'] for r in ratios_]
        min_rmse = min(rmse_vals)
        rmse_20x = gain_results.get(20, {}).get('log_rmse', float('nan'))
        gain_pass = (rmse_20x <= min_rmse * 1.15) if not np.isnan(rmse_20x) else True

        tier3_pass = tstr_pass and trts_pass and gain_pass

        return {
            'tier3_pass':    tier3_pass,
            'tstr_res':      tstr_res,
            'tstr_pass':     tstr_pass,
            'trts_res':      trts_res,
            'trts_pass':     trts_pass,
            'gain_results':  gain_results,
            'gain_pass':     gain_pass,
        }

    def _run_tier4(self, feat_synth, y_synth) -> dict:
        disc = compute_discriminative_score_stratified(
            self.feat_real, self.y_real, feat_synth, y_synth)
        tier4_pass = disc['ds_pass']

        return {
            'tier4_pass': tier4_pass,
            'disc':       disc,
            'mean_ds':    disc['mean_ds'],
            'ds_pass':    disc['ds_pass'],
        }

    # Summary printer
    def _print_summary(self, r: dict) -> None:
        PASS = lambda x: '✅ PASS' if x else '❌ FAIL'
        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║           VALIDATION GATE - FINAL DECISION              ║")
        print("╠══════════════════════════════════════════════════════════╣")

        if 'tier1_pass' in r:
            print(f"║  Tier 1 - Statistical Distributional  : {PASS(r['tier1_pass']):<20}║")
            print(f"║    KS mean frac                       : {r.get('ks_mean_frac', float('nan')):.3f}  (≥0.90){'':>16}║")
            print(f"║    JSD mean / max                     : {r.get('mean_jsd', float('nan')):.4f} / {r.get('max_jsd', float('nan')):.4f}           ║")
            print(f"║    MMD²                               : {r.get('mmd2', float('nan')):.6f}                ║")
            print(f"║    SWD mean / max                     : {r.get('swd_mean', float('nan')):.4f} / {r.get('swd_max', float('nan')):.4f}           ║")

        if 'tier2_pass' in r:
            print("╠══════════════════════════════════════════════════════════╣")
            print(f"║  Tier 2 - Physics-Anchored            : {PASS(r['tier2_pass']):<20}║")
            rs = r.get('rs', {})
            print(f"║    PFF class pass frac                : {r.get('pff_frac', float('nan')):.3f}  (≥0.70){'':>16}║")
            print(f"║    Randles-Ševčík R²                  : {rs.get('R2_synth', float('nan')):.4f}  (≥0.90)         ║")
            print(f"║    ACF Δ (full signal)                : {r.get('delta_acf', float('nan')):.5f}  (<0.10)        ║")

        if 'tier3_pass' in r:
            print("╠══════════════════════════════════════════════════════════╣")
            print(f"║  Tier 3 - Utility                     : {PASS(r['tier3_pass']):<20}║")
            tstr = r.get('tstr_res', {})
            print(f"║    TSTR log-RMSE ratio                : {tstr.get('log_rmse_ratio', float('nan')):.4f}  (≤1.20)         ║")

        if 'tier4_pass' in r:
            print("╠══════════════════════════════════════════════════════════╣")
            print(f"║  Tier 4 - Discriminative Score        : {PASS(r['tier4_pass']):<20}║")
            print(f"║    Mean DS                            : {r.get('mean_ds', float('nan')):.4f}  (≥0.70)         ║")

        print("╠══════════════════════════════════════════════════════════╣")
        vfi  = r.get('vfi', float('nan'))
        povl = r.get('pdf_overlap', {}).get('mean', float('nan'))
        print(f"║  NEW  VoltammogramFidelityIndex (VFI)  : {vfi:.4f}  [{_vfi_label(vfi):<9}]  ║")
        print(f"║  NEW  PDF Overlap Score (mean)         : {povl:.4f}                       ║")
        print("╠══════════════════════════════════════════════════════════╣")
        final = r.get('final_pass', False)
        if final:
            print("║  ✅  BATCH VALIDATED - Proceed to training pool          ║")
        else:
            print("║  ❌  BATCH REJECTED - Re-tune augmentation parameters    ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()


#  
# Section 9 - Convenience entry-point
def quick_compare(
    E:       np.ndarray,
    X_real:  np.ndarray,
    y_real:  np.ndarray,
    batches: dict,
    run_tiers: Sequence[int] = (1, 2),
) -> pd.DataFrame:
    """
    Compare multiple synthetic batches against a common real baseline.

    Runs the ValidationGate on each batch and returns a comparison DataFrame
    with one row per batch, showing the key scalar metrics and both custom
    similarity scores.

    Parameters
    E        : potential array
    X_real   : real signal matrix
    y_real   : real concentration labels
    batches  : dict {batch_name: (X_synth, y_synth)}
    run_tiers: which tiers to include (Tiers 3–4 are slow; Tiers 1–2 are fast)

    Returns
    pd.DataFrame with columns: batch, KS_mean, JSD_mean, MMD2, SWD_mean,
                                PFF_frac, RS_R2, delta_ACF,
                                VFI, PDF_overlap_mean, final_pass

    Example
    >>> df = quick_compare(E, X_real, y_real,
    ...                    {'batch_v1': (X1, y1), 'batch_v2': (X2, y2)},
    ...                    run_tiers=[1, 2])
    >>> print(df.to_string())
    """
    gate = ValidationGate(E, X_real, y_real)
    rows = []
    for name, (X_s, y_s) in batches.items():
        r = gate.run(X_s, y_s, verbose=False, run_tiers=run_tiers)
        rows.append({
            'batch':           name,
            'KS_mean_frac':    round(r.get('ks_mean_frac',  float('nan')), 4),
            'JSD_mean':        round(r.get('mean_jsd',       float('nan')), 4),
            'MMD2':            round(r.get('mmd2',           float('nan')), 6),
            'SWD_mean':        round(r.get('swd_mean',       float('nan')), 4),
            'PFF_class_frac':  round(r.get('pff_frac',       float('nan')), 3),
            'RS_R2_synth':     round(r.get('rs', {}).get('R2_synth', float('nan')), 4),
            'delta_ACF':       round(r.get('delta_acf',      float('nan')), 5),
            'VFI':             round(r.get('vfi',            float('nan')), 4),
            'VFI_label':       _vfi_label(r.get('vfi', 0.0)),
            'PDF_overlap_mean': round(r.get('pdf_overlap', {}).get('mean', float('nan')), 4),
            'tier1_pass':      r.get('tier1_pass'),
            'tier2_pass':      r.get('tier2_pass'),
        })
    return pd.DataFrame(rows)
