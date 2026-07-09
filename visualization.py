"""Visual comparison of synthetic voltammogram sources (physics-informed
augmentation vs GAN architectures) against real calibration data.

Every plotting function accepts a `real` tuple `(X_real, y_real)` and a
`sources` dict `{source_name: (X, y)}` so that any number of generators can
be overlaid side by side, e.g.:

>>> real = (X_real, y_real)
>>> sources = {
...     'Physics-Augmented': (X_aug, y_aug),
...     'TimeGAN':            (X_timegan, y_timegan),
...     'WGAN-GP':             (X_wgangp, y_wgangp),
... }
>>> plot_signal_overlay(E, real, sources, concentration=10.0).show()

The first group (plot_signal_overlay / plot_overlay_grid /
plot_mean_envelope_comparison / plot_ip_vs_concentration /
plot_feature_pca_scatter) are generic "does this look right" comparisons.

The second group (plot_pdf_overlap_explainer / plot_pdf_overlap_by_class /
plot_vfi_breakdown_bars / plot_fidelity_radar / plot_jsd_per_class /
plot_metric_summary_bars) do not recompute any metric - they visualise the
PDF Overlap Score, VoltammogramFidelityIndex, JSD, MMD2 and SWD numbers that
`validation_metrics.ValidationGate` / `PDFOverlapScore` /
`VoltammogramFidelityIndex` already compute, so the numbers in the
validation gate summary have a picture attached to them.

All figures are styled with `plot_style.apply_default_plotly_layout`.
Multi-panel figures widen the default 880px canvas afterwards (the same
deviation `full_range_data_augmentation.ipynb` itself makes for its
`plot_real_vs_synthetic` grid) - every other layout property is untouched.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from plot_style import (
    PLOTLY_FIGURE_HEIGHT_PX,
    PLOTLY_FIGURE_WIDTH_PX,
    PLOTLY_TEMPLATE,
    apply_default_plotly_layout,
)
from validation_metrics import VoltammogramFidelityIndex

# Section 0 - shared palette + small helpers
SOURCE_PALETTE = {
    'Real': '#111111',
    'Physics-Augmented': '#17becf',
    'Augmented_Baseline': '#17becf',  # alias used by validation_metrics.quick_compare examples
    'TimeGAN': '#e34a1a',
    'WGAN-GP': '#5a23c4',
    'WGANGP': '#5a23c4',
}
_FALLBACK_COLORS = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#bcbd22']


def _resolve_colors(names: Sequence[str]) -> dict:
    colors, fallback_i = {}, 0
    for name in names:
        if name in colors:
            continue
        if name in SOURCE_PALETTE:
            colors[name] = SOURCE_PALETTE[name]
        else:
            colors[name] = _FALLBACK_COLORS[fallback_i % len(_FALLBACK_COLORS)]
            fallback_i += 1
    return colors


def _as_arrays(X, y) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _select_signals(X: np.ndarray, y: np.ndarray, target_c: float, n_traces: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Exact-match replicates if available, else the n_traces log-nearest ones."""
    exact = np.where(np.isclose(y, target_c))[0]
    idx = exact if len(exact) > 0 else np.argsort(
        np.abs(np.log(np.clip(y, 1e-12, None)) - np.log(max(target_c, 1e-12))))[:n_traces]
    if len(idx) > n_traces:
        idx = rng.choice(idx, size=n_traces, replace=False)
    return idx


def _estimate_ip(I: np.ndarray, E: np.ndarray, E_win: tuple[float, float]) -> float:
    mask = (E >= E_win[0]) & (E <= E_win[1])
    return float(np.max(I[mask])) if mask.any() else float(np.max(I))


# Section 1 - shape / overlay comparisons
def plot_signal_overlay(E: np.ndarray, real: tuple, sources: dict, concentration: float,
                         n_traces: int = 15, seed: int = 7, title: Optional[str] = None) -> go.Figure:
    """Simple overlay: real replicates vs every synthetic source at one concentration."""
    rng = np.random.default_rng(seed)
    X_real, y_real = _as_arrays(*real)
    colors = _resolve_colors(['Real'] + list(sources.keys()))

    fig = go.Figure()

# Plot Real Signals 
    if concentration is not None:
        real_idx = _select_signals(X_real, y_real, concentration, n_traces=15, rng=rng)
    else:
        # Pick random signals across all concentrations
        max_real = len(X_real)
        n_real = max_real if n_traces is None else min(15, max_real)
        real_idx = rng.choice(max_real, n_real, replace=False)

    for k, i in enumerate(real_idx):
        fig.add_trace(go.Scatter(
            x=E, y=X_real[i], mode='lines', line=dict(color=colors['Real'], width=2.0),
            opacity=0.9, name='Real', legendgroup='Real', showlegend=(k == 0)))

    # Plot Generated Signals
    for name, (X, y) in sources.items():
        X, y = _as_arrays(X, y)
        
        if concentration is not None:
            idx = _select_signals(X, y, concentration, n_traces=n_traces, rng=rng)
        else:
            # Pick random signals across all generated data
            max_gen = len(X)
            n_gen = max_gen if n_traces is None else min(n_traces, max_gen)
            idx = rng.choice(max_gen, n_gen, replace=False)
            
        for k, i in enumerate(idx):
            fig.add_trace(go.Scatter(
                x=E, y=X[i], mode='lines', line=dict(color=colors[name], width=1.0),
                opacity=0.45, name=name, legendgroup=name, showlegend=(k == 0)))

    default_title = f'Signal overlay near c = {concentration} µM' if concentration is not None else 'Signal overlay (All Concentrations)'
    apply_default_plotly_layout(fig, title or default_title)
    return fig


def plot_overlay_grid(E: np.ndarray, real: tuple, sources: dict,
                       concentrations: Sequence[float] = (0.5, 10.0, 75.0),
                       n_traces: int = 10, seed: int = 7, title: Optional[str] = None) -> go.Figure:
    """Overlay grid across several concentrations - low / mid / high in one figure."""
    rng = np.random.default_rng(seed)
    X_real, y_real = _as_arrays(*real)
    colors = _resolve_colors(['Real'] + list(sources.keys()))

    fig = make_subplots(rows=1, cols=len(concentrations),
                         subplot_titles=[f'c = {c} µM' for c in concentrations],
                         horizontal_spacing=0.06)

    for col, c_t in enumerate(concentrations, 1):
        real_idx = _select_signals(X_real, y_real, c_t, n_traces=1000, rng=rng)
        for k, i in enumerate(real_idx):
            fig.add_trace(go.Scatter(
                x=E, y=X_real[i], mode='lines', line=dict(color=colors['Real'], width=1.8),
                opacity=0.9, name='Real', legendgroup='Real',
                showlegend=(k == 0 and col == 1)), row=1, col=col)
        for name, (X, y) in sources.items():
            X, y = _as_arrays(X, y)
            idx = _select_signals(X, y, c_t, n_traces=n_traces, rng=rng)
            for k, i in enumerate(idx):
                fig.add_trace(go.Scatter(
                    x=E, y=X[i], mode='lines', line=dict(color=colors[name], width=0.9),
                    opacity=0.4, name=name, legendgroup=name,
                    showlegend=(k == 0 and col == 1)), row=1, col=col)

    apply_default_plotly_layout(fig, title or 'Real vs synthetic overlay (multi-concentration)')
    fig.update_layout(width=max(PLOTLY_FIGURE_WIDTH_PX, 380 * len(concentrations)))
    fig.update_yaxes(title_text='Current I (µA)', row=1, col=1)
    return fig


def plot_mean_envelope_comparison(E: np.ndarray, real: tuple, sources: dict, concentration: float,
                                   band: str = 'std', title: Optional[str] = None) -> go.Figure:
    """Mean ± spread envelope per source - a shape/noise-texture view, not individual traces."""
    X_real, y_real = _as_arrays(*real)
    colors = _resolve_colors(['Real'] + list(sources.keys()))
    fig = go.Figure()
    rng = np.random.default_rng(0)

    def _add_band(name: str, X: np.ndarray, y: np.ndarray) -> None:
        idx = _select_signals(X, y, concentration, n_traces=max(len(X), 1), rng=rng)
        if len(idx) < 2:
            return
        M = X[idx]
        mean = M.mean(axis=0)
        spread = (M.std(axis=0) if band == 'std'
                  else (np.percentile(M, 75, axis=0) - np.percentile(M, 25, axis=0)) / 2)
        color = colors[name]
        fig.add_trace(go.Scatter(
            x=np.r_[E, E[::-1]], y=np.r_[mean + spread, (mean - spread)[::-1]],
            fill='toself', fillcolor=color, opacity=0.15, line=dict(width=0),
            showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=E, y=mean, mode='lines', line=dict(color=color, width=2.2), name=name))

    _add_band('Real', X_real, y_real)
    for name, (X, y) in sources.items():
        _add_band(name, *_as_arrays(X, y))

    apply_default_plotly_layout(fig, title or f'Mean ± {band} envelope near c = {concentration} µM')
    return fig


# Section 2 - feature-space comparisons
def plot_ip_vs_concentration(E: np.ndarray, real: tuple, sources: dict, ip_spline=None,
                              E_win: tuple[float, float] = (-0.55, -0.25),
                              title: Optional[str] = None) -> go.Figure:
    """Peak current vs concentration for every source, optionally against the PCHIP target."""
    X_real, y_real = _as_arrays(*real)
    colors = _resolve_colors(['Real'] + list(sources.keys()))
    fig = go.Figure()

    ip_real = np.array([_estimate_ip(I, E, E_win) for I in X_real])
    fig.add_trace(go.Scatter(
        x=y_real, y=ip_real, mode='markers', name='Real',
        marker=dict(color=colors['Real'], size=11, symbol='star', line=dict(color='black', width=1))))

    for name, (X, y) in sources.items():
        X, y = _as_arrays(X, y)
        ip = np.array([_estimate_ip(I, E, E_win) for I in X])
        fig.add_trace(go.Scatter(x=y, y=ip, mode='markers', name=name,
                                  marker=dict(color=colors[name], size=5, opacity=0.45)))

    if ip_spline is not None:
        c_grid, ip_grid = ip_spline.dense_grid()
        fig.add_trace(go.Scatter(x=c_grid, y=ip_grid, mode='lines', name='PCHIP target (real)',
                                  line=dict(color='darkorange', width=2.2, dash='dash')))

    fig.update_xaxes(type='log')
    apply_default_plotly_layout(fig, title or 'Peak current (Ip) vs concentration',
                                 xaxis_title='Concentration (µM, log)', yaxis_title='Ip (µA)')
    return fig


def plot_feature_pca_scatter(E: np.ndarray, real: tuple, sources: dict,
                              E_win: tuple[float, float] = (-0.55, -0.25),
                              title: Optional[str] = None, seed: int = 42) -> go.Figure:
    """2-D PCA of Ip-normalised peak-window shapes - visual proxy for the gate's MMD/SWD checks."""
    mask = (E >= E_win[0]) & (E <= E_win[1])
    X_real, y_real = _as_arrays(*real)
    colors = _resolve_colors(['Real'] + list(sources.keys()))

    def _norm(X: np.ndarray) -> np.ndarray:
        W = X[:, mask]
        ip = np.clip(W.max(axis=1, keepdims=True), 1e-10, None)
        return W / ip

    blocks, labels = [_norm(X_real)], ['Real'] * len(X_real)
    for name, (X, y) in sources.items():
        X, _y = _as_arrays(X, y)
        blocks.append(_norm(X))
        labels += [name] * len(X)
    stacked = np.vstack(blocks)
    labels = np.array(labels)

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(stacked)
    var = pca.explained_variance_ratio_ * 100

    fig = go.Figure()
    for name in ['Real'] + list(sources.keys()):
        m = labels == name
        fig.add_trace(go.Scatter(
            x=coords[m, 0], y=coords[m, 1], mode='markers', name=name,
            marker=dict(color=colors[name], size=8 if name == 'Real' else 6,
                        opacity=0.9 if name == 'Real' else 0.55,
                        symbol='star' if name == 'Real' else 'circle',
                        line=dict(color='black', width=0.5) if name == 'Real' else None)))

    apply_default_plotly_layout(
        fig, title or 'PCA of Ip-normalised peak-window shapes',
        xaxis_title=f'PC1 ({var[0]:.1f}% var)', yaxis_title=f'PC2 ({var[1]:.1f}% var)')
    return fig


# Section 3 - metric explainers (PDF Overlap Score / VFI / JSD / MMD / SWD)
def plot_pdf_overlap_explainer(E: np.ndarray, real: tuple, source: tuple, concentration: float,
                                source_name: str = 'Synthetic', E_win: tuple[float, float] = (-0.55, -0.25),
                                n_bins: int = 50, title: Optional[str] = None) -> go.Figure:
    """
    Shades the histogram intersection behind the PDF Overlap Score and
    annotates the resulting overlap fraction, using the exact same
    Bhattacharyya-intersection math as
    `validation_metrics.PDFOverlapScore._overlap`.
    """
    X_real, y_real = _as_arrays(*real)
    X_synth, y_synth = _as_arrays(*source)
    mask = (E >= E_win[0]) & (E <= E_win[1])

    real_classes = np.unique(y_real)
    nearest_real_c = real_classes[np.argmin(
        np.abs(np.log(real_classes) - np.log(max(concentration, 1e-12))))]

    Xr_win = X_real[y_real == nearest_real_c][:, mask]
    synth_mask = np.abs(np.log(np.clip(y_synth, 1e-12, None)) - np.log(max(concentration, 1e-12))) < 0.35
    Xs_win = X_synth[synth_mask][:, mask]
    if len(Xs_win) == 0:
        raise ValueError(f'No {source_name} signals found near concentration {concentration} µM')

    ip_mean = Xr_win.max(axis=1).mean()
    I_real = (Xr_win / (ip_mean + 1e-10)).ravel()
    I_synth = (Xs_win / (ip_mean + 1e-10)).ravel()

    n_b = int(np.clip(n_bins, 10, max(10, len(I_real) // 2)))
    r_min, r_max = float(I_real.min()), float(I_real.max())
    bins = np.linspace(r_min, r_max, n_b + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    p_hist, _ = np.histogram(I_real, bins=bins, density=True)
    q_hist, _ = np.histogram(np.clip(I_synth, r_min, r_max), bins=bins, density=True)
    p_norm = p_hist / (p_hist.sum() + 1e-12)
    q_norm = q_hist / (q_hist.sum() + 1e-12)
    overlap = float(np.sum(np.minimum(p_norm, q_norm)))

    colors = _resolve_colors(['Real', source_name])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers, y=p_norm, width=bin_width, marker_color=colors['Real'],
                          opacity=0.55, name='Real'))
    fig.add_trace(go.Bar(x=centers, y=q_norm, width=bin_width, marker_color=colors[source_name],
                          opacity=0.55, name=source_name))
    fig.add_trace(go.Bar(x=centers, y=np.minimum(p_norm, q_norm), width=bin_width,
                          marker_color='#333333', opacity=0.85, name='Overlap (min density)'))
    fig.update_layout(barmode='overlay')
    apply_default_plotly_layout(
        fig, title or f'PDF overlap @ {nearest_real_c} µM — {source_name} vs Real  (overlap = {overlap:.3f})',
        xaxis_title='Ip-normalised peak current', yaxis_title='Normalised density (bin mass)')
    return fig


def plot_pdf_overlap_by_class(pdf_overlap_by_source: dict, title: Optional[str] = None) -> go.Figure:
    """
    Grouped bar of `PDFOverlapScore.per_class(...)` results per source.

    pdf_overlap_by_source : {source_name: {concentration: overlap_score}}
        (pass the dict returned by `PDFOverlapScore.per_class`, or
        `ValidationGate.run()`'s `results['pdf_overlap']`, per source)
    """
    colors = _resolve_colors(list(pdf_overlap_by_source.keys()))
    all_classes = sorted({c for d in pdf_overlap_by_source.values() for c in d.keys() if c != 'mean'})

    fig = go.Figure()
    for name, overlaps in pdf_overlap_by_source.items():
        y = [overlaps.get(c, np.nan) for c in all_classes]
        fig.add_trace(go.Bar(x=[str(c) for c in all_classes], y=y, name=name, marker_color=colors[name]))
    fig.add_hline(y=0.75, line=dict(color='gray', dash='dot'), annotation_text='good overlap (0.75)')
    fig.update_layout(barmode='group')
    apply_default_plotly_layout(
        fig, title or 'PDF Overlap Score by concentration class',
        xaxis_title='Concentration (µM)', yaxis_title='PDF overlap score [0, 1]')
    return fig


def compute_vfi_components(gate_results: dict, vfi_calculator: Optional[VoltammogramFidelityIndex] = None) -> dict:
    """
    Break a `ValidationGate.run()` results dict into the 4 VFI penalty terms
    plus the final score. `VoltammogramFidelityIndex.compute` only returns
    the scalar VFI, so this re-applies its own clipping formulas (no new
    metric math - mean_jsd/swd_mean/delta_acf/mean_feat_w1 are already
    computed by the gate) to get the per-term breakdown needed for plotting.
    """
    calc = vfi_calculator or VoltammogramFidelityIndex()
    mean_jsd = gate_results.get('mean_jsd', 0.0)
    mean_swd = gate_results.get('swd_mean', 0.0)
    delta_acf = gate_results.get('delta_acf', 0.0)
    mean_feat_w1 = gate_results.get('mean_feat_w1', 0.0)

    p_jsd = float(np.clip(mean_jsd / calc.jsd_scale, 0.0, 1.0))
    p_swd = float(np.clip(mean_swd / calc.swd_scale, 0.0, 1.0))
    p_acf = float(np.clip(delta_acf / calc.acf_scale, 0.0, 1.0))
    p_feat = float(np.clip(mean_feat_w1 / calc.w1_scale, 0.0, 1.0))
    total_penalty = calc.w_jsd * p_jsd + calc.w_swd * p_swd + calc.w_acf * p_acf + calc.w_feat * p_feat
    vfi = float(np.clip(1.0 - total_penalty, 0.0, 1.0))

    return {'p_jsd': p_jsd, 'p_swd': p_swd, 'p_acf': p_acf, 'p_feat': p_feat, 'vfi': vfi}


def plot_vfi_breakdown_bars(vfi_components_by_source: dict, title: Optional[str] = None) -> go.Figure:
    """
    Stacked penalty bars per source with the resulting VFI marked - shows
    which distributional mismatch is eating into the VFI for each generator.

    vfi_components_by_source : {source_name: compute_vfi_components(...) dict}
    """
    terms = [('p_jsd', 'JSD penalty'), ('p_swd', 'SWD penalty'),
             ('p_acf', 'ACF penalty'), ('p_feat', 'Feature-W1 penalty')]
    term_colors = ['#e34a1a', '#f0ad4e', '#5a23c4', '#2ca02c']
    names = list(vfi_components_by_source.keys())

    fig = go.Figure()
    for (key, label), tcolor in zip(terms, term_colors):
        fig.add_trace(go.Bar(x=names, y=[vfi_components_by_source[n][key] for n in names],
                              name=label, marker_color=tcolor))
    fig.add_trace(go.Scatter(
        x=names, y=[vfi_components_by_source[n]['vfi'] for n in names],
        mode='markers+text', name='VFI', marker=dict(color='black', size=13, symbol='diamond'),
        text=[f"{vfi_components_by_source[n]['vfi']:.2f}" for n in names], textposition='top center'))
    fig.update_layout(barmode='stack')
    apply_default_plotly_layout(
        fig, title or 'VoltammogramFidelityIndex — penalty breakdown',
        xaxis_title='Source', yaxis_title='Penalty contribution (stacked) / VFI')
    return fig


def plot_fidelity_radar(vfi_components_by_source: dict, title: Optional[str] = None) -> go.Figure:
    """Polar view of (1 - penalty) fidelity on each VFI axis - a shape-based alternative to the stacked bars."""
    colors = _resolve_colors(list(vfi_components_by_source.keys()))
    axes = ['JSD', 'SWD', 'ACF', 'Feature-W1']
    keys = ['p_jsd', 'p_swd', 'p_acf', 'p_feat']

    fig = go.Figure()
    for name, comp in vfi_components_by_source.items():
        values = [1.0 - comp[k] for k in keys]
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=axes + axes[:1], fill='toself',
            name=name, line=dict(color=colors[name]), opacity=0.65))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], visible=True)))
    apply_default_plotly_layout(fig, title or 'Per-axis fidelity (1 − penalty), by source',
                                 xaxis_title=None, yaxis_title=None)
    return fig


def plot_jsd_per_class(jsd_maps_by_source: dict, title: Optional[str] = None) -> go.Figure:
    """
    Grouped bar of `compute_per_class_jsd(...)` results per source - shows
    exactly which concentration is failing the Tier-1 JSD gate.

    jsd_maps_by_source : {source_name: {concentration: jsd_value}}
        (pass `ValidationGate.run()`'s `results['jsd_map']` per source)
    """
    colors = _resolve_colors(list(jsd_maps_by_source.keys()))
    all_classes = sorted({c for d in jsd_maps_by_source.values() for c in d.keys()})

    fig = go.Figure()
    for name, jsd_map in jsd_maps_by_source.items():
        y = [jsd_map.get(c, np.nan) for c in all_classes]
        fig.add_trace(go.Bar(x=[str(c) for c in all_classes], y=y, name=name, marker_color=colors[name]))
    fig.add_hline(y=0.15, line=dict(color='gray', dash='dot'), annotation_text='gate threshold (mean < 0.15)')
    fig.update_layout(barmode='group')
    apply_default_plotly_layout(
        fig, title or 'Jensen-Shannon Divergence by concentration class',
        xaxis_title='Concentration (µM)', yaxis_title='JSD (bits)')
    return fig


def plot_metric_summary_bars(comparison_df, metrics: Sequence[str] = ('JSD_mean', 'MMD2', 'SWD_mean'),
                              title: Optional[str] = None) -> go.Figure:
    """
    Bar panel straight from `validation_metrics.quick_compare(...)`'s output
    DataFrame (columns: batch, KS_mean_frac, JSD_mean, MMD2, SWD_mean,
    PFF_class_frac, RS_R2_synth, delta_ACF, ...) - one panel per metric.
    """
    colors = _resolve_colors(list(comparison_df['batch']))
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=list(metrics), horizontal_spacing=0.08)
    for col, metric in enumerate(metrics, 1):
        fig.add_trace(go.Bar(
            x=comparison_df['batch'], y=comparison_df[metric],
            marker_color=[colors[b] for b in comparison_df['batch']], showlegend=False), row=1, col=col)
    apply_default_plotly_layout(fig, title or 'Validation-gate metric comparison',
                                 xaxis_title=None, yaxis_title=None)
    fig.update_layout(width=max(PLOTLY_FIGURE_WIDTH_PX, 320 * len(metrics)))
    return fig
