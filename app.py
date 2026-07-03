"""Streamlit dashboard for tuning physics-informed augmentation and comparing
it against the pre-trained TimeGAN / WGAN-GP synthetic batches.

State management, in one sentence: the sidebar only ever writes into
`st.session_state`; the heavy work (generation + the validation gate) runs
once inside the "Generate Data" button handler and its results are cached in
`st.session_state['dashboard']`, so tweaking a display-only widget (which
concentration to inspect, which sources to plot) just re-renders Plotly
figures from already-computed arrays instead of re-running anything.

All data engineering and plotting logic lives in `augmentation_pipeline.py`
and `visualization.py` - this file only wires widgets to their functions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import streamlit as st

import visualization as fa
from augmentation_pipeline import AugmentationConfig, FullRangeDataAugmentor
from validation_metrics import PDFOverlapScore, ValidationGate

# Section 0 - static configuration
DATASET_PATH = 'datasets/Standard calibration in culture media_extended.xlsx'
POTENTIAL_GRID_PATH = 'raw/raw_potential_grid.csv'
REAL_SIGNALS_PATH = 'raw/raw_signals_real.csv'

# Replicate labels for the calibration workbook columns (same list used by
# full_range_data_augmentation.ipynb and data_analysis.ipynb).
CONCENTRATIONS_uM = [
    100, 100, 100, 50, 50, 50, 25, 25, 25, 15, 15, 15,
    10, 10, 10, 7.5, 7.5, 7.5, 5, 5, 5, 2.5, 2.5,
    1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.25, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1,
]

GAN_SOURCE_DIRS = {
    'Trained on Real data': 'gan_output/samples_trained_on_real',
    'Trained on Combined (real + physics-aug) data': 'gan_output/samples_trained_on_combined',
}

REPRESENTATIVE_CONCENTRATIONS_uM = [0.1, 0.5, 2.5, 10.0, 25.0, 75.0]

RECOMMENDED = AugmentationConfig()  # the dataclass defaults *are* the recommended settings


def _vfi_label(vfi: float) -> str:
    if vfi >= 0.90:
        return 'Excellent'
    if vfi >= 0.75:
        return 'Good'
    if vfi >= 0.60:
        return 'Marginal'
    return 'Poor'


# Section 1 - cached, expensive, rarely-changing resources
@st.cache_resource(show_spinner='Loading calibration workbook and fitting the noise model...')
def load_augmentor() -> FullRangeDataAugmentor:
    return FullRangeDataAugmentor.from_excel(DATASET_PATH, concentrations_uM=CONCENTRATIONS_uM)


@st.cache_data(show_spinner=False)
def load_real_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E = pd.read_csv(POTENTIAL_GRID_PATH).values.flatten()
    df = pd.read_csv(REAL_SIGNALS_PATH)
    y = df['concentration'].to_numpy()
    X = df.drop(columns=['concentration']).to_numpy()
    return E, X, y


@st.cache_data(show_spinner=False)
def load_gan_batch(gan_dir: str, architecture: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(f'{gan_dir}/{architecture}_signals.csv')
    return df.drop(columns=['concentration']).to_numpy(), df['concentration'].to_numpy()


# Section 2 - sidebar control panel
def recommended_slider(label: str, key: str, recommended: float, span: float,
                        step: float, fmt: str = '%.4f', help: str | None = None) -> float:
    """A slider whose track is centred on `recommended` (recommended +/- span).

    `value=recommended` only seeds `st.session_state[key]` the first time the
    widget is created; on every later rerun (including after the Reset
    button sets `st.session_state[key]` directly) Streamlit reads the
    existing session-state value instead, so this stays a single source of
    truth for the widget's current value.
    """
    val = st.sidebar.slider(label, min_value=recommended - span, max_value=recommended + span,
                             value=recommended, step=step, format=fmt, key=key, help=help)
    st.sidebar.caption(f'● recommended = {recommended:g} (slider centre)')
    return val


def recommended_defaults() -> dict:
    return {f.name: getattr(RECOMMENDED, f.name) for f in dataclasses.fields(RECOMMENDED)}


def render_control_panel() -> AugmentationConfig:
    st.sidebar.title('Control panel')

    st.sidebar.subheader('Comparison sources')
    gan_source_label = st.sidebar.selectbox(
        'TimeGAN / WGAN-GP checkpoint', options=list(GAN_SOURCE_DIRS.keys()), key='gan_source_label',
        help='Which pre-trained GAN checkpoint to compare against. Training happens offline '
             'via training/timegan_training.py and training/wgangp_training.py - this app '
             'loads their saved sample batches.')

    if st.sidebar.button('↺ Reset all to recommended', width='stretch'):
        # Delete (rather than overwrite) so the widgets below fall back to
        # their own `value=recommended` default on the next run - setting
        # `st.session_state[key]` directly would collide with that `value=`
        # argument and trigger Streamlit's "widget created with a default
        # value but also set via the Session State API" warning.
        for key in recommended_defaults():
            st.session_state.pop(key, None)
        st.rerun()

    st.sidebar.subheader('Physics-informed augmentation')

    with st.sidebar.expander('Interpolation & instrument noise', expanded=True):
        use_pchip = st.checkbox('Use PCHIP-corrected alpha (vs linear)', key='use_pchip',
                                 value=RECOMMENDED.use_pchip)
        use_lw_noise = st.checkbox('Use Long & Winefordner noise model (vs constant sigma)',
                                    key='use_lw_noise', value=RECOMMENDED.use_lw_noise)
        noise_sigma_const_uA = st.number_input(
            'Constant sigma (uA) - used only when L&W noise is off', min_value=0.0, max_value=0.02,
            value=RECOMMENDED.noise_sigma_const_uA,
            step=0.0005, format='%.4f', key='noise_sigma_const_uA', disabled=use_lw_noise)
        snr_floor = recommended_slider('SNR floor', 'snr_floor', RECOMMENDED.snr_floor,
                                        span=2.0, step=0.1, fmt='%.1f')

    with st.sidebar.expander('Concentration-scaled baseline distortion', expanded=False):
        enable_baseline = st.checkbox('Enable baseline distortion', key='enable_baseline',
                                       value=RECOMMENDED.enable_baseline)
        baseline_amp_max_uA = recommended_slider(
            'Max baseline amplitude (uA)', 'baseline_amp_max_uA', RECOMMENDED.baseline_amp_max_uA,
            span=0.05, step=0.005, fmt='%.3f')
        baseline_scale_c_ref_uM = recommended_slider(
            'tanh half-saturation concentration (uM)', 'baseline_scale_c_ref_uM',
            RECOMMENDED.baseline_scale_c_ref_uM, span=5.0, step=0.5, fmt='%.1f')

    with st.sidebar.expander('Horizontal potential drift', expanded=False):
        enable_drift = st.checkbox('Enable potential drift', key='enable_drift',
                                    value=RECOMMENDED.enable_drift)
        potential_drift_sigma_low_V = recommended_slider(
            'Drift sigma - low (V)', 'potential_drift_sigma_low_V',
            RECOMMENDED.potential_drift_sigma_low_V, span=0.005, step=0.001, fmt='%.3f')
        potential_drift_sigma_high_V = recommended_slider(
            'Drift sigma - high (V)', 'potential_drift_sigma_high_V',
            RECOMMENDED.potential_drift_sigma_high_V, span=0.010, step=0.001, fmt='%.3f')

    with st.sidebar.expander('Stratified concentration sampling', expanded=True):
        stratified = st.checkbox('Log-stratified allocation (vs flat/uniform)', key='stratified',
                                  value=RECOMMENDED.stratified)
        n_total_synthetic = int(recommended_slider(
            'Total synthetic signals', 'n_total_synthetic', RECOMMENDED.n_total_synthetic,
            span=200, step=10, fmt='%d'))
        low_conc_threshold_uM = recommended_slider(
            'Low-concentration threshold (uM)', 'low_conc_threshold_uM',
            RECOMMENDED.low_conc_threshold_uM, span=2.5, step=0.1, fmt='%.1f')
        low_conc_boost = recommended_slider(
            'Low-concentration oversampling boost', 'low_conc_boost',
            RECOMMENDED.low_conc_boost, span=2.0, step=0.1, fmt='%.1f')

    rng_seed = st.sidebar.number_input('Random seed', min_value=0, max_value=10_000,
                                        value=RECOMMENDED.rng_seed, step=1, key='rng_seed')

    if potential_drift_sigma_low_V > potential_drift_sigma_high_V:
        st.sidebar.warning('Drift sigma low > high - swapping them for generation.')
        potential_drift_sigma_low_V, potential_drift_sigma_high_V = (
            potential_drift_sigma_high_V, potential_drift_sigma_low_V)

    config = AugmentationConfig(
        use_pchip=use_pchip, use_lw_noise=use_lw_noise, noise_sigma_const_uA=noise_sigma_const_uA,
        snr_floor=snr_floor, enable_baseline=enable_baseline, baseline_amp_max_uA=baseline_amp_max_uA,
        baseline_scale_c_ref_uM=baseline_scale_c_ref_uM, enable_drift=enable_drift,
        potential_drift_sigma_low_V=potential_drift_sigma_low_V,
        potential_drift_sigma_high_V=potential_drift_sigma_high_V,
        n_total_synthetic=n_total_synthetic, stratified=stratified,
        low_conc_threshold_uM=low_conc_threshold_uM, low_conc_boost=low_conc_boost,
        rng_seed=int(rng_seed),
    )

    generate_clicked = st.sidebar.button('⚡ Generate data', type='primary', width='stretch')
    return config, gan_source_label, generate_clicked


# Section 3 - generation + validation gate (runs only on button click)
def run_generation(config: AugmentationConfig, gan_source_label: str) -> None:
    E, X_real, y_real = load_real_batch()
    gan_dir = GAN_SOURCE_DIRS[gan_source_label]

    progress = st.progress(0, text='Starting...')
    with st.status('Building the comparison dashboard...', expanded=True) as status:
        status.write('Generating the physics-informed batch (augmentation_pipeline.py)...')
        progress.progress(15)
        augmentor = load_augmentor()
        dataset = augmentor.generate(config)

        status.write(f'Loading pre-generated TimeGAN / WGAN-GP samples ({gan_source_label})...')
        progress.progress(35)
        X_timegan, y_timegan = load_gan_batch(gan_dir, 'timegan')
        X_wgangp, y_wgangp = load_gan_batch(gan_dir, 'wgangp')

        sources = {
            'Physics-Augmented': (dataset.signals_uA, dataset.concentrations_uM),
            'TimeGAN': (X_timegan, y_timegan),
            'WGAN-GP': (X_wgangp, y_wgangp),
        }

        status.write('Running the validation gate (JSD / MMD² / SWD / ACF, Tiers 1-2)...')
        progress.progress(60)
        gate = ValidationGate(E, X_real, y_real)
        gate_results = {name: gate.run(X, y, verbose=False, run_tiers=[1, 2])
                         for name, (X, y) in sources.items()}

        status.write('Scoring PDF Overlap and VoltammogramFidelityIndex...')
        progress.progress(85)
        pdf_scorer = PDFOverlapScore()
        pdf_overlap = {}
        for name, (X, y) in sources.items():
            overlaps = pdf_scorer.per_class(X_real, y_real, X, y, E)
            overlaps['mean'] = pdf_scorer.mean(overlaps)
            pdf_overlap[name] = overlaps
        vfi_components = {name: fa.compute_vfi_components(r) for name, r in gate_results.items()}

        progress.progress(100, text='Done')
        status.update(label='Dashboard ready', state='complete', expanded=False)
    progress.empty()

    st.session_state['dashboard'] = dict(
        E=E, real=(X_real, y_real), sources=sources, gate_results=gate_results,
        pdf_overlap=pdf_overlap, vfi_components=vfi_components,
        config_used=config, gan_source_used=gan_source_label, ip_spline=augmentor.ip_spline,
        anchor_concentrations=augmentor.calibration.anchor_concentrations_uM,
    )


# Section 4 - dashboard rendering (reads st.session_state only, never recomputes)
def render_overview(d: dict) -> None:
    st.subheader('Fidelity at a glance')
    st.caption('The two headline scores this dashboard is built around: '
               '**PDF Overlap Score** (Bhattacharyya intersection of Ip-normalised peak-current '
               'distributions) and **VoltammogramFidelityIndex** (penalty-weighted composite of '
               'JSD / SWD / ACF / feature-Wasserstein).')
    cols = st.columns(len(d['sources']))
    for col, name in zip(cols, d['sources']):
        r = d['gate_results'][name]
        with col:
            st.metric(f'{name} · VFI', f"{r['vfi']:.3f}", help=_vfi_label(r['vfi']))
            st.metric(f'{name} · PDF Overlap', f"{d['pdf_overlap'][name]['mean']:.3f}")
            st.caption(f"JSD {r['mean_jsd']:.3f} · MMD² {r['mmd2']:.4f} · SWD {r['swd_mean']:.3f} "
                       f"· ΔACF {r['delta_acf']:.4f}")


def render_overlay_tab(d: dict, active_sources: dict) -> None:
    anchors = d['anchor_concentrations']
    c1 = st.select_slider('Concentration (uM) - simple overlay / mean envelope', options=anchors,
                           value=10.0 if 10.0 in anchors else anchors[len(anchors) // 2],
                           key='overlay_conc')
    left, right = st.columns(2)
    with left:
        st.plotly_chart(fa.plot_signal_overlay(d['E'], d['real'], active_sources, concentration=c1),
                         width='stretch')
    with right:
        st.plotly_chart(fa.plot_mean_envelope_comparison(d['E'], d['real'], active_sources, concentration=c1),
                         width='stretch')

    grid_concs = st.multiselect('Concentrations (uM) - overlay grid', options=REPRESENTATIVE_CONCENTRATIONS_uM,
                                 default=[0.5, 10.0, 75.0], key='overlay_grid_concs')
    if grid_concs:
        st.plotly_chart(fa.plot_overlay_grid(d['E'], d['real'], active_sources, concentrations=sorted(grid_concs)),
                         width='stretch')
    else:
        st.info('Pick at least one concentration for the overlay grid.')


def render_feature_space_tab(d: dict, active_sources: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(fa.plot_ip_vs_concentration(d['E'], d['real'], active_sources, ip_spline=d['ip_spline']),
                         width='stretch')
    with right:
        st.plotly_chart(fa.plot_feature_pca_scatter(d['E'], d['real'], active_sources),
                         width='stretch')


def render_pdf_overlap_tab(d: dict, active_sources: dict) -> None:
    st.markdown(
        'The **PDF Overlap Score** is the Bhattacharyya intersection `∫ min(p(x), q(x)) dx` between '
        'the real and synthetic Ip-normalised peak-current histograms, at one concentration class. '
        '1.0 = identical distributions, 0.0 = disjoint. The shaded plot below is exactly that '
        'intersection integral, made visible.')
    explain_conc = st.select_slider('Concentration (uM) - PDF overlap explainer', options=d['anchor_concentrations'],
                                     value=10.0 if 10.0 in d['anchor_concentrations'] else d['anchor_concentrations'][0],
                                     key='pdf_explain_conc')
    explain_source = st.selectbox('Source', options=list(active_sources.keys()), key='pdf_explain_source')
    if explain_source:
        st.plotly_chart(
            fa.plot_pdf_overlap_explainer(d['E'], d['real'], active_sources[explain_source],
                                           concentration=explain_conc, source_name=explain_source),
            width='stretch')

    pdf_overlap_active = {name: d['pdf_overlap'][name] for name in active_sources}
    st.plotly_chart(fa.plot_pdf_overlap_by_class(pdf_overlap_active), width='stretch')


def render_vfi_tab(d: dict, active_sources: dict) -> None:
    st.markdown(
        'The **VoltammogramFidelityIndex** starts at 1.0 and subtracts four normalised penalties '
        '(JSD, SWD, ACF-drift, feature-Wasserstein) - the same idea as `difflib.SequenceMatcher.ratio()`, '
        'but for distributional mismatch instead of edit distance.')
    vfi_active = {name: d['vfi_components'][name] for name in active_sources}
    left, right = st.columns(2)
    with left:
        st.plotly_chart(fa.plot_vfi_breakdown_bars(vfi_active), width='stretch')
    with right:
        st.plotly_chart(fa.plot_fidelity_radar(vfi_active), width='stretch')


def render_raw_metrics_tab(d: dict, active_sources: dict) -> None:
    st.markdown('Per-class JSD and the gate\'s headline scalar metrics - what the VFI/PDF-overlap '
                'scores above are built from.')
    jsd_maps_active = {name: d['gate_results'][name]['jsd_map'] for name in active_sources}
    st.plotly_chart(fa.plot_jsd_per_class(jsd_maps_active), width='stretch')

    rows = []
    for name in active_sources:
        r = d['gate_results'][name]
        rows.append({
            'batch': name, 'tier1_pass': r.get('tier1_pass'), 'tier2_pass': r.get('tier2_pass'),
            'JSD_mean': round(r['mean_jsd'], 4), 'MMD2': round(r['mmd2'], 6), 'SWD_mean': round(r['swd_mean'], 4),
            'delta_ACF': round(r['delta_acf'], 5), 'VFI': round(r['vfi'], 4),
            'PDF_overlap_mean': round(d['pdf_overlap'][name]['mean'], 4),
        })
    comparison_df = pd.DataFrame(rows).sort_values('VFI', ascending=False)
    st.dataframe(comparison_df, width='stretch', hide_index=True)
    st.plotly_chart(fa.plot_metric_summary_bars(comparison_df, metrics=('JSD_mean', 'MMD2', 'SWD_mean')),
                     width='stretch')


def render_dashboard(d: dict) -> None:
    st.caption(
        f"Physics-Augmented: n={len(d['sources']['Physics-Augmented'][0])} · "
        f"GAN checkpoint: *{d['gan_source_used']}* · seed={d['config_used'].rng_seed}")

    render_overview(d)

    selected = st.multiselect('Sources to display in the plots below', options=list(d['sources'].keys()),
                               default=list(d['sources'].keys()), key='active_sources')
    active_sources = {name: d['sources'][name] for name in selected}
    if not active_sources:
        st.warning('Select at least one source to render the comparison plots.')
        return

    tab_overlay, tab_features, tab_pdf, tab_vfi, tab_raw = st.tabs(
        ['📈 Signal overlays', '🧬 Feature space', '🎯 PDF Overlap Index',
         '🏆 Voltammogram Fidelity Index', '📊 Raw gate metrics'])
    with tab_overlay:
        render_overlay_tab(d, active_sources)
    with tab_features:
        render_feature_space_tab(d, active_sources)
    with tab_pdf:
        render_pdf_overlap_tab(d, active_sources)
    with tab_vfi:
        render_vfi_tab(d, active_sources)
    with tab_raw:
        render_raw_metrics_tab(d, active_sources)


# Section 5 - page entry point
def main() -> None:
    st.set_page_config(page_title='Pyocyanin Synthetic Data Studio', layout='wide')
    st.title('Pyocyanin Synthetic Data Studio')
    st.caption('Tune the physics-informed augmentation pipeline and compare it against the '
               'pre-trained TimeGAN / WGAN-GP batches, side by side with the validation gate.')

    config, gan_source_label, generate_clicked = render_control_panel()

    if generate_clicked:
        run_generation(config, gan_source_label)

    if 'dashboard' in st.session_state:
        render_dashboard(st.session_state['dashboard'])
    else:
        st.info('Configure the augmentation parameters in the sidebar, then click '
                '**Generate data** to build the comparison dashboard.')


if __name__ == '__main__':
    main()
