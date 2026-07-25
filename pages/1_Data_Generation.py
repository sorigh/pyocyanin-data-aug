"""Streamlit page: tune the physics-informed augmentation pipeline, compare it
against the pre-trained TimeGAN / WGAN-GP synthetic batches, and generate
fresh GAN data on demand.

State management: every widget here only ever writes into
`st.session_state`; the heavy work (generation + the validation gate) runs
once inside a button handler and its results are cached in
`st.session_state['dashboard']`, so tweaking a display-only widget (which
concentration to inspect, which sources to plot) just re-renders Plotly
figures from already-computed arrays instead of re-running anything.

Datasets a user chooses to keep are kept in `st.session_state['datasets']` 
using `data_registry.save_dataset_to_session()`.
"Model Training & Testing" page reads from there for the dataset catalog.

All data engineering and plotting logic lives in `augmentation_pipeline.py`,
`gan_inference.py` and `visualization.py`
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import data_registry as dr
import gan_inference
import visualization as fa
from augmentation_pipeline import AugmentationConfig
from validation_metrics import PDFOverlapScore, ValidationGate

RECOMMENDED = AugmentationConfig()  # the dataclass defaults *are* the recommended settings


def _vfi_label(vfi: float) -> str:
    if vfi >= 0.90:
        return 'Excellent'
    if vfi >= 0.75:
        return 'Good'
    if vfi >= 0.60:
        return 'Marginal'
    return 'Poor'


# Section 1 - physics-informed control panel (rendered in-page, not the sidebar)
def recommended_slider(label: str, key: str, recommended: float, span: float,
                        step: float, fmt: str = '%.4f', help: str | None = None) -> float:
    """A slider with a centered track on `recommended` (recommended +/- span).

    `value=recommended` only seeds `st.session_state[key]` the first time the
    widget is created; on every later rerun (including after the Reset
    button sets `st.session_state[key]` directly) Streamlit reads the
    existing session-state value instead, so this stays a single source of
    truth for the widget's current value.
    """
    val = st.slider(label, min_value=recommended - span, max_value=recommended + span,
                     value=recommended, step=step, format=fmt, key=key, help=help)
    st.caption(f'● recommended = {recommended:g} (slider centre)')
    return val


def recommended_defaults() -> dict:
    return {f.name: getattr(RECOMMENDED, f.name) for f in dataclasses.fields(RECOMMENDED)}


def render_control_panel() -> tuple[AugmentationConfig, str, bool]:
    st.subheader('Physics-informed augmentation parameters')
    st.caption('Every slider below manipulates one ingredient of the synthetic-signal pipeline. '
               'Tune them, then click **Generate data** to rebuild the comparison dashboard.')

    top_left, top_right = st.columns([3, 1])
    with top_left:
        gan_source_label = st.selectbox(
            'TimeGAN / WGAN-GP checkpoint to compare against', options=list(dr.GAN_SAMPLE_DIRS.keys()),
            key='gan_source_label',
            help='Which pre-trained GAN checkpoint to overlay as a comparison. Training happens '
                 'offline via training/timegan_training.py and training/wgangp_training.py - this '
                 'loads their saved sample batches. To generate a brand-new GAN batch instead of '
                 'just comparing, use the "GAN Data Generation" tab.')
    with top_right:
        st.write('')
        if st.button('Reset all to recommended', width='stretch'):
            # Delete so the widgets below fall back to
            # their own `value=recommended` default on the next run - setting
            # `st.session_state[key]` directly would collide with that `value=`
            # argument and trigger Streamlit's "widget created with a default
            # value but also set via the Session State API" warning.
            for key in recommended_defaults():
                st.session_state.pop(key, None)
            st.rerun()

    with st.expander('Interpolation & instrument noise', expanded=True):
        st.caption('How a signal at an arbitrary concentration is built from the real calibration '
                   'curves, and how much instrument noise is layered on top.')
        col1, col2 = st.columns(2)
        with col1:
            use_pchip = st.checkbox(
                'Use PCHIP-corrected alpha (vs linear)', key='use_pchip', value=RECOMMENDED.use_pchip,
                help='Every synthetic signal is a blend of the two nearest real calibration curves. '
                     'PCHIP corrects the blend weight by peak current (not raw concentration), '
                     'matching the true Ip(c) saturation curve instead of overshooting between anchors.')
            use_lw_noise = st.checkbox(
                'Use Long & Winefordner noise model (vs constant sigma)', key='use_lw_noise',
                value=RECOMMENDED.use_lw_noise,
                help='Long & Winefordner (1983): instrument noise grows with peak current via '
                     'sigma(c)^2 = sigma_abs^2 + (RSD * Ip(c))^2, fit from the real replicate '
                     'variance. Turn off to use one constant sigma at every concentration instead.')
            noise_sigma_const_uA = st.number_input(
                'Constant sigma (uA) - used only when L&W noise is off', min_value=0.0, max_value=0.02,
                value=RECOMMENDED.noise_sigma_const_uA,
                step=0.0005, format='%.4f', key='noise_sigma_const_uA', disabled=use_lw_noise,
                help='A flat noise standard deviation applied everywhere, ignoring how peak current '
                     'scales with concentration. Only used when the L&W model above is off.')
        with col2:
            snr_floor = recommended_slider(
                'SNR floor', 'snr_floor', RECOMMENDED.snr_floor, span=2.0, step=0.1, fmt='%.1f',
                help='Caps how noisy the peak region is ever allowed to get: the L&W sigma is '
                     'clipped so peak current / noise never drops below this ratio. Lower the SNR '
                     'floor to make low-concentration signals harder to tell apart from noise; '
                     'raise it for cleaner signals across the whole 0.1-100 uM range.')

    with st.expander('Concentration-scaled baseline distortion', expanded=False):
        st.caption('A smooth, even-symmetric bump added to the baseline, whose amplitude scales '
                   'with concentration (via tanh) - mimics capacitive/background drift that grows '
                   'with analyte loading.')
        enable_baseline = st.checkbox('Enable baseline distortion', key='enable_baseline',
                                       value=RECOMMENDED.enable_baseline)
        baseline_amp_max_uA = recommended_slider(
            'Max baseline amplitude (uA)', 'baseline_amp_max_uA', RECOMMENDED.baseline_amp_max_uA,
            span=0.05, step=0.005, fmt='%.3f',
            help='Upper bound on the bump height at saturating concentration. Higher values make '
                 'the baseline wobble more, which can mask genuine peak-shape differences.')
        baseline_scale_c_ref_uM = recommended_slider(
            'tanh half-saturation concentration (uM)', 'baseline_scale_c_ref_uM',
            RECOMMENDED.baseline_scale_c_ref_uM, span=5.0, step=0.5, fmt='%.1f',
            help='The concentration at which the distortion amplitude reaches half its maximum. '
                 'Lower values make the distortion kick in (and saturate) at lower concentrations.')

    with st.expander('Horizontal potential drift', expanded=False):
        st.caption('Simulates electrode-conditioning drift by shifting the whole signal sideways '
                   'along the potential axis by a small random amount.')
        enable_drift = st.checkbox('Enable potential drift', key='enable_drift',
                                    value=RECOMMENDED.enable_drift)
        potential_drift_sigma_low_V = recommended_slider(
            'Drift sigma - low (V)', 'potential_drift_sigma_low_V',
            RECOMMENDED.potential_drift_sigma_low_V, span=0.005, step=0.001, fmt='%.3f',
            help='Lower bound of the per-signal drift-strength range. Each generated signal draws '
                 'its own drift sigma uniformly between the low and high values, then shifts by '
                 'N(0, sigma^2) volts.')
        potential_drift_sigma_high_V = recommended_slider(
            'Drift sigma - high (V)', 'potential_drift_sigma_high_V',
            RECOMMENDED.potential_drift_sigma_high_V, span=0.010, step=0.001, fmt='%.3f',
            help='Upper bound of the per-signal drift-strength range. Wider low-high gaps produce '
                 'a more heterogeneous batch of horizontal shifts.')

    with st.expander('Stratified concentration sampling', expanded=True):
        st.caption('How many synthetic signals to generate and how their target concentrations are '
                   'spread across the 0.1-100 uM range.')
        stratified = st.checkbox(
            'Log-stratified allocation (vs flat/uniform)', key='stratified', value=RECOMMENDED.stratified,
            help='When on, each log-decade between calibration anchors gets an equal share of '
                 'signals (plus the low-concentration boost below). When off, every anchor segment '
                 'gets the same flat count regardless of its width in log-space.')
        n_total_synthetic = int(recommended_slider(
            'Total synthetic signals', 'n_total_synthetic', RECOMMENDED.n_total_synthetic,
            span=200, step=10, fmt='%d',
            help='How many synthetic signals the batch will contain in total, split across '
                 'concentration segments according to the allocation rule above.'))
        low_conc_threshold_uM = recommended_slider(
            'Low-concentration threshold (uM)', 'low_conc_threshold_uM',
            RECOMMENDED.low_conc_threshold_uM, span=2.5, step=0.1, fmt='%.1f',
            help='Concentration segments whose midpoint falls below this threshold are treated as '
                 '"low-concentration" and receive the oversampling boost below.')
        low_conc_boost = recommended_slider(
            'Low-concentration oversampling boost', 'low_conc_boost',
            RECOMMENDED.low_conc_boost, span=2.0, step=0.1, fmt='%.1f',
            help='Multiplier applied to the sampling weight of low-concentration segments. Values '
                 '> 1 deliberately over-represent the hardest-to-detect signals in the batch, which '
                 'is usually what you want for training a regressor that must resolve low doses.')

    rng_seed = st.number_input('Random seed', min_value=0, max_value=10_000,
                                value=RECOMMENDED.rng_seed, step=1, key='rng_seed',
                                help='Reseeds numpy before generation - reuse the same seed to '
                                     'exactly reproduce a batch.')

    if potential_drift_sigma_low_V > potential_drift_sigma_high_V:
        st.warning('Drift sigma low > high - swapping them for generation.')
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

    generate_clicked = st.button('Generate data', type='primary', width='stretch')
    return config, gan_source_label, generate_clicked


# Section 2 - generation + validation gate (runs only on button click)
def run_generation(config: AugmentationConfig, gan_source_label: str) -> None:
    E, X_real, y_real = dr.load_real_batch()
    gan_dir = dr.GAN_SAMPLE_DIRS[gan_source_label]

    progress = st.progress(0, text='Starting...')
    with st.status('Building the comparison dashboard...', expanded=True) as status:
        status.write('Generating the physics-informed batch (augmentation_pipeline.py)...')
        progress.progress(15)
        augmentor = dr.load_augmentor()
        dataset = augmentor.generate(config)

        status.write(f'Loading pre-generated TimeGAN / WGAN-GP samples ({gan_source_label})...')
        progress.progress(35)
        X_timegan, y_timegan = dr.load_gan_sample_batch(gan_dir, 'timegan')
        X_wgangp, y_wgangp = dr.load_gan_sample_batch(gan_dir, 'wgangp')

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


# Section 3 - dashboard rendering (reads st.session_state only, never recomputes)
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

    grid_concs = st.multiselect('Concentrations (uM) - overlay grid', options=dr.REPRESENTATIVE_CONCENTRATIONS_uM,
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


def render_save_to_session(d: dict) -> None:
    st.divider()
    st.subheader('Use this dataset on the Model Training & Testing page')
    n = len(d['sources']['Physics-Augmented'][0])
    st.caption(f'Saves the **Physics-Augmented** batch above (n={n}, seed={d["config_used"].rng_seed}) into this '
               'session so it shows up as "Custom generated" in the training-data picker on the other page, or '
               'download it as a CSV. Generating a new batch here and saving again overwrites the previous save.')
    X, y = d['sources']['Physics-Augmented']
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Save this dataset for Model Training', type='primary', width='stretch'):
            dr.save_dataset_to_session(
                'custom_generated', d['E'], X, y,
                meta={'origin': 'Data Generation page (physics-informed)',
                      'config': dataclasses.asdict(d['config_used']),
                      'generated_at': datetime.now().isoformat(timespec='seconds')})
            st.success(f'Saved {n} signals as "Custom generated" - switch to the Model Training & Testing page to use them.')
    with col2:
        st.download_button(
            'Download dataset as CSV', data=dr.dataset_to_csv_bytes(X, y),
            file_name=f'physics_augmented_seed{d["config_used"].rng_seed}_n{n}.csv',
            mime='text/csv', width='stretch')


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
        ['Signal overlays', 'Feature space', 'PDF Overlap Index',
         'Voltammogram Fidelity Index', 'Raw gate metrics'])
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

    render_save_to_session(d)


# Section 4 - dedicated GAN data generation menu (full GAN integration)
def render_gan_generation_tab() -> None:
    st.subheader('Generate new data from a saved GAN checkpoint')
    st.caption('Unlike the comparison overlay in the other tab (which replays a fixed, pre-generated '
               'sample batch), this loads the saved generator weights and draws a fresh batch of any '
               'size on demand.')

    col1, col2 = st.columns(2)
    with col1:
        architecture = st.selectbox('Architecture', options=['WGAN-GP', 'TimeGAN'], key='gan_gen_architecture',
                                     help='WGAN-GP: 1-D convolutional generator trained with a Wasserstein '
                                          'critic + gradient penalty. TimeGAN: GRU-based generator trained '
                                          'in a learned temporal-embedding space for better joint dynamics.')
        checkpoint_label = st.selectbox('Checkpoint', options=list(dr.GAN_MODEL_DIRS.keys()), key='gan_gen_checkpoint',
                                         help='Which saved training run to draw from.')
    with col2:
        n_points = st.number_input('Number of signals to generate', min_value=1, max_value=5000, value=100,
                                    step=10, key='gan_gen_n_points')
        conc_min, conc_max = st.slider(
            'Concentration range to condition on (uM)', min_value=0.1, max_value=100.0, value=(0.1, 100.0),
            key='gan_gen_conc_range',
            help='The generator was trained with concentration conditioning over the full 0.1-100 uM '
                 'range; narrowing this range only changes which concentrations are sampled, not what '
                 'the model learned.')
        seed = st.number_input('Random seed', min_value=0, max_value=10_000, value=42, step=1, key='gan_gen_seed')

    generate_clicked = st.button('Generate GAN dataset', type='primary', width='stretch')

    if generate_clicked:
        model_dir = dr.GAN_MODEL_DIRS[checkpoint_label]
        with st.spinner(f'Loading the {architecture} generator and drawing {n_points} signals...'):
            if architecture == 'WGAN-GP':
                df_gen = gan_inference.generate_wgangp(model_dir, int(n_points), conc_min, conc_max, seed=int(seed))
            else:
                df_gen = gan_inference.generate_timegan(model_dir, int(n_points), conc_min, conc_max, seed=int(seed))
            
            X_gen, y_gen = gan_inference.dataframe_to_xy(df_gen)
            
            # Capture the real data instead of throwing it away
            E, X_real, y_real = dr.load_real_batch()
            
        # Store both sets explicitly to avoid confusion
        st.session_state['gan_preview'] = dict(
            E=E, 
            X_gen=X_gen, y_gen=y_gen, 
            X_real=X_real, y_real=y_real, 
            architecture=architecture,
            checkpoint_label=checkpoint_label, 
            n=int(n_points), 
            seed=int(seed)
        )

    preview = st.session_state.get('gan_preview')
    if preview:
        st.success(f'Generated {preview["n"]} {preview["architecture"]} signals ({preview["checkpoint_label"]}).')
        left, right = st.columns([2, 1])
        with left:
            fig = fa.plot_signal_overlay(
            preview['E'], 
            (preview['X_real'], preview['y_real']), 
            {preview['architecture']: (preview['X_gen'], preview['y_gen'])},
            concentration=None # stop filtering the signals, show all
            )
            st.plotly_chart(fig, width='stretch')
        with right:
            st.dataframe(pd.DataFrame({'concentration': preview['y_gen']}).describe(), width='stretch')

        st.caption('Saves this GAN batch into the session so it shows up as "GAN synthetic" in the '
                   'training-data picker on the Model Training & Testing page, or download it as a CSV. '
                   'Generating again and re-saving overwrites the previous save.')
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Save this GAN dataset for Model Training', type='primary', key='save_gan_dataset',
                         width='stretch'):
                dr.save_dataset_to_session(
                    'gan_generated', preview['E'], preview['X_gen'], preview['y_gen'],
                    meta={'origin': f'Data Generation page (GAN: {preview["architecture"]})',
                          'checkpoint': preview['checkpoint_label'], 'seed': preview['seed'],
                          'generated_at': datetime.now().isoformat(timespec='seconds')})
                st.success(f'Saved {preview["n"]} signals as "GAN synthetic" - switch to the Model Training '
                           '& Testing page to use them.')
        with col2:
            safe_checkpoint = preview['checkpoint_label'].lower().replace(' ', '_')
            st.download_button(
                'Download dataset as CSV', data=dr.dataset_to_csv_bytes(preview['X_gen'], preview['y_gen']),
                file_name=f'{preview["architecture"].lower()}_{safe_checkpoint}_n{preview["n"]}.csv',
                mime='text/csv', width='stretch', key='download_gan_dataset')


# Section 5 - page entry point
def main() -> None:
    st.title('Data Generation')
    st.caption('Tune the physics-informed augmentation pipeline and compare it against the '
               'pre-trained TimeGAN / WGAN-GP batches, or draw a fresh GAN batch on demand. '
               'Save either result to use it on the Model Training & Testing page.')

    tab_physics, tab_gan = st.tabs(['Physics-Informed Generation', 'GAN Data Generation'])

    with tab_physics:
        config, gan_source_label, generate_clicked = render_control_panel()

        if generate_clicked:
            run_generation(config, gan_source_label)

        if 'dashboard' in st.session_state:
            render_dashboard(st.session_state['dashboard'])
        else:
            st.info('Configure the augmentation parameters above, then click '
                    '**Generate data** to build the comparison dashboard.')

    with tab_gan:
        render_gan_generation_tab()


if __name__ == '__main__':
    main()
