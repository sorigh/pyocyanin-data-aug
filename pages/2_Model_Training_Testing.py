"""Streamlit page: train a regressor on exactly the data/feature suite the
user picks (Module A), then test it transparently against a whole dataset or
a single hand-tuned signal (Module B).

Trained model "bundles" (models + imputer + feature columns + a record of
what they were trained on) live in `st.session_state['trained_runs']`, keyed
by run id, plus an always-present `'pretrained_baseline'` entry loaded from
`models/*.joblib` (built once, offline, by `export_models.py`) when that
directory exists. Nothing here retrains the baseline bundle or overwrites
its files - "training" always produces a new in-session bundle.

Datasets to train or test on come from `data_registry.get_available_sources()`,
which is exactly what `pages/1_Data_Generation.py` writes into when the user
clicks "Save this dataset for Model Training".
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import pdist, squareform

import data_registry as dr
from augmentation_pipeline import AugmentationConfig, FEATURE_COLUMNS_BY_SUITE, FeatureVectorizer, SignalAugmentor


from model_registry import MODEL_LABELS, RAW_SIGNAL_MODELS, predict, train_models
from plot_style import apply_default_plotly_layout
from voltammogram_signal import Signal

PRETRAINED_MODELS_DIR = 'models'
RECOMMENDED = AugmentationConfig()
FOOLED_PCT_THRESHOLD = 40.0
BORDERLINE_PCT_THRESHOLD = 15.0
LEAKAGE_NN_SAMPLE_CAP = 800  # bound the O(n^2) leave-one-out distance computation


# Section 1 - pretrained baseline bundle (models/*.joblib, from export_models.py)
#
# A bundle's models don't all share one feature representation any more:
# every model in MODEL_LABELS except 'cnn' consumes an engineered feature
# suite (core/extended/experimental), while 'cnn' (RAW_SIGNAL_MODELS) consumes
# the raw 229-point signal instead. So `feature_columns`, `imputers` and
# `train_features` are all dicts keyed by suite name, and `model_suites`
# says which suite each model in the bundle actually uses -
# `bundle['feature_columns'][bundle['model_suites'][model_name]]` is the
# pattern every call site below uses to get "the right suite for this model".
@st.cache_resource(show_spinner=False)
def load_pretrained_model(name: str):
    return joblib.load(f'{PRETRAINED_MODELS_DIR}/{name}.joblib')


@st.cache_resource(show_spinner=False)
def load_pretrained_imputer(suite: str):
    filename = 'raw_signal_imputer' if suite == 'raw_signal' else 'feature_imputer'
    return joblib.load(f'{PRETRAINED_MODELS_DIR}/{filename}.joblib')


@st.cache_data(show_spinner=False)
def load_pretrained_feature_metadata() -> dict:
    with open(f'{PRETRAINED_MODELS_DIR}/feature_metadata.json') as f:
        return json.load(f)


def build_pretrained_baseline_bundle(sources: dict) -> dict | None:
    try:
        meta = load_pretrained_feature_metadata()
        models = {name: load_pretrained_model(name) for name in MODEL_LABELS}
    except FileNotFoundError:
        return None

    model_suites = meta['model_suites']
    feature_columns = meta['feature_columns']
    suites_in_use = sorted(set(model_suites.values()))
    imputers = {suite: load_pretrained_imputer(suite) for suite in suites_in_use}

    real = sources['real']
    train_features = {suite: dr.featurize(real['E'], real['X'], real['y'], suite=suite)
                       for suite in suites_in_use}

    return dict(
        run_id='pretrained_baseline', models=models, model_suites=model_suites,
        feature_columns=feature_columns, imputers=imputers, train_features=train_features,
        train_source_keys=['real'], train_source_labels=[real['label']],
        n_train=real['n'], trained_at='pre-exported (export_models.py)')


def init_trained_runs(sources: dict) -> None:
    if 'trained_runs' in st.session_state:
        return
    st.session_state['trained_runs'] = {}
    baseline = build_pretrained_baseline_bundle(sources)
    if baseline is not None:
        st.session_state['trained_runs']['pretrained_baseline'] = baseline
        st.session_state['active_run_id'] = 'pretrained_baseline'


# Section 2 - Module A: in-depth model selection & training
DATASET_SOURCE_KEYS = ['real', 'stable_augmented', 'custom_generated', 'gan_generated']


def render_training_module(sources: dict) -> None:
    st.header('Module A · Model Selection & Training')
    st.caption('Pick exactly what data and which feature suite the model should learn from, then '
               'train new model(s) in-session. The pre-exported baseline (real data only, '
               '`experimental` suite) is always available for comparison.')

    selected_keys = st.multiselect(
        'Training data', options=DATASET_SOURCE_KEYS, default=['real'],
        format_func=lambda k: dr.format_source_option(k, sources),
        help='Select more than one source to train on their combined union - selecting all four is '
             '"all combined". Sources not yet generated on the Data Generation page are greyed out.')

    suite = st.selectbox(
        'Feature suite', options=['core', 'extended', 'experimental'], index=0,
        help='core: peak current/potential/AUC/FWHM only (4 features). extended: + PCA1 and first/'
             'second derivative extrema (7). experimental: + 16 additional shape, statistical and '
             'spectral features (23). Defaults to core. Ignored by 1D-CNN, which always trains on '
             'the raw signal regardless of this choice.')

    model_keys = st.multiselect('Algorithms to train', options=list(MODEL_LABELS), default=list(MODEL_LABELS),
                                 format_func=lambda k: MODEL_LABELS[k])
    if any(k in RAW_SIGNAL_MODELS for k in model_keys):
        st.caption('1D-CNN ignores the feature suite above - it always trains on the raw 229-point signal.')

    unavailable = [k for k in selected_keys if not sources[k]['available']]
    if unavailable:
        st.warning('Not generated yet, so excluded from training: ' +
                   ', '.join(sources[k]['label'] for k in unavailable) +
                   ' - build them on the Data Generation page first.')

    if st.button('⚡ Train model(s)', type='primary', width='stretch'):
        usable_keys = [k for k in selected_keys if sources[k]['available']]
        if not usable_keys:
            st.error('Select at least one available training data source.')
        elif not model_keys:
            st.error('Select at least one algorithm to train.')
        else:
            with st.spinner('Featurizing and training...'):
                E, X, y = dr.combine_sources(usable_keys, sources)
                feat_df = dr.featurize(E, X, y, suite=suite)
                feature_columns = FEATURE_COLUMNS_BY_SUITE[suite]
                X_feat, y_feat = feat_df[feature_columns], feat_df['concentration']

                needs_raw_signal = any(k in RAW_SIGNAL_MODELS for k in model_keys)
                raw_feat_df, X_raw_feat = None, None
                if needs_raw_signal:
                    raw_feat_df = dr.featurize(E, X, y, suite='raw_signal')
                    X_raw_feat = raw_feat_df[FEATURE_COLUMNS_BY_SUITE['raw_signal']]

                models, imputer, raw_signal_imputer = train_models(
                    X_feat, y_feat, model_keys, X_raw_signal=X_raw_feat)

            model_suites = {k: ('raw_signal' if k in RAW_SIGNAL_MODELS else suite) for k in model_keys}
            feature_columns_by_suite = {suite: feature_columns}
            imputers = {suite: imputer}
            train_features = {suite: feat_df}
            if needs_raw_signal:
                feature_columns_by_suite['raw_signal'] = FEATURE_COLUMNS_BY_SUITE['raw_signal']
                imputers['raw_signal'] = raw_signal_imputer
                train_features['raw_signal'] = raw_feat_df

            run_id = f'run_{len(st.session_state["trained_runs"])}'
            st.session_state['trained_runs'][run_id] = dict(
                run_id=run_id, models=models, model_suites=model_suites,
                feature_columns=feature_columns_by_suite, imputers=imputers, train_features=train_features,
                train_source_keys=usable_keys,
                train_source_labels=[sources[k]['label'] for k in usable_keys],
                n_train=len(y_feat),
                trained_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            st.session_state['active_run_id'] = run_id
            st.success(f'Trained {len(models)} model(s) on {len(y_feat)} samples '
                       f'({", ".join(sources[k]["label"] for k in usable_keys)}), {suite} suite'
                       + (' + raw signal for 1D-CNN' if needs_raw_signal else '') + '.')

    runs = st.session_state['trained_runs']
    if len(runs) > 1:
        st.caption('Bundles trained this session:')
        rows = [dict(run=rid, trained_at=b['trained_at'],
                      suite=', '.join(sorted(set(b['model_suites'].values()))), n_train=b['n_train'],
                      trained_on=', '.join(b['train_source_labels']))
                for rid, b in runs.items()]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# Section 3 - Module B: testing & transparency
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return dict(
        MAE=float(np.mean(np.abs(err))),
        MAPE=float(np.mean(np.abs(err) / y_true) * 100),
        R2=1 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
    )


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> go.Figure:
    lo = max(0.05, min(y_true.min(), y_pred.min()) * 0.8)
    hi = max(y_true.max(), y_pred.max()) * 1.2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='Perfect prediction',
                              line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions',
                              marker=dict(size=8, color='#17becf')))
    fig = apply_default_plotly_layout(fig, title_text=title, xaxis_title='Expected concentration (uM)',
                                       yaxis_title='Predicted concentration (uM)')
    fig.update_xaxes(type='log')
    fig.update_yaxes(type='log')
    return fig


def plot_history_parity(history_df: pd.DataFrame) -> go.Figure:
    lims = [0.05, 150]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines', name='Perfect prediction',
                              line=dict(color='gray', dash='dash')))
    palette = ['#17becf', '#e34a1a', '#5a23c4']
    for (model_name, sub), color in zip(history_df.groupby('model'), palette):
        fig.add_trace(go.Scatter(x=sub['expected'], y=sub['predicted'], mode='markers', name=model_name,
                                  marker=dict(size=10, color=color)))
    fig = apply_default_plotly_layout(fig, title_text='Predicted vs. expected concentration (this session)',
                                       xaxis_title='Expected concentration (uM)',
                                       yaxis_title='Predicted concentration (uM)')
    fig.update_xaxes(type='log')
    fig.update_yaxes(type='log')
    return fig


def render_transparency_card(bundle: dict, model_name: str) -> None:
    st.subheader('What was this model trained on?')
    suite = bundle['model_suites'][model_name]
    cols = st.columns(4)
    cols[0].metric('Algorithm', MODEL_LABELS.get(model_name, model_name))
    cols[1].metric('Feature suite', suite)
    cols[2].metric('Training samples', bundle['n_train'])
    cols[3].metric('# features', len(bundle['feature_columns'][suite]))
    st.caption('Trained on: ' + ', '.join(bundle['train_source_labels']))


def render_dataset_test(bundle: dict, model_name: str, sources: dict) -> None:
    st.subheader('Test on an entire dataset')
    available_keys = [k for k in DATASET_SOURCE_KEYS if sources[k]['available']]
    test_key = st.selectbox('Dataset to test', options=available_keys,
                             format_func=lambda k: dr.format_source_option(k, sources), key='test_dataset_key')

    if test_key in bundle['train_source_keys']:
        st.warning(f"⚠️ **Data leakage:** {sources[test_key]['label']} was part of this model's "
                   'training data. The metrics below reflect memorization, not generalization - pick '
                   'an independent dataset, or train a bundle that excludes it, for an honest read.')

    if st.button('Run test on this dataset', type='primary'):
        s = sources[test_key]
        suite = bundle['model_suites'][model_name]
        feature_columns = bundle['feature_columns'][suite]
        imputer = bundle['imputers'][suite]
        feat_df = dr.featurize(s['E'], s['X'], s['y'], suite=suite)
        X_feat = feat_df[feature_columns]
        y_true = feat_df['concentration'].to_numpy(dtype=float)
        y_pred = predict(bundle['models'][model_name], imputer, X_feat)
        st.session_state['dataset_test_result'] = dict(
            test_key=test_key, y_true=y_true, y_pred=y_pred,
            metrics=compute_regression_metrics(y_true, y_pred), feat_df=feat_df)

    result = st.session_state.get('dataset_test_result')
    if result:
        m = result['metrics']
        cols = st.columns(3)
        cols[0].metric('MAE (uM)', f"{m['MAE']:.3f}")
        cols[1].metric('MAPE', f"{m['MAPE']:.1f}%")
        cols[2].metric('R²', f"{m['R2']:.3f}")
        title = f"{MODEL_LABELS.get(model_name, model_name)} on {sources[result['test_key']]['label']}"
        st.plotly_chart(plot_parity(result['y_true'], result['y_pred'], title), width='stretch')
        st.caption('Features the model evaluated (first 10 of %d rows):' % len(result['feat_df']))
        st.dataframe(result['feat_df'].head(10), width='stretch', hide_index=True)


def render_manual_augmentation_controls() -> tuple[float, AugmentationConfig, bool]:
    st.caption('Manually tune the same physics-informed augmentation ingredients used on the Data '
               'Generation page - one signal at a time.')
    conc_grid = [round(float(c), 3) for c in np.geomspace(0.1, 100.0, 60)]
    default_c = min(conc_grid, key=lambda c: abs(c - 10.0))
    target_c = st.select_slider('Target concentration (uM) - the "expected" value',
                                 options=conc_grid, value=default_c, key='manual_target_c')

    col1, col2 = st.columns(2)
    with col1:
        use_pchip = st.checkbox('PCHIP-corrected interpolation (vs linear)',
                                 value=RECOMMENDED.use_pchip, key='m_use_pchip')
        use_lw_noise = st.checkbox('Long & Winefordner noise model (vs constant sigma)',
                                    value=RECOMMENDED.use_lw_noise, key='m_use_lw_noise')
        noise_sigma_const_uA = st.slider(
            'Constant noise sigma (uA) - used only when L&W is off',
            min_value=0.0, max_value=0.02, value=RECOMMENDED.noise_sigma_const_uA,
            step=0.0005, format='%.4f', key='m_noise_sigma', disabled=use_lw_noise)
        snr_floor = st.slider('SNR floor', min_value=1.0, max_value=10.0,
                               value=RECOMMENDED.snr_floor, step=0.1, key='m_snr_floor',
                               help='Caps how noisy the peak region is allowed to get: peak current / '
                                    'noise never drops below this ratio.')
    with col2:
        enable_baseline = st.checkbox('Concentration-scaled baseline distortion',
                                       value=RECOMMENDED.enable_baseline, key='m_enable_baseline')
        baseline_amp_max_uA = st.slider(
            'Max baseline amplitude (uA)', min_value=0.0, max_value=0.15,
            value=RECOMMENDED.baseline_amp_max_uA, step=0.005, format='%.3f',
            key='m_baseline_amp', disabled=not enable_baseline)
        enable_drift = st.checkbox('Horizontal potential drift',
                                    value=RECOMMENDED.enable_drift, key='m_enable_drift')
        drift_sigma_V = st.slider(
            'Drift sigma (V)', min_value=0.0, max_value=0.02,
            value=RECOMMENDED.potential_drift_sigma_low_V, step=0.001, format='%.3f',
            key='m_drift_sigma', disabled=not enable_drift)

    seed = st.number_input('Random seed', min_value=0, max_value=10_000, value=42, step=1, key='m_seed')
    generate_clicked = st.button('⚡ Generate / regenerate test signal', type='primary', width='stretch')

    config = AugmentationConfig(
        use_pchip=use_pchip, use_lw_noise=use_lw_noise, noise_sigma_const_uA=noise_sigma_const_uA,
        snr_floor=snr_floor, enable_baseline=enable_baseline, baseline_amp_max_uA=baseline_amp_max_uA,
        baseline_scale_c_ref_uM=RECOMMENDED.baseline_scale_c_ref_uM, enable_drift=enable_drift,
        potential_drift_sigma_low_V=drift_sigma_V, potential_drift_sigma_high_V=drift_sigma_V,
        rng_seed=int(seed),
    )
    return target_c, config, generate_clicked


def generate_and_extract(target_c: float, config: AugmentationConfig, augmentor, suite: str
                          ) -> tuple[np.ndarray, Signal, list]:
    np.random.seed(config.rng_seed)
    augmentor.noise_model.snr_floor = config.snr_floor
    signal_augmentor = SignalAugmentor(augmentor.calibration, augmentor.ip_spline, augmentor.noise_model, config)
    I = signal_augmentor.generate_signal(target_c)

    Signal.set_common_potential_E(augmentor.calibration.potential_grid_V)
    Signal.set_common_baseline_I(np.array([]))  # anchor curves (and thus I) are already baseline-subtracted
    try:
        sig = Signal(I)  # still needed for the peak plot regardless of suite
        if suite == 'raw_signal':
            # The 1D-CNN trains directly on raw/raw_signals_real.csv's values
            # (see data_registry.featurize's 'raw_signal' branch) - Signal's
            # constructor Savitzky-Golay-smooths self.I for peak detection,
            # so using that here instead of the true raw I would be a
            # train/inference mismatch.
            feature_vector = I.tolist()
        else:
            feature_vector = FeatureVectorizer.vectorize(sig, suite=suite)
    finally:
        Signal.set_common_baseline_I(augmentor.calibration.blank_baseline_uA)
    return I, sig, feature_vector


def plot_generated_signal(E: np.ndarray, I: np.ndarray, peak) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E, y=I, mode='lines', name='Generated signal',
                              line=dict(color='#111111', width=2)))
    fig.add_trace(go.Scatter(
        x=E[peak.start_idx:peak.end_idx + 1], y=I[peak.start_idx:peak.end_idx + 1],
        mode='lines', name='Peak region', fill='tozeroy',
        line=dict(color='#17becf'), fillcolor='rgba(23,190,207,0.25)'))
    fig.add_trace(go.Scatter(x=[peak.Ep], y=[peak.Ip], mode='markers', name='Peak (Ip, Ep)',
                              marker=dict(color='#e34a1a', size=11, symbol='diamond')))
    return apply_default_plotly_layout(fig, title_text='Manually augmented test signal')


def check_single_signal_leakage(feature_vector: list, target_c: float, bundle: dict, model_name: str
                                 ) -> tuple[bool, str]:
    """Nearest-neighbour-in-feature-space + exact-concentration leakage check.

    Flags the manually generated signal as a likely leak if either (a) its
    target concentration exactly matches one already in the training set, or
    (b) it sits closer to some training row than that training set's own
    typical (25th-percentile) nearest-neighbour distance - i.e. it looks like
    an unremarkable member of the training set rather than an independent probe.

    Runs in whichever representation `model_name` actually uses (engineered
    features, or the raw 229-point signal for 'cnn') - the z-scored distance
    check is dimension-agnostic, so no special-casing needed beyond picking
    the right suite's train_features/feature_columns.
    """
    suite = bundle['model_suites'][model_name]
    train_features = bundle['train_features'][suite]
    feature_columns = bundle['feature_columns'][suite]
    X_train = train_features[feature_columns].to_numpy(dtype=float)

    concentrations = train_features['concentration'].to_numpy(dtype=float)
    if np.any(np.isclose(concentrations, target_c, rtol=1e-6)):
        return True, f'the target concentration {target_c:g} uM is identical to one already in the training set'

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xz = (X_train - mu) / sigma
    vz = (np.asarray(feature_vector, dtype=float) - mu) / sigma
    min_dist = float(np.linalg.norm(Xz - vz, axis=1).min())

    if len(Xz) < 2:
        return False, ''

    sample = Xz
    if len(Xz) > LEAKAGE_NN_SAMPLE_CAP:
        rng = np.random.RandomState(0)
        sample = Xz[rng.choice(len(Xz), LEAKAGE_NN_SAMPLE_CAP, replace=False)]
    D = squareform(pdist(sample))
    np.fill_diagonal(D, np.inf)
    threshold = float(np.percentile(D.min(axis=1), 25))

    if min_dist <= threshold:
        return True, (f"this signal's features sit closer to a training example (z-distance {min_dist:.2f}) "
                       f'than training examples typically sit to their own nearest neighbour '
                       f'(threshold {threshold:.2f})')
    return False, ''


def render_single_signal_test(bundle: dict, model_name: str) -> None:
    st.subheader('Generate a single new signal')
    augmentor = dr.load_augmentor()
    target_c, config, generate_clicked = render_manual_augmentation_controls()

    suite = bundle['model_suites'][model_name]
    feature_columns = bundle['feature_columns'][suite]
    imputer = bundle['imputers'][suite]

    state_key = f"single_signal_{bundle['run_id']}_{suite}"
    if generate_clicked or state_key not in st.session_state:
        I, sig, feature_vector = generate_and_extract(target_c, config, augmentor, suite)
        st.session_state[state_key] = dict(E=augmentor.calibration.potential_grid_V, I=I, peak=sig.peak,
                                            target_c=target_c, feature_vector=feature_vector)
    last = st.session_state[state_key]

    X_row = pd.DataFrame([last['feature_vector']], columns=feature_columns)
    predicted_c = float(predict(bundle['models'][model_name], imputer, X_row)[0])

    is_leaky, reason = check_single_signal_leakage(last['feature_vector'], last['target_c'], bundle, model_name)
    if is_leaky:
        st.warning(f'⚠️ **Possible leakage:** {reason}. Tweak the target concentration or the noise / '
                   'baseline / drift settings further so this test signal is clearly independent from '
                   'the training data.')

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(plot_generated_signal(last['E'], last['I'], last['peak']), width='stretch')
    with right:
        abs_pct_error = 100 * abs(predicted_c - last['target_c']) / last['target_c']
        st.metric('Expected concentration (uM)', f"{last['target_c']:.3f}")
        st.metric(f"{MODEL_LABELS.get(model_name, model_name)} prediction (uM)", f'{predicted_c:.3f}',
                  delta=f'{predicted_c - last["target_c"]:+.3f} uM')
        st.metric('Absolute error', f'{abs_pct_error:.1f}%')

        if abs_pct_error < BORDERLINE_PCT_THRESHOLD:
            st.success('**Verdict:** model held up - prediction tracks the target closely.')
        elif abs_pct_error < FOOLED_PCT_THRESHOLD:
            st.warning('**Verdict:** borderline - noticeable drift from the target.')
        else:
            st.error('**Verdict:** model was fooled - prediction is far from the target.')

        if st.button('➕ Add to comparison history', width='stretch'):
            row = dict(model=MODEL_LABELS.get(model_name, model_name), expected=last['target_c'],
                       predicted=predicted_c, abs_pct_error=abs_pct_error)
            st.session_state.setdefault('test_history', []).append(row)

    st.caption('Raw signal the model is evaluating for this signal:' if suite == 'raw_signal' else
               'Features the model is evaluating for this signal:')
    st.dataframe(pd.DataFrame([last['feature_vector']], columns=feature_columns),
                 width='stretch', hide_index=True)

    if st.session_state.get('test_history'):
        st.subheader('Comparison history (this session)')
        hist_df = pd.DataFrame(st.session_state['test_history'])
        st.dataframe(hist_df, width='stretch', hide_index=True)
        st.plotly_chart(plot_history_parity(hist_df), width='stretch')
        if st.button('Clear history'):
            st.session_state['test_history'] = []
            st.rerun()


def render_testing_module(sources: dict) -> None:
    st.header('Module B · Testing & Transparency')
    runs = st.session_state.get('trained_runs', {})
    if not runs:
        st.info('Train a model in Module A above first.')
        return

    run_id = st.selectbox('Active model bundle', options=list(runs), key='active_run_id',
                           format_func=lambda rid: f"{rid} · trained {runs[rid]['trained_at']}")
    bundle = runs[run_id]
    model_name = st.selectbox('Algorithm', options=list(bundle['models']),
                               format_func=lambda k: MODEL_LABELS.get(k, k))

    render_transparency_card(bundle, model_name)

    test_mode = st.radio('Test mode', options=['Test an entire dataset', 'Generate a single new signal'],
                          horizontal=True, key='test_mode')
    if test_mode == 'Test an entire dataset':
        render_dataset_test(bundle, model_name, sources)
    else:
        render_single_signal_test(bundle, model_name)


# Section 4 - page entry point
def main() -> None:
    st.title('🧪 Model Training & Testing')
    st.caption('Choose exactly what data and feature suite a model learns from, then test it '
               'transparently against a whole dataset or a single hand-tuned signal.')

    sources = dr.get_available_sources()
    init_trained_runs(sources)

    render_training_module(sources)
    st.divider()
    render_testing_module(sources)


if __name__ == '__main__':
    main()
