"""Streamlit page: train a regressor on exactly the data/feature suite the
user picks (Module A), then test it against a whole dataset or
a single hand-tuned signal (Module B).

Trained model "bundles" (models + imputer + feature columns + a record of
what they were trained on) live in `st.session_state['trained_runs']`, keyed
by run id,+ `'pretrained_baseline'` entry loaded from
`models/*.joblib` (built by `export_models.py`) 

The session bundle doesn't erase the baseline models

Datasets to train or test on come from `data_registry.get_available_sources()`,
which is what `pages/1_Data_Generation.py` writes into when the user
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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, LeaveOneOut

import data_registry as dr
import sweep_registry
from augmentation_pipeline import FEATURE_COLUMNS_BY_SUITE


from model_registry import MODEL_FACTORIES, MODEL_LABELS, RAW_SIGNAL_MODELS, predict, train_models
from plot_style import apply_default_plotly_layout
from voltammogram_signal import Signal

import paths

PRETRAINED_MODELS_DIR = paths.REGRESSORS_DIR
FOOLED_PCT_THRESHOLD = 40.0
BORDERLINE_PCT_THRESHOLD = 15.0

LOO_MAX_N = 50  # true LeaveOneOut only below this row count
EXPENSIVE_MODELS = RAW_SIGNAL_MODELS | {'mlp'}  # cnn + mlp: never true LOO, always k-fold


# Section 1: pretrained baseline bundle (models/*.joblib, from export_models.py)
#
# every model in MODEL_LABELS except 'cnn' consumes a feature
# suite (core/extended/experimental), 'cnn' consumes raw points.
# `feature_columns`, `imputers` and `train_features` are 
# dicts with keys by suite name, and `model_suites`
# says which suite each model in the bundle uses.
# pattern to get right suite:
# `bundle['feature_columns'][bundle['model_suites'][model_name]]`
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


# Section 2 - Module A: in-depth model selection and training
DATASET_SOURCE_KEYS = ['real', 'stable_augmented', 'stable_wgangp', 'stable_timegan', 'custom_generated', 'gan_generated']


def _model_suite_for(model_key: str, mode: str, default_suite: str, best_models_df: "pd.DataFrame | None") -> str:
    """ The feature suite for a model based on the execution mode.

    Behavior logic:
    - RAW_SIGNAL_MODELS always use 'raw_signal', regardless of mode.
    - 'sweep' mode: looks up the optimal suite for this specific model from `best_models_df`.
    - 'default' mode: Falls back to `default_suite` for all standard models.
    """
    if model_key in RAW_SIGNAL_MODELS:
        return 'raw_signal'
    if mode == 'sweep':
        return str(best_models_df.set_index('model').loc[model_key, 'suite'])
    return default_suite


def fit_training_bundle(model_keys: list[str], mode: str, default_suite: str, sources_pool: tuple,
                         sweep_condition: "str | None" = None, best_models_df: "pd.DataFrame | None" = None
                         ) -> dict:
    """Featurize once per feature suite needed across `model_keys`,
    then fit every requested model (from model_registry.MODEL_FACTORIES
    or from the condition sweep's tuned hyperparameters
    (mode='sweep', sweep_registry.build_sweep_model). 
    
    model_registry.train_models()'s convention of fitting each model on
    its raw feature frame and keeping a median imputer.

    Returns the dict of fields render_training_module() stores as one
    'trained_runs' bundle entry (everything except run_id/trained_at).
    """

    E, X, y = sources_pool
    model_suites = {k: _model_suite_for(k, mode, default_suite, best_models_df) for k in model_keys}
    suites_needed = sorted(set(model_suites.values()))

    train_features = {s: dr.featurize(E, X, y, suite=s) for s in suites_needed}
    feature_columns_by_suite = {s: FEATURE_COLUMNS_BY_SUITE[s] for s in suites_needed}
    imputers = {s: SimpleImputer(strategy='median').fit(train_features[s][feature_columns_by_suite[s]])
                for s in suites_needed}

    models: dict = {}
    sweep_model_meta: dict = {}
    for key in model_keys:
        suite = model_suites[key]
        X_feat = train_features[suite][feature_columns_by_suite[suite]]
        y_feat = train_features[suite]['concentration']
        if mode == 'sweep':
            row = best_models_df.set_index('model').loc[key]
            model_type = 'cnn' if key == 'cnn' else ('mlp' if key == 'mlp' else 'ml')
            params_entry = sweep_registry.load_sweep_params(sweep_condition, key, suite, row['search_method'])
            model = sweep_registry.build_sweep_model(key, model_type, params_entry)
            sweep_model_meta[key] = dict(suite=suite, search_method=str(row['search_method']),
                                          model_type=model_type, nested_cv_mae=float(row['mae']))
        else:
            model = MODEL_FACTORIES[key]()
        model.fit(X_feat, y_feat)
        models[key] = model

    n_train = len(train_features[model_suites[model_keys[0]]])
    bundle = dict(models=models, model_suites=model_suites, feature_columns=feature_columns_by_suite,
                  imputers=imputers, train_features=train_features, n_train=n_train,
                  hyperparam_source=mode)
    if mode == 'sweep':
        bundle['sweep_condition'] = sweep_condition
        bundle['sweep_model_meta'] = sweep_model_meta
    return bundle


def render_training_module(sources: dict) -> None:
    st.header('Module A : Model Selection & Training')
    st.caption('Pick exactly what data and which feature suite the model should learn from, then '
               'train new model(s) in-session. The pre-exported baseline (real data only, '
               '`experimental` suite) is always available for comparison.')

    available_conditions = sweep_registry.list_conditions_available()
    hp_mode = 'default'
    sweep_condition = None
    best_models_df = None
    if available_conditions:
        hp_mode = st.radio(
            'Hyperparameters', options=['default', 'sweep'], horizontal=True,
            format_func=lambda m: 'Default (fixed)' if m == 'default' else 'Sweep-tuned (per condition)',
            help="'Sweep-tuned' loads the per-condition hyperparameters found by the lab/physics-aug/"
                 'GAN/combined training sweep (run_condition_sweep.py) - each model uses whichever '
                 'feature suite and search method the sweep found best for it.')

    if hp_mode == 'sweep':
        sweep_condition = st.selectbox(
            'Sweep condition', options=available_conditions,
            format_func=lambda c: sweep_registry.CONDITION_LABELS.get(c, c),
            help='Also pins the training data to exactly what this condition was tuned on - see the '
                 'locked selection below.')
        locked_keys = sweep_registry.CONDITION_SOURCE_KEYS[sweep_condition]
        selected_keys = st.multiselect(
            'Training data', options=locked_keys, default=locked_keys, disabled=True,
            format_func=lambda k: dr.format_source_option(k, sources),
            help='Locked to this condition\'s own training data - sweep-tuned hyperparameters are only '
                 'meaningful for the data they were actually tuned on.')
        suite = None
        st.caption('Feature suite: chosen per model by the sweep (see table below) - the suite picker '
                   'is not used in this mode.')
    else:
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

    # xgboost_tuned is a fixed ablation-study baseline (model_registry.py), never
    # produced by run_condition_sweep.py's ML_MODELS list
    model_options = [k for k in MODEL_LABELS if not (hp_mode == 'sweep' and k == 'xgboost_tuned')]
    model_keys = st.multiselect('Algorithms to train', options=model_options, default=['ridge'],
                                 format_func=lambda k: MODEL_LABELS[k])
    if any(k in RAW_SIGNAL_MODELS for k in model_keys):
        st.caption('1D-CNN ignores the feature suite above - it always trains on the raw 229-point signal.')

    if hp_mode == 'sweep' and sweep_condition and model_keys:
        best_models_df = sweep_registry.load_best_models_table(sweep_condition)
        preview = best_models_df[best_models_df['model'].isin(model_keys)][
            ['model', 'suite', 'search_method', 'mae', 'n_samples']]
        st.caption('Hyperparameters that will be used (sweep winner per model, condition = '
                   f'{sweep_registry.CONDITION_LABELS.get(sweep_condition, sweep_condition)}):')
        st.dataframe(preview, width='stretch', hide_index=True)
        missing = [k for k in model_keys if k not in set(best_models_df['model'])]
        if missing:
            st.warning('No sweep result for: ' + ', '.join(MODEL_LABELS.get(k, k) for k in missing) +
                       " - excluded from training in sweep-tuned mode.")
            model_keys = [k for k in model_keys if k not in missing]

    unavailable = [k for k in selected_keys if not sources[k]['available']]
    if unavailable:
        st.warning('Not generated yet, so excluded from training: ' +
                   ', '.join(sources[k]['label'] for k in unavailable) +
                   ' - build them on the Data Generation page first.')

    if st.button('Train model(s)', type='primary', width='stretch'):
        usable_keys = [k for k in selected_keys if sources[k]['available']]
        if not usable_keys:
            st.error('Select at least one available training data source.')
        elif not model_keys:
            st.error('Select at least one algorithm to train.')
        else:
            with st.spinner('Featurizing and training...'):
                sources_pool = dr.combine_sources(usable_keys, sources)
                bundle_fields = fit_training_bundle(
                    model_keys, hp_mode, suite, sources_pool,
                    sweep_condition=sweep_condition, best_models_df=best_models_df)

            run_id = f'run_{len(st.session_state["trained_runs"])}'
            st.session_state['trained_runs'][run_id] = dict(
                run_id=run_id, train_source_keys=usable_keys,
                train_source_labels=[sources[k]['label'] for k in usable_keys],
                trained_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **bundle_fields)
            st.session_state['active_run_id'] = run_id
            hp_note = f' (sweep-tuned, {sweep_condition})' if hp_mode == 'sweep' else ''
            st.success(f'Trained {len(bundle_fields["models"])} model(s) on {bundle_fields["n_train"]} samples '
                       f'({", ".join(sources[k]["label"] for k in usable_keys)}){hp_note}.')

    runs = st.session_state['trained_runs']
    if len(runs) > 1:
        st.caption('Bundles trained this session:')
        rows = [dict(run=rid, trained_at=b['trained_at'],
                      suite=', '.join(sorted(set(b['model_suites'].values()))), n_train=b['n_train'],
                      trained_on=', '.join(b['train_source_labels']))
                for rid, b in runs.items()]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# Section 3 - Module B: testing and transparency
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

    if bundle.get('hyperparam_source') == 'sweep':
        meta = bundle['sweep_model_meta'][model_name]
        condition_label = sweep_registry.CONDITION_LABELS.get(bundle['sweep_condition'], bundle['sweep_condition'])
        st.caption(f"⚙️ Hyperparameters: sweep-tuned - condition **{condition_label}**, "
                   f"suite **{meta['suite']}**, search **{meta['search_method']}**, "
                   f"nested-CV MAE **{meta['nested_cv_mae']:.4f}**.")


def determine_cv_strategy(model_name: str, n: int) -> tuple[str, int]:
    """LeaveOneOut vs k-fold, and fold count, for the leave-out evaluation.
     
    Same convention in run_condition_sweep.py's
    (LOO for the ~40-row 'lab' condition, 5-fold for everything larger)

    true LOO only below LOO_MAX_N rows, and never for mlp/cnn (EXPENSIVE_MODELS)
    """
    if model_name in EXPENSIVE_MODELS:
        return 'kfold', max(2, min(5, n))
    if n <= LOO_MAX_N:
        return 'loo', n
    return 'kfold', max(2, min(5, n))


def rebuild_unfit_model(bundle: dict, model_name: str):
    """A clean unfit clone of the model construction the bundle used
    for `model_name` - same hyperparameters, whether they came from
    model_registry.MODEL_FACTORIES (default) or the condition sweep."""
    if bundle.get('hyperparam_source') == 'sweep':
        meta = bundle['sweep_model_meta'][model_name]
        params_entry = sweep_registry.load_sweep_params(
            bundle['sweep_condition'], model_name, meta['suite'], meta['search_method'])
        return sweep_registry.build_sweep_model(model_name, meta['model_type'], params_entry)
    return MODEL_FACTORIES[model_name]()


def run_leave_out_evaluation(bundle: dict, model_name: str, test_key: str, sources: dict,
                              progress_cb=None) -> dict:
    """No-leakage evaluation of `model_name` on `test_key`'s own
    rows: refits a fresh unfit clone once per fold (rebuild_unfit_model()),
    training on every fold but the held-out one, predicting only the held-out
    rows, and pooling predictions across all folds.
    """
    s = sources[test_key]
    suite = bundle['model_suites'][model_name]
    feature_columns = bundle['feature_columns'][suite]
    feat_df = dr.featurize(s['E'], s['X'], s['y'], suite=suite)
    X = feat_df[feature_columns].to_numpy(dtype=float)
    y = feat_df['concentration'].to_numpy(dtype=float)
    n = len(y)

    strategy, n_folds = determine_cv_strategy(model_name, n)
    splitter = LeaveOneOut() if strategy == 'loo' else KFold(n_splits=n_folds, shuffle=True, random_state=42)
    total = splitter.get_n_splits(X)

    y_true, y_pred = [], []
    for i, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        imputer = SimpleImputer(strategy='median').fit(X[train_idx])
        model = rebuild_unfit_model(bundle, model_name)
        model.fit(X[train_idx], y[train_idx])  # raw (unimputed), same convention as model_registry.train_models()
        pred = np.asarray(model.predict(imputer.transform(X[test_idx])), dtype=float)
        y_true.extend(y[test_idx].tolist())
        y_pred.extend(pred.tolist())
        if progress_cb:
            progress_cb(i, total)

    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    return dict(strategy=strategy, n_folds=total, y_true=y_true_arr, y_pred=y_pred_arr,
                metrics=compute_regression_metrics(y_true_arr, y_pred_arr), feat_df=feat_df)


def render_dataset_test(bundle: dict, model_name: str, sources: dict, test_key: str) -> None:
    st.subheader('Test on an entire dataset')

    is_leaky = test_key in bundle['train_source_keys']
    leave_out_checked = False
    if is_leaky:
        st.warning(f"Warning! Possible **Data leakage:** {sources[test_key]['label']} was part of this model's "
                   'training data. The metrics below reflect memorization, not generalization. Pick '
                   'an independent dataset, or train a bundle that excludes it.')

        n = sources[test_key]['n']
        if n < 2:
            st.caption('Too few rows in this dataset for a leave-out evaluation.')
        else:
            leave_out_checked = st.checkbox(
                'Run an honest leave-out evaluation instead (retrains the model, holding out folds)',
                key='leave_out_checked')
        if leave_out_checked:
            strategy, n_folds = determine_cv_strategy(model_name, n)
            strategy_label = 'Leave-One-Out' if strategy == 'loo' else f'{n_folds}-fold cross-validation'
            st.caption(f'Resolved strategy: **{strategy_label}** on {sources[test_key]["label"]} (n={n}) - '
                       f'the model is refit {n_folds} time(s), each time held out on a different slice of '
                       'rows and evaluated only on those.')
            other_sources = [k for k in bundle['train_source_keys'] if k != test_key]
            if other_sources:
                st.caption("Rows from " + ', '.join(sources[k]['label'] for k in other_sources) +
                           " stay in every fold's training set - only " +
                           f"{sources[test_key]['label']}'s own rows are held out fold-by-fold.")
            if model_name in EXPENSIVE_MODELS:
                st.warning(f'⏱️ {MODEL_LABELS.get(model_name, model_name)} retrains a full neural network '
                           f'{n_folds} time(s) for this evaluation - this can take a few minutes.')
            if st.button('▶ Run leave-out evaluation', type='primary', key='run_loo_btn'):
                progress = st.progress(0.0, text='Starting...')

                def _cb(i: int, total: int) -> None:
                    progress.progress(i / total, text=f'Fold {i}/{total}')

                with st.spinner('Refitting per fold...'):
                    loo = run_leave_out_evaluation(bundle, model_name, test_key, sources, progress_cb=_cb)
                progress.empty()
                st.session_state['dataset_test_result'] = dict(
                    test_key=test_key, y_true=loo['y_true'], y_pred=loo['y_pred'], metrics=loo['metrics'],
                    feat_df=loo['feat_df'], is_leave_out=True, strategy=loo['strategy'], n_folds=loo['n_folds'])

    if not leave_out_checked and st.button('Run test on this dataset', type='primary'):
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
            metrics=compute_regression_metrics(y_true, y_pred), feat_df=feat_df, is_leave_out=False)

    result = st.session_state.get('dataset_test_result')
    if result:
        m = result['metrics']
        cols = st.columns(3)
        cols[0].metric('MAE (uM)', f"{m['MAE']:.3f}")
        cols[1].metric('MAPE', f"{m['MAPE']:.1f}%")
        cols[2].metric('R²', f"{m['R2']:.3f}")
        if result.get('is_leave_out'):
            label = 'Leave-One-Out' if result['strategy'] == 'loo' else f"{result['n_folds']}-fold CV"
            st.info(f'Leave-out evaluation ({label}) - honest, no leakage: every prediction above came '
                    "from a fold that never saw that row during its own fit.")
            title = (f"{MODEL_LABELS.get(model_name, model_name)} - {label} evaluation on "
                     f"{sources[result['test_key']]['label']}")
        else:
            title = f"{MODEL_LABELS.get(model_name, model_name)} on {sources[result['test_key']]['label']}"
        st.plotly_chart(plot_parity(result['y_true'], result['y_pred'], title), width='stretch')
        st.caption('Features the model evaluated (first 10 of %d rows):' % len(result['feat_df']))
        st.dataframe(result['feat_df'].head(10), width='stretch', hide_index=True)


def plot_generated_signal(E: np.ndarray, I: np.ndarray, peak) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E, y=I, mode='lines', name='Signal',
                              line=dict(color='#111111', width=2)))
    fig.add_trace(go.Scatter(
        x=E[peak.start_idx:peak.end_idx + 1], y=I[peak.start_idx:peak.end_idx + 1],
        mode='lines', name='Peak region', fill='tozeroy',
        line=dict(color='#17becf'), fillcolor='rgba(23,190,207,0.25)'))
    fig.add_trace(go.Scatter(x=[peak.Ep], y=[peak.Ip], mode='markers', name='Peak (Ip, Ep)',
                              marker=dict(color='#e34a1a', size=11, symbol='diamond')))
    return apply_default_plotly_layout(fig, title_text='Selected signal')


@st.cache_data(show_spinner=False)
def _featurize_cached(E: np.ndarray, X: np.ndarray, y: np.ndarray, suite: str) -> pd.DataFrame:
    return dr.featurize(E, X, y, suite=suite)


def render_single_point_test(bundle: dict, model_name: str, sources: dict, test_key: str) -> None:
    st.subheader('Inspect a single signal from this dataset')
    st.caption('One real row from the dataset selected above - no synthetic generation, just how this '
               'model scores an actual signal from that data.')

    is_leaky = test_key in bundle['train_source_keys']
    if is_leaky:
        st.warning(f"Warning! Possible **Data leakage:** {sources[test_key]['label']} was part of this "
                   "model's training data - a strong prediction here reflects memorization, not "
                   "generalization. Pick an independent dataset above for an honest single-signal check.")

    s = sources[test_key]
    suite = bundle['model_suites'][model_name]
    feature_columns = bundle['feature_columns'][suite]
    imputer = bundle['imputers'][suite]

    feat_df = _featurize_cached(s['E'], s['X'], s['y'], suite)
    row_order = np.argsort(feat_df['concentration'].to_numpy(dtype=float)).tolist()
    row_idx = st.selectbox(
        'Signal to inspect', options=row_order,
        format_func=lambda i: f"Row {i} - {feat_df['concentration'].iloc[i]:g} uM", key='single_point_row_idx')

    I = np.asarray(s['X'][row_idx], dtype=float)
    target_c = float(s['y'][row_idx])
    feature_vector = feat_df.iloc[row_idx][feature_columns].tolist()

    Signal.set_common_potential_E(s['E'])
    Signal.set_common_baseline_I(np.array([]))  # dataset signals are already baseline-subtracted
    try:
        sig = Signal(I)
    finally:
        Signal.set_common_baseline_I(np.zeros_like(s['E']))

    X_row = pd.DataFrame([feature_vector], columns=feature_columns)
    predicted_c = float(predict(bundle['models'][model_name], imputer, X_row)[0])

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(plot_generated_signal(s['E'], I, sig.peak), width='stretch')
    with right:
        abs_pct_error = 100 * abs(predicted_c - target_c) / target_c
        st.metric('Expected concentration (uM)', f'{target_c:.3f}')
        st.metric(f"{MODEL_LABELS.get(model_name, model_name)} prediction (uM)", f'{predicted_c:.3f}',
                  delta=f'{predicted_c - target_c:+.3f} uM')
        st.metric('Absolute error', f'{abs_pct_error:.1f}%')

        if abs_pct_error < BORDERLINE_PCT_THRESHOLD:
            st.success('**Verdict:** model held up - prediction tracks the target closely.')
        elif abs_pct_error < FOOLED_PCT_THRESHOLD:
            st.warning('**Verdict:** borderline - noticeable drift from the target.')
        else:
            st.error('**Verdict:** model was fooled - prediction is far from the target.')

        if st.button('+ Add to comparison history', width='stretch'):
            row = dict(model=MODEL_LABELS.get(model_name, model_name), expected=target_c,
                       predicted=predicted_c, abs_pct_error=abs_pct_error)
            st.session_state.setdefault('test_history', []).append(row)

    st.caption('Raw signal the model is evaluating for this signal:' if suite == 'raw_signal' else
               'Features the model is evaluating for this signal:')
    st.dataframe(pd.DataFrame([feature_vector], columns=feature_columns), width='stretch', hide_index=True)

    if st.session_state.get('test_history'):
        st.subheader('Comparison history (this session)')
        hist_df = pd.DataFrame(st.session_state['test_history'])
        st.dataframe(hist_df, width='stretch', hide_index=True)
        st.plotly_chart(plot_history_parity(hist_df), width='stretch')
        if st.button('Clear history'):
            st.session_state['test_history'] = []
            st.rerun()


def render_testing_module(sources: dict) -> None:
    st.header('Module B : Testing & Transparency')
    runs = st.session_state.get('trained_runs', {})
    if not runs:
        st.info('Train a model in Module A above first.')
        return

    run_id = st.selectbox('Active model bundle', options=list(runs), key='active_run_id',
                           format_func=lambda rid: f"{rid} -  trained {runs[rid]['trained_at']}")
    bundle = runs[run_id]
    model_name = st.selectbox('Algorithm', options=list(bundle['models']),
                               format_func=lambda k: MODEL_LABELS.get(k, k))

    render_transparency_card(bundle, model_name)

    available_keys = [k for k in DATASET_SOURCE_KEYS if sources[k]['available']]
    if not available_keys:
        st.warning('No datasets available to test on yet - build one on the Data Generation page first.')
        return
    test_key = st.selectbox('Dataset to test', options=available_keys,
                             format_func=lambda k: dr.format_source_option(k, sources), key='test_dataset_key')

    test_mode = st.radio('Test mode', options=['Test an entire dataset', 'Inspect a single signal'],
                          horizontal=True, key='test_mode')
    if test_mode == 'Test an entire dataset':
        render_dataset_test(bundle, model_name, sources, test_key)
    else:
        render_single_point_test(bundle, model_name, sources, test_key)


# Section 4 - page entry point
def main() -> None:
    st.title(' Model Training & Testing')
    st.caption('Choose exactly what data and feature suite a model learns from, then test it '
               'transparently against a whole dataset or a single signal drawn from it.')

    sources = dr.get_available_sources()
    init_trained_runs(sources)

    render_training_module(sources)
    st.divider()
    render_testing_module(sources)


if __name__ == '__main__':
    main()
