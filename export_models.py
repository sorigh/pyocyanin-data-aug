"""
Trains and saves the regressors used as the "pretrained baseline" bundle on
the Streamlit "Model Training & Testing" page (pages/2_Model_Training_Testing.py):
Ridge, Random Forest, XGBoost, SVR (tuned), XGBoost (tuned), MLP (tuned) and
1D-CNN (tuned).

Hyperparameters and the feature suite for Ridge/Random Forest/XGBoost are
*not* re-derived here - they are copied verbatim from `ABLATION_MODELS` /
`ABLATION_FEATURE_SUITE` in `full_range_data_augmentation.ipynb` (Section 10,
"Ablation Study"), which is the notebook that already put these three models
through Protocol A / C testing on the full 0.1-100 uM range. Reusing those
exact configs means the Streamlit page is checking synthetic signals against
the same model that was already validated for that purpose, instead of a
freshly re-tuned one.

(model_training.ipynb runs a *nested* LOOCV + BayesSearchCV search on the
smaller 'core' feature suite instead. That search is for estimating
generalisation error, not for picking one final config - Random Forest and
XGBoost never converge on a single winner across folds there. Ridge's result
does converge (alpha=0.1, fit_intercept=False, 39/40 folds) but a different
fit_intercept than the ablation's, and on a different feature suite - so it
isn't mixed in here.)

SVR and XGBoost (tuned) hyperparameters come from training/tune_ml_hparams.py
(nested LOOCV + skopt.BayesSearchCV); MLP and 1D-CNN come from
training/tune_dl_hparams.py (Optuna). Both scripts' hyperparameters are
hardcoded into model_registry.py (see that module's docstring for why) - this
script just fits/loads models using those factories. MLP and CNN specifically
*load* training/tune_dl_hparams.py's already-trained weights
(models/regressors/{mlp,cnn}_tuned.pt) rather than refitting: that run already
trained on the exact same real data this script would otherwise retrain on,
so reusing it is free and reproduces exactly the model whose metrics are in
models/regressors/mlp_cnn_best_params.json.

Models are trained on every real calibration replicate - `data/vectorized/experimental.csv`
(40 rows, engineered features) for the tabular models, `data/raw/raw_signals_real.csv`
(40 rows, raw signal) for the 1D-CNN - not on any augmented/synthetic data,
since the whole point of the Testing page is to see how a real-data-only
model responds to signals it has never seen.

Run once (and again whenever vectorized/experimental.csv, raw/raw_signals_real.csv,
or the tuning scripts' results/ outputs change):
    python export_models.py
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

from augmentation_pipeline import FEATURE_COLUMNS_BY_SUITE
from model_registry import MODEL_FACTORIES, RAW_SIGNAL_MODELS

import paths
VECTORIZED_EXPERIMENTAL_CSV = paths.VECTORIZED_EXPERIMENTAL_CSV
RAW_SIGNALS_REAL_CSV = paths.REAL_SIGNALS_CSV
MODELS_DIR = paths.REGRESSORS_DIR

# Where training/tune_dl_hparams.py dropped its checkpoints when last run.
# Only the weights (and, for the MLP, the feature scaler) are read from here
# - the hyperparameters that produced them are hardcoded into model_registry.py.
TUNED_MLP_STATE_DICT = paths.TUNED_MLP_STATE_DICT
TUNED_MLP_SCALER = paths.TUNED_MLP_SCALER
TUNED_CNN_STATE_DICT = paths.TUNED_CNN_STATE_DICT

FEATURE_SUITE = 'experimental'
FEATURE_COLUMNS = FEATURE_COLUMNS_BY_SUITE[FEATURE_SUITE]
RAW_SIGNAL_COLUMNS = FEATURE_COLUMNS_BY_SUITE['raw_signal']

# Which input representation each shipped model expects - saved into
# feature_metadata.json so pages/2_Model_Training_Testing.py knows, per
# model, whether to hand it engineered features or the raw signal.
MODEL_SUITES = {name: ('raw_signal' if name in RAW_SIGNAL_MODELS else FEATURE_SUITE)
                 for name in MODEL_FACTORIES}


def main() -> None:
    df = pd.read_csv(VECTORIZED_EXPERIMENTAL_CSV)
    X, y = df[FEATURE_COLUMNS], df['concentration']

    raw_df = pd.read_csv(RAW_SIGNALS_REAL_CSV)
    X_raw, y_raw = raw_df[RAW_SIGNAL_COLUMNS], raw_df['concentration']

    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, factory in MODEL_FACTORIES.items():
        if name == 'mlp':
            model = factory().load_pretrained_weights(len(FEATURE_COLUMNS), TUNED_MLP_STATE_DICT, TUNED_MLP_SCALER)
            n_train, note = len(X), f'weights reused from {TUNED_MLP_STATE_DICT}, not refit'
        elif name == 'cnn':
            model = factory().load_pretrained_weights(len(RAW_SIGNAL_COLUMNS), TUNED_CNN_STATE_DICT)
            n_train, note = len(X_raw), f'weights reused from {TUNED_CNN_STATE_DICT}, not refit'
        elif name in RAW_SIGNAL_MODELS:
            model = factory()
            model.fit(X_raw, y_raw)
            n_train, note = len(X_raw), 'fit on raw signal'
        else:
            model = factory()
            model.fit(X, y)
            n_train, note = len(X), f'fit on {FEATURE_SUITE} features'
        path = f'{MODELS_DIR}/{name}.joblib'
        joblib.dump(model, path)
        print(f'Saved {path}  ({n_train} real samples - {note})')

    # A shared imputer for the engineered-feature models: real training data
    # has no NaNs (fitting it is a no-op here), but a manually-augmented
    # single signal can occasionally hit a degenerate edge case in one of the
    # experimental features (e.g. get_asymetry dividing by a ~0 slope) -
    # RandomForestRegressor and XGBRegressor can't take a NaN input, so this
    # is the safety net that lets the page show a prediction instead of a
    # stack trace. Ridge already has its own imputer baked into its pipeline
    # per the ablation config; running it through this one first just makes
    # that inner imputer a no-op too.
    imputer = SimpleImputer(strategy='median').fit(X)
    joblib.dump(imputer, f'{MODELS_DIR}/feature_imputer.joblib')
    print(f'Saved {MODELS_DIR}/feature_imputer.joblib')

    # A second imputer for the raw-signal model(s) (currently just 'cnn') -
    # its 229 columns aren't compatible with the feature imputer above.
    raw_signal_imputer = SimpleImputer(strategy='median').fit(X_raw)
    joblib.dump(raw_signal_imputer, f'{MODELS_DIR}/raw_signal_imputer.joblib')
    print(f'Saved {MODELS_DIR}/raw_signal_imputer.joblib')

    with open(f'{MODELS_DIR}/feature_metadata.json', 'w') as f:
        json.dump({
            'model_suites': MODEL_SUITES,
            'feature_columns': {FEATURE_SUITE: FEATURE_COLUMNS, 'raw_signal': RAW_SIGNAL_COLUMNS},
        }, f, indent=2)
    print(f'Saved {MODELS_DIR}/feature_metadata.json')


if __name__ == '__main__':
    main()
