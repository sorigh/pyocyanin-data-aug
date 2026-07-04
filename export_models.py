"""
Trains and saves the three regressors used as the "pretrained baseline"
bundle on the Streamlit "Model Training & Testing" page
(pages/2_Model_Training_Testing.py): Ridge, Random Forest and XGBoost.

Hyperparameters and the feature suite are *not* re-derived here - they are
copied verbatim from `ABLATION_MODELS` / `ABLATION_FEATURE_SUITE` in
`full_range_data_augmentation.ipynb` (Section 10, "Ablation Study"), which is
the notebook that already put these three models through Protocol A / C
testing on the full 0.1-100 uM range. Reusing those exact configs means the
Streamlit page is checking synthetic signals against the same model that was
already validated for that purpose, instead of a freshly re-tuned one.

(model_training.ipynb runs a *nested* LOOCV + BayesSearchCV search on the
smaller 'core' feature suite instead. That search is for estimating
generalisation error, not for picking one final config - Random Forest and
XGBoost never converge on a single winner across folds there. Ridge's result
does converge (alpha=0.1, fit_intercept=False, 39/40 folds) but a different
fit_intercept than the ablation's, and on a different feature suite - so it
isn't mixed in here.)

Models are trained on every real calibration replicate
(`vectorized/experimental.csv`, 40 rows) - not on any augmented/synthetic
data - since the whole point of the Testing page is to see how a
real-data-only model responds to signals it has never seen.

Run once (and again whenever vectorized/experimental.csv changes):
    python export_models.py
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

from augmentation_pipeline import FEATURE_COLUMNS_BY_SUITE
from model_registry import MODEL_FACTORIES

VECTORIZED_EXPERIMENTAL_CSV = 'vectorized/experimental.csv'
MODELS_DIR = 'models'

FEATURE_COLUMNS = FEATURE_COLUMNS_BY_SUITE['experimental']


def main() -> None:
    df = pd.read_csv(VECTORIZED_EXPERIMENTAL_CSV)
    X, y = df[FEATURE_COLUMNS], df['concentration']

    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, factory in MODEL_FACTORIES.items():
        model = factory()
        model.fit(X, y)
        path = f'{MODELS_DIR}/{name}.joblib'
        joblib.dump(model, path)
        print(f'Saved {path}  (trained on {len(X)} real samples, {len(FEATURE_COLUMNS)} features)')

    # A single shared imputer the Streamlit page runs *every* prediction
    # through, regardless of which model is selected. Training data has no
    # NaNs (fitting it is a no-op here), but a manually-augmented single
    # signal can occasionally hit a degenerate edge case in one of the
    # experimental features (e.g. get_asymetry dividing by a ~0 slope) -
    # RandomForestRegressor and XGBRegressor can't take a NaN input, so this
    # is the safety net that lets the page show a prediction instead of a
    # stack trace. Ridge already has its own imputer baked into its pipeline
    # per the ablation config; running it through this one first just makes
    # that inner imputer a no-op too.
    imputer = SimpleImputer(strategy='median').fit(X)
    joblib.dump(imputer, f'{MODELS_DIR}/feature_imputer.joblib')
    print(f'Saved {MODELS_DIR}/feature_imputer.joblib')

    with open(f'{MODELS_DIR}/feature_metadata.json', 'w') as f:
        json.dump({'feature_suite': 'experimental', 'feature_columns': FEATURE_COLUMNS}, f, indent=2)
    print(f'Saved {MODELS_DIR}/feature_metadata.json')


if __name__ == '__main__':
    main()
