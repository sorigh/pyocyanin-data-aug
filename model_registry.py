"""Regressor factories shared by `export_models.py` (offline export of the
real-data-only baseline) and the "Model Training & Testing" page (in-session
training on whatever data/feature-suite the user picks).

Hyperparameters are copied verbatim from `ABLATION_MODELS` in
`full_range_data_augmentation.ipynb` (Section 10, "Ablation Study") - see
`export_models.py`'s docstring for why. Keeping the factories in one place
means both training paths (offline export, in-app retraining) fit the exact
same model architectures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

MODEL_LABELS = {'ridge': 'Ridge', 'random_forest': 'Random Forest', 'xgboost': 'XGBoost'}

MODEL_FACTORIES = {
    'ridge': lambda: make_pipeline(
        SimpleImputer(strategy='median'),
        Ridge(alpha=0.1, fit_intercept=True, random_state=42)),
    'random_forest': lambda: RandomForestRegressor(
        n_estimators=500, max_depth=9, max_features=0.75,
        min_samples_split=2, min_samples_leaf=1,
        random_state=42, n_jobs=-1),
    'xgboost': lambda: XGBRegressor(
        n_estimators=235, max_depth=3, learning_rate=0.0698,
        subsample=0.7682, colsample_bytree=0.9395,
        gamma=0.3391, min_child_weight=3,
        random_state=42, verbosity=0, n_jobs=-1),
}


def train_models(X: pd.DataFrame, y: pd.Series, model_keys: list[str]) -> tuple[dict, SimpleImputer]:
    """Fit one model per `model_keys` entry plus a shared median imputer.

    The imputer mirrors `export_models.py`'s safety net: real training data
    has no NaNs, but a manually-augmented or GAN-generated single signal can
    occasionally hit a degenerate edge case in one experimental feature.
    """
    imputer = SimpleImputer(strategy='median').fit(X)
    models = {}
    for key in model_keys:
        model = MODEL_FACTORIES[key]()
        model.fit(X, y)
        models[key] = model
    return models, imputer


def predict(model, imputer: SimpleImputer, X: pd.DataFrame) -> np.ndarray:
    X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)
    return np.asarray(model.predict(X_imputed), dtype=float)
