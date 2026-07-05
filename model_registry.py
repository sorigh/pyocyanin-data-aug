"""Regressor factories shared by `export_models.py` (offline export of the
real-data-only baseline) and the "Model Training & Testing" page (in-session
training on whatever data/feature-suite the user picks).

'ridge' / 'random_forest' / 'xgboost' hyperparameters are copied verbatim from
`ABLATION_MODELS` in `full_range_data_augmentation.ipynb` (Section 10,
"Ablation Study") - see `export_models.py`'s docstring for why.

'svr' / 'xgboost_tuned' / 'mlp' / 'cnn' hyperparameters are copied verbatim
from `models/regressors/svr_xgb_best_params.json` and
`models/regressors/mlp_cnn_best_params.json` - the outputs of
`training/tune_ml_hparams.py` (nested LOOCV + skopt.BayesSearchCV) and
`training/tune_dl_hparams.py` (Optuna) respectively. They're hardcoded here,
not read from those JSON files at import time, so that re-running the tuning
scripts (which would silently overwrite those files) can't change frontend
behaviour without someone deliberately updating this file and that change
showing up in a diff. 'xgboost_tuned' is kept as a separate entry from
'xgboost' rather than replacing it, so the ablation-study baseline stays
available for comparison against the LOOCV-tuned version in the frontend.

Keeping every factory in one place means both training paths (offline
export, in-app retraining) fit the exact same model architectures.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from torch_models import (
    CONC_MAX, CONC_MIN, SIGNAL_LEN, build_cnn, build_mlp,
    log10_conc_denormalise, log10_conc_normalise,
)

MODEL_LABELS = {
    'ridge': 'Ridge', 'random_forest': 'Random Forest', 'xgboost': 'XGBoost',
    'svr': 'SVR (RBF, tuned)', 'xgboost_tuned': 'XGBoost (LOOCV-tuned)',
    'mlp': 'MLP (tuned)', 'cnn': '1D-CNN (tuned)',
}

# Models that consume the raw 229-point signal (the 'raw_signal' pseudo-suite
# in augmentation_pipeline.FEATURE_COLUMNS_BY_SUITE) instead of an engineered
# feature suite. Everything else in MODEL_LABELS is a tabular model.
RAW_SIGNAL_MODELS = {'cnn'}

# From models/regressors/svr_xgb_best_params.json ('svr'.best_params / 'xgboost'.best_params)
_SVR_BEST_PARAMS = dict(C=868.4208570619496, epsilon=0.0022767724858461062, gamma=0.0001)
_XGB_TUNED_BEST_PARAMS = dict(
    n_estimators=500, max_depth=3, learning_rate=0.0501746552420551,
    subsample=1.0, colsample_bytree=1.0, min_child_weight=2,
    gamma=7.160304412758296e-05, reg_alpha=8.493583217951526e-06, reg_lambda=0.011728497290648301,
)

# From models/regressors/mlp_cnn_best_params.json. The MLP was re-tuned after
# an initial run trained on raw, unstandardized features (peak_FWHM ~0.05 vs
# wavelet_energy ~1e6) collapsed to predicting a constant regardless of
# input - see the "must standardize" note on MLPWrapper below.
_MLP_BEST_PARAMS = dict(
    lr=0.003336101309469784, n_layers=1, hidden_0=16,
    dropout=0.13611590802176332, batch_size=4, weight_decay=5.924870051994317e-05,
)
_MLP_BEST_EPOCH = 35
_CNN_BEST_PARAMS = dict(
    lr=0.005867953705629066, n_conv_layers=3, base_filters=16, kernel_size=9,
    dropout=0.37435044417008234, fc_hidden=64, batch_size=16, weight_decay=3.657236861819232e-05,
)
_CNN_BEST_EPOCH = 62


# Section 1 - sklearn-compatible wrappers around the tuned PyTorch models
class TorchRegressorWrapper(BaseEstimator, RegressorMixin):
    """Base class giving an Optuna-tuned torch_models regressor a
    fit(X, y) / predict(X) surface, so it can sit in MODEL_FACTORIES next to
    the sklearn/XGBoost models and go through the exact same `train_models()`
    / `predict()` calls as everything else.

    fit() always trains a *fresh* network from scratch, using the tuned
    architecture/optimiser hyperparameters (found once, by Optuna, on real
    data) for `best_epoch` epochs - it never re-runs the search. This is what
    lets Module A retrain 'mlp'/'cnn' in-session on any data source
    combination the user picks, same as every other model.

    predict() returns concentrations already denormalised to uM.
    """

    def __init__(self, best_params: dict, best_epoch: int, conc_min: float = CONC_MIN, conc_max: float = CONC_MAX):
        self.best_params = best_params
        self.best_epoch = best_epoch
        self.conc_min = conc_min
        self.conc_max = conc_max

    def _build_model_for_shape(self, shape_hint: int) -> nn.Module:
        raise NotImplementedError

    def _to_tensor(self, X) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, X, y) -> "TorchRegressorWrapper":
        X_t = self._to_tensor(X)
        y_norm = log10_conc_normalise(np.asarray(y, dtype=float), self.conc_min, self.conc_max).astype(np.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32)

        self.model_ = self._build_model_for_shape(X_t.shape[1])
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.best_params['lr'],
                                      weight_decay=self.best_params['weight_decay'])
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.best_params['batch_size'], shuffle=True)

        self.model_.train()
        for _ in range(max(int(self.best_epoch), 1)):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
        self.model_.eval()
        return self

    def load_pretrained_weights(self, shape_hint: int, state_dict_path: str) -> "TorchRegressorWrapper":
        """Skip fit(): build the tuned architecture and load already-trained
        weights directly. Used by export_models.py for the baseline bundle,
        which reuses training/tune_dl_hparams.py's output (already trained on
        all real data with these exact hyperparameters) instead of paying to
        retrain an identical model from a fresh random init."""
        self.model_ = self._build_model_for_shape(shape_hint)
        self.model_.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
        self.model_.eval()
        return self

    def predict(self, X) -> np.ndarray:
        X_t = self._to_tensor(X)
        with torch.no_grad():
            pred_norm = self.model_(X_t).numpy()
        return log10_conc_denormalise(pred_norm, self.conc_min, self.conc_max)


class MLPWrapper(TorchRegressorWrapper):
    """The MLP's raw engineered features span >6 orders of magnitude
    (peak_FWHM ~0.05 vs wavelet_energy ~1e6). Left unstandardized, the first
    Linear+ReLU layer reliably saturates dead for every input - confirmed on
    this exact dataset, where a first unscaled training run produced a
    checkpoint whose output had zero variance across all 40 real signals.
    So unlike CNNWrapper (raw signal, already a single physical quantity at
    a consistent scale), this wrapper always standardizes X before it
    reaches the network - fitting a fresh StandardScaler in fit(), or
    loading training/tune_dl_hparams.py's saved one in load_pretrained_weights().
    """

    def __init__(self, best_params: dict, best_epoch: int, scaler=None,
                 conc_min: float = CONC_MIN, conc_max: float = CONC_MAX):
        super().__init__(best_params, best_epoch, conc_min, conc_max)
        self.scaler = scaler

    def _build_model_for_shape(self, shape_hint: int) -> nn.Module:
        return build_mlp(self.best_params, input_dim=shape_hint)

    def _raw_array(self, X) -> np.ndarray:
        return X.to_numpy(dtype=np.float64) if hasattr(X, 'to_numpy') else np.asarray(X, dtype=np.float64)

    def _to_tensor(self, X) -> torch.Tensor:
        arr = self._raw_array(X)
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        return torch.tensor(arr, dtype=torch.float32)

    def fit(self, X, y) -> "MLPWrapper":
        self.scaler = StandardScaler().fit(self._raw_array(X))
        return super().fit(X, y)

    def load_pretrained_weights(self, shape_hint: int, state_dict_path: str,
                                 scaler_path: str | None = None) -> "MLPWrapper":
        if scaler_path is not None:
            self.scaler = joblib.load(scaler_path)
        return super().load_pretrained_weights(shape_hint, state_dict_path)


class CNNWrapper(TorchRegressorWrapper):
    def _build_model_for_shape(self, shape_hint: int) -> nn.Module:
        return build_cnn(self.best_params, signal_len=shape_hint)

    def _to_tensor(self, X) -> torch.Tensor:
        arr = X.to_numpy(dtype=np.float32) if hasattr(X, 'to_numpy') else np.asarray(X, dtype=np.float32)
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(-1)  # (B, signal_len, 1)


# Section 2 - model factories
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
    'svr': lambda: make_pipeline(StandardScaler(), SVR(kernel='rbf', **_SVR_BEST_PARAMS)),
    'xgboost_tuned': lambda: XGBRegressor(
        objective='reg:squarederror', random_state=42, verbosity=0, n_jobs=-1,
        **_XGB_TUNED_BEST_PARAMS),
    'mlp': lambda: MLPWrapper(_MLP_BEST_PARAMS, _MLP_BEST_EPOCH),
    'cnn': lambda: CNNWrapper(_CNN_BEST_PARAMS, _CNN_BEST_EPOCH),
}


# Section 3 - shared fit / predict entry points
def train_models(X: pd.DataFrame, y: pd.Series, model_keys: list[str],
                  X_raw_signal: pd.DataFrame | None = None) -> tuple[dict, SimpleImputer, SimpleImputer | None]:
    """Fit one model per `model_keys` entry plus a median imputer per input
    representation in use.

    `X` holds whichever engineered feature suite (core/extended/experimental)
    the caller picked; `X_raw_signal` (only needed if a model in
    RAW_SIGNAL_MODELS - currently just 'cnn' - is requested) holds the raw
    229-point signal instead. Each raw-signal model is fit against
    `X_raw_signal`; everything else is fit against `X`.

    The imputers mirror `export_models.py`'s safety net: real training data
    has no NaNs, but a manually-augmented or GAN-generated single signal can
    occasionally hit a degenerate edge case in one experimental feature.
    """
    imputer = SimpleImputer(strategy='median').fit(X)
    raw_signal_imputer = SimpleImputer(strategy='median').fit(X_raw_signal) if X_raw_signal is not None else None

    models = {}
    for key in model_keys:
        model = MODEL_FACTORIES[key]()
        if key in RAW_SIGNAL_MODELS:
            if X_raw_signal is None:
                raise ValueError(f"'{key}' needs the raw_signal feature suite (X_raw_signal) to train.")
            model.fit(X_raw_signal, y)
        else:
            model.fit(X, y)
        models[key] = model
    return models, imputer, raw_signal_imputer


def predict(model, imputer: SimpleImputer, X: pd.DataFrame) -> np.ndarray:
    X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)
    return np.asarray(model.predict(X_imputed), dtype=float)
