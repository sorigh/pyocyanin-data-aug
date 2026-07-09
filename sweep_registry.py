"""Loads pre-computed condition sweep results for the frontend UI.

Provides tuned hyperparameters as an alternative to `model_registry.py` defaults. 
This layer maps the four sweep conditions to the dataset 
sources, guaranteeing that tuned hyperparameters are only ever applied to the 
exact data they were trained on:

    * lab         -> ['real']
    * physics_aug -> ['stable_augmented']
    * gan         -> ['stable_wgangp', 'stable_timegan']
    * combined    -> ['real', 'stable_augmented', 'stable_wgangp', 'stable_timegan']
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

import model_registry
import paths
import run_condition_sweep as rcs

CONDITION_SOURCE_KEYS = {
    'lab': ['real'],
    'physics_aug': ['stable_augmented'],
    'gan': ['stable_wgangp', 'stable_timegan'],
    'combined': ['real', 'stable_augmented', 'stable_wgangp', 'stable_timegan'],
}

CONDITION_LABELS = {
    'lab': 'Lab (real data only, n=40, LOO-tuned)',
    'physics_aug': 'Physics-augmented only (n=300, 5-fold-tuned)',
    'gan': 'GAN only (WGAN-GP + TimeGAN on real, n=600, 5-fold-tuned)',
    'combined': 'Combined (real + physics-aug + GAN, n=940, 5-fold-tuned)',
}


def list_conditions_available() -> list[str]:
    """Which of the 4 conditions actually have a results table on disk"""
    return [c for c in paths.CONDITION_NAMES if paths.condition_best_models_csv(c).exists()]


def load_best_models_table(condition: str) -> pd.DataFrame:
    """One row per model = its best (suite, search_method) for this
    condition"""
    return pd.read_csv(paths.condition_best_models_csv(condition))


@lru_cache(maxsize=1)
def _flat_job_lookup() -> dict[tuple[str, str, str], "rcs.Job"]:
    """Builds cached mapping of `(condition, model, suite)` to `Job` objects.
    Scans the `models/` directory for `_best_params.json` files. Jobs without 
    a defined suite default to `'raw_signal'` suite key.
    """
    jobs = rcs.discover_jobs_flat(paths.MODELS_DIR)
    return {(job.condition, job.model, job.suite or 'raw_signal'): job for job in jobs}


def load_sweep_params(condition: str, model: str, suite: str, search_method: str) -> dict:
    """Loads a model's best parameters and metrics from its sweep output JSON.
    Note: `search_method` kept in the signature to allow unpacking from `load_best_models_table()` rows."""
    job = _flat_job_lookup().get((condition, model, suite))
    if job is None:
        raise FileNotFoundError(
            f"No sweep result found for condition={condition!r} model={model!r} suite={suite!r} "
            f"under {paths.MODELS_DIR} the sweep output for this combination may not be copied")
    data = json.loads(job.output_json.read_text())
    return data[job.model]


def build_sweep_model(model_key: str, model_type: str, params_entry: dict):
    """Dispatch to the right (unfit) model builder for one sweep result
    entry"""
    if model_type == 'ml':
        return model_registry.build_from_sweep_params(model_key, params_entry['best_params'])
    if model_type == 'mlp':
        return model_registry.MLPWrapper(params_entry['best_params'], params_entry['best_epoch'])
    if model_type == 'cnn':
        return model_registry.CNNWrapper(params_entry['best_params'], params_entry['best_epoch'])
    raise ValueError(f'Unknown model_type: {model_type!r}')
