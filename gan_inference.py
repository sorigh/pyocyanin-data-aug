"""On-the-fly signal generation from the pre-trained TimeGAN / WGAN-GP checkpoints.

The training scripts under `training/` are batch jobs: they train, save a
checkpoint and dump one fixed sample batch to `gan_output/samples_*`. This
module instead loads the saved generator weights and lets a caller (the
"Data Generation" Streamlit page) draw an arbitrary number of fresh signals
at request time, so the GAN stops being a static comparison overlay and
becomes a real data source a user can size on demand.

Nothing here retrains anything - `training/wgangp_training.py` and
`training/timegan_training.py` remain the only way to produce a new
checkpoint.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from training import timegan_training as _timegan
from training import wgangp_training as _wgan


def _load_wgangp_generator(model_dir: str) -> tuple[_wgan.Generator, dict]:
    with open(f'{model_dir}/wgangp_config.json') as f:
        hp = json.load(f)
    hp['betas'] = tuple(hp['betas'])
    generator = _wgan.Generator(hp)
    generator.load_state_dict(torch.load(f'{model_dir}/wgangp_generator.pt', map_location='cpu'))
    generator.eval()
    return generator, hp


def _load_timegan_generator(model_dir: str) -> tuple[_timegan.TemporalGenerator, _timegan.Recovery, float, float, dict]:
    with open(f'{model_dir}/timegan_config.json') as f:
        cfg = json.load(f)
    X_min, X_max = cfg.pop('X_min'), cfg.pop('X_max')
    generator = _timegan.TemporalGenerator(cfg)
    recovery = _timegan.Recovery(cfg)
    generator.load_state_dict(torch.load(f'{model_dir}/timegan_generator.pt', map_location='cpu'))
    recovery.load_state_dict(torch.load(f'{model_dir}/timegan_recovery.pt', map_location='cpu'))
    generator.eval()
    recovery.eval()
    return generator, recovery, X_min, X_max, cfg


def generate_wgangp(model_dir: str, n: int, conc_min: float = 0.1, conc_max: float = 100.0,
                     seed: int | None = None) -> pd.DataFrame:
    """Draw `n` fresh signals from a saved WGAN-GP generator.

    Returns a DataFrame shaped like `raw/raw_signals_real.csv`
    (`concentration`, `I_0` .. `I_{signal_len-1}`).
    """
    if seed is not None:
        _wgan.set_seed(seed)
    generator, hp = _load_wgangp_generator(model_dir)
    hp = dict(hp, n_generate=n, conc_min=conc_min, conc_max=conc_max)
    device = torch.device('cpu')
    return _wgan.generate_signals(generator, hp, device, n=n)


def generate_timegan(model_dir: str, n: int, conc_min: float = 0.1, conc_max: float = 100.0,
                      seed: int | None = None) -> pd.DataFrame:
    """Draw `n` fresh signals from a saved TimeGAN generator (+ recovery net).

    Returns a DataFrame shaped like `raw/raw_signals_real.csv`.
    """
    if seed is not None:
        _timegan.set_seed(seed)
    generator, recovery, X_min, X_max, cfg = _load_timegan_generator(model_dir)
    cfg = dict(cfg, n_generate=n, conc_min=conc_min, conc_max=conc_max)
    device = torch.device('cpu')
    return _timegan.generate_signals(generator, recovery, cfg, device, X_min, X_max, n=n)


def dataframe_to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split a `generate_*` result into (signals, concentrations) arrays."""
    y = df['concentration'].to_numpy(dtype=float)
    X = df.drop(columns=['concentration']).to_numpy(dtype=float)
    return X, y
