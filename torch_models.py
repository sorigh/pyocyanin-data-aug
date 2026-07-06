"""Shared PyTorch regressor architectures for the MLP and 1D-CNN concentration
predictors, plus the log10 concentration scaling they train against.

Defined once here - instead of inline in training/tune_dl_hparams.py - because
two different consumers now need the exact same class definitions:
    - training/tune_dl_hparams.py: searches hyperparameters with Optuna and
      trains the checkpoints in models/mlp.joblib / models/cnn.joblib.
    - model_registry.py: reconstructs the same architecture at inference time
      in the Streamlit app and loads those checkpoints' weights into it.

If the two ever drifted (e.g. someone tweaks the CNN in one file but not the
other), `load_state_dict` would either fail loudly or - worse - load silently
into a mismatched architecture. Keeping one definition makes that impossible.

`log10_conc_normalise` / `log10_conc_denormalise` are the same mapping
training/timegan_training.py uses for its own conditioning input (log10 + affine
to [-1, 1], c_min/c_max = [0.1, 100.0]). That module is left with its own copy
since it's an independent GAN training script; this copy is the one shared
between the tuning script and the frontend.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SIGNAL_LEN = 229
CONC_MIN = 0.1
CONC_MAX = 100.0


def log10_conc_normalise(c: np.ndarray, c_min: float = CONC_MIN, c_max: float = CONC_MAX) -> np.ndarray:
    """Map concentration to [-1, 1] via log10 scaling."""
    log_c   = np.log10(np.clip(c, 1e-9, None))
    log_min = np.log10(c_min)
    log_max = np.log10(c_max)
    return 2.0 * (log_c - log_min) / (log_max - log_min) - 1.0


def log10_conc_denormalise(c_norm: np.ndarray, c_min: float = CONC_MIN, c_max: float = CONC_MAX) -> np.ndarray:
    log_min = np.log10(c_min)
    log_max = np.log10(c_max)
    log_c   = (np.clip(c_norm, -6.0, 6.0) + 1.0) / 2.0 * (log_max - log_min) + log_min
    return 10.0 ** log_c


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) -> (B,)"""
        return self.net(x).squeeze(-1)


class CNN1DRegressor(nn.Module):
    """1D-CNN regressor. forward() strictly accepts (B, signal_len, 1)."""

    def __init__(self, signal_len: int, conv_channels: list[int], kernel_size: int,
                 dropout: float, fc_hidden: int):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in conv_channels:
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, fc_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )
        self.signal_len = signal_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, signal_len, 1) -> (B,)"""
        assert x.shape[1:] == (self.signal_len, 1), f"Expected (B, {self.signal_len}, 1), got {tuple(x.shape)}"
        x = x.permute(0, 2, 1)          # (B, 1, signal_len)
        x = self.conv(x)                # (B, C, signal_len)
        x = self.pool(x).squeeze(-1)    # (B, C)
        return self.fc(x).squeeze(-1)   # (B,)


def build_mlp(best_params: dict, input_dim: int) -> MLPRegressor:
    n_layers = best_params["n_layers"]
    hidden_sizes = [best_params[f"hidden_{i}"] for i in range(n_layers)]
    return MLPRegressor(input_dim, hidden_sizes, best_params["dropout"])


def build_cnn(best_params: dict, signal_len: int = SIGNAL_LEN) -> CNN1DRegressor:
    n_conv_layers = best_params["n_conv_layers"]
    base_filters = best_params["base_filters"]
    conv_channels = [base_filters * (2 ** i) for i in range(n_conv_layers)]
    return CNN1DRegressor(signal_len, conv_channels, best_params["kernel_size"],
                           best_params["dropout"], best_params["fc_hidden"])
