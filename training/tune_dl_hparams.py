"""
Optuna Hyperparameter Search for the MLP and 1D-CNN Regressors
Here is a concise version of the docstring reduced to four main descriptive phrases:

Uses Optuna (median pruner) to end trial mid training when
validation curve falls behind the median of previous trials at the same epoch.
Compute savings. 
(in tune_ml_hparams.py SVR and XGBoost get tuned with skopt.BayesSearchCV 
because they're cheap to refit thousands of times)
--

MLP: trained on the same 1D extracted-feature CSVs as tune_ml_hparams.py
(defaults to experimental).
1D-CNN: trained on the raw 229-point signal directly.

Target normalisation
--
Maps logarithmic concentration targets to a [-1, 1] 
scale to ensure stable MSE loss calculation, while 
reporting final metrics in standard units.

The MLPRegressor / CNN1DRegressor architecture in torch_models.py (repo root),
 model_registry.py also gets it from there 

Usage
-----
    python training/tune_dl_hparams.py
    python training/tune_dl_hparams.py --models cnn --n-trials 100 --max-epochs 300
    python training/tune_dl_hparams.py --features-csv vectorized/experimental.csv \\
        --signals-csv raw/raw_signals_real.csv

Output
------
    models/mlp_cnn_best_params.json   - winning hyperparams + held-out metrics
    models/mlp_tuned.pt               - final MLP weights (best hyperparams, full data)
    models/mlp_scaler.joblib          - StandardScaler fit on the full data the MLP trained on
                                         (must be applied before every MLP predict() call - see
                                         model_registry.MLPWrapper)
    models/cnn_tuned.pt               - final CNN weights (best hyperparams, full data)
    logs/tune_dl_hparams_studies.csv  - every trial's params + outcome, per model
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import joblib
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for torch_models
from torch_models import CNN1DRegressor, MLPRegressor, build_cnn, build_mlp  # noqa: E402
from torch_models import log10_conc_denormalise, log10_conc_normalise  # noqa: E402


# 0. HYPERPARAMETERS
HP = {
    # Data
    "features_csv":  "vectorized/experimental.csv",   # for the MLP
    "signals_csv":   "raw/raw_signals_real.csv",       # for the 1D-CNN
    "target_col":    "concentration",
    "drop_cols":     ["sig_id"],
    "signal_len":    229,
    "conc_min":      0.1,
    "conc_max":      100.0,

    # Search
    "models":        ["mlp", "cnn"],
    "n_trials":      50,
    "max_epochs":    300,
    "patience":      25,          # early-stop patience within a trial
    "val_split":     0.2,
    "pruner_warmup": 10,          # epochs before pruning can kick in

    "seed":          42,
    "output_json":   "models/mlp_cnn_best_params.json",
    "log_csv":       "logs/tune_dl_hparams_studies.csv",
    "models_dir":    "models",

    # Optional honest generalisation-error estimate (mirrors tune_ml_hparams.py's
    # nested LOOCV/K-fold report). Off by default since it multiplies runtime by
    # the number of outer folds - each fold re-runs its own Optuna search.
    "eval_cv":       "none",      # "none" | "loo" | "kfold"
    "eval_splits":   5,           # only used when eval_cv == "kfold"
    "eval_n_trials": 20,          # trials per outer fold (cheaper than the final n_trials)
}


# 1. UTILITIES  (kept in sync with training/timegan_training.py)
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"  Device: {dev}")
    return dev


def stratified_split(y_raw: np.ndarray, val_split: float, seed: int):
    """Train/val split, stratified on the raw concentration classes so every
    concentration level (of which there are only ~10-12) is represented in
    both partitions where possible. Falls back to a plain random split if
    stratification can't be satisfied (e.g. a class with a single member)."""
    idx = np.arange(len(y_raw))
    try:
        return train_test_split(idx, test_size=val_split, random_state=seed, stratify=y_raw)
    except ValueError:
        print("  Warning: stratified split failed (a concentration class is too small) - "
              "falling back to a plain random split.")
        return train_test_split(idx, test_size=val_split, random_state=seed)


# 2. DATA
def load_mlp_features(hp: dict) -> tuple[np.ndarray, np.ndarray, list]:
    df = pd.read_csv(hp["features_csv"])
    drop = [c for c in hp["drop_cols"] if c in df.columns] + [hp["target_col"]]
    feature_cols = [c for c in df.columns if c not in drop]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[hp["target_col"]].to_numpy(dtype=np.float32)
    print(f"  [MLP] Loaded {hp['features_csv']}  |  {X.shape[0]} rows  |  {X.shape[1]} features: {feature_cols}")
    return X, y, feature_cols


def load_cnn_signals(hp: dict) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(hp["signals_csv"])
    I_cols = [c for c in df.columns if c.startswith("I_")]
    X = df[I_cols].to_numpy(dtype=np.float32)
    y = df[hp["target_col"]].to_numpy(dtype=np.float32)
    assert X.shape[1] == hp["signal_len"], (
        f"Expected signal_len={hp['signal_len']}, got {X.shape[1]} columns in {hp['signals_csv']}")
    print(f"  [CNN] Loaded {hp['signals_csv']}  |  {X.shape[0]} rows  |  signal_len={X.shape[1]}")
    return X, y


# 4. GENERIC TRAIN LOOP WITH EARLY STOPPING + OPTUNA PRUNING
def train_with_pruning(model: nn.Module, optimizer: torch.optim.Optimizer,
                        train_loader: DataLoader, X_val: torch.Tensor, y_val_raw: np.ndarray,
                        hp: dict, device: torch.device, trial: "optuna.Trial | None") -> tuple[float, int]:
    loss_fn = nn.MSELoss()
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred_norm = model(X_val).cpu().numpy()
        # A diverging trial can push the normalised prediction far outside
        # [-1, 1]; clip before the 10**x in log10_conc_denormalise so a bad
        # trial reports a large-but-finite MAE instead of an inf/overflow
        # warning (still ranks worst for Optuna, just without the noise).
        val_pred_norm = np.clip(val_pred_norm, -6.0, 6.0)
        val_pred_uM = log10_conc_denormalise(val_pred_norm, hp["conc_min"], hp["conc_max"])
        val_mae = float(np.mean(np.abs(val_pred_uM - y_val_raw)))

        if trial is not None:
            trial.report(val_mae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_mae < best_val_mae - 1e-6:
            best_val_mae, best_epoch, epochs_no_improve = val_mae, epoch, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hp["patience"]:
                break

    return best_val_mae, best_epoch


# 5. OPTUNA OBJECTIVES
def objective_mlp(trial: optuna.Trial, X_train, y_train_norm, X_val, y_val_raw, hp, device) -> float:
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    n_layers     = trial.suggest_int("n_layers", 1, 3)
    hidden_sizes = [trial.suggest_categorical(f"hidden_{i}", [16, 32, 64, 128, 256]) for i in range(n_layers)]
    dropout      = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size   = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = MLPRegressor(X_train.shape[1], hidden_sizes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train_norm, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_val_mae, best_epoch = train_with_pruning(model, optimizer, train_loader, X_val_t, y_val_raw, hp, device, trial)
    trial.set_user_attr("best_epoch", best_epoch)
    return best_val_mae


def objective_cnn(trial: optuna.Trial, X_train, y_train_norm, X_val, y_val_raw, hp, device) -> float:
    lr            = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 4)
    base_filters  = trial.suggest_categorical("base_filters", [8, 16, 32])
    conv_channels = [base_filters * (2 ** i) for i in range(n_conv_layers)]
    kernel_size   = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
    dropout       = trial.suggest_float("dropout", 0.1, 0.5)
    fc_hidden     = trial.suggest_categorical("fc_hidden", [16, 32, 64])
    batch_size    = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = CNN1DRegressor(hp["signal_len"], conv_channels, kernel_size, dropout, fc_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1),
                              torch.tensor(y_train_norm, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1).to(device)

    best_val_mae, best_epoch = train_with_pruning(model, optimizer, train_loader, X_val_t, y_val_raw, hp, device, trial)
    trial.set_user_attr("best_epoch", best_epoch)
    return best_val_mae


# 6. SEARCH (on a given pool) + RETRAIN ON THAT SAME POOL'S FULL DATA
def search_and_retrain(name: str, objective_fn, build_model_fn, X: np.ndarray, y_raw: np.ndarray,
                        hp: dict, device: torch.device, scale_features: bool, n_trials: int,
                        ) -> tuple[dict, pd.DataFrame, nn.Module, "StandardScaler | None"]:
    """Runs Optuna search on (X, y_raw), then retrains the winning
    architecture on that same (X, y_raw) pool in full (no held-out split).

    Called once on the whole dataset by run_study() and once per outer 
    fold's training portion by run_nested_eval()
    ( fold-local model that never sees its own held-out test data).

    scale_features=True standardizes X (fit on the train split for the
    search, refit on the full pool for the final retrain) before it ever
    reaches the network. 
    """
    train_idx, val_idx = stratified_split(y_raw, hp["val_split"], hp["seed"])
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_raw, y_val_raw = y_raw[train_idx], y_raw[val_idx]
    y_train_norm = log10_conc_normalise(y_train_raw, hp["conc_min"], hp["conc_max"]).astype(np.float32)

    if scale_features:
        search_scaler = StandardScaler().fit(X_train)
        X_train = search_scaler.transform(X_train).astype(np.float32)
        X_val = search_scaler.transform(X_val).astype(np.float32)

    print(f"\n  [{name}] Optuna search: {n_trials} trials, "
          f"{len(X_train)} train / {len(X_val)} val signals ...")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=hp["seed"]),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=hp["pruner_warmup"]),
    )
    t0 = time.time()
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train_norm, X_val, y_val_raw, hp, device),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"  [{name}] Search done in {time.time() - t0:.0f}s | "
          f"{n_pruned}/{len(study.trials)} trials pruned | best val MAE (uM): {study.best_value:.4f}")

    # Retrain on the FULL pool with the winning hyperparams, for the number
    # of epochs that scored best during the search (no early stopping)
    best_epoch = max(study.best_trial.user_attrs.get("best_epoch", 1), 1)
    y_norm_full = log10_conc_normalise(y_raw, hp["conc_min"], hp["conc_max"]).astype(np.float32)

    final_scaler = None
    X_for_final = X
    if scale_features:
        final_scaler = StandardScaler().fit(X)
        X_for_final = final_scaler.transform(X).astype(np.float32)

    model = build_model_fn(study.best_params, X.shape[1] if X.ndim == 2 else hp["signal_len"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=study.best_params["lr"],
                                  weight_decay=study.best_params["weight_decay"])
    X_full_t = torch.tensor(X_for_final, dtype=torch.float32)
    if X_full_t.ndim == 2 and X_full_t.shape[1] == hp["signal_len"] and name == "cnn":
        X_full_t = X_full_t.unsqueeze(-1)
    train_ds = TensorDataset(X_full_t, torch.tensor(y_norm_full, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=study.best_params["batch_size"], shuffle=True)

    model.train()
    loss_fn = nn.MSELoss()
    for _ in range(best_epoch):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    result = {
        "best_params":   study.best_params,
        "best_epoch":    best_epoch,
        "val_mae_uM":    study.best_value,
        "n_trials":      len(study.trials),
        "n_pruned":      n_pruned,
        "n_train":       len(X_train),
        "n_val":         len(X_val),
    }

    trials_df = study.trials_dataframe()
    trials_df.insert(0, "model", name)
    return result, trials_df, model, final_scaler


def run_study(name: str, objective_fn, build_model_fn, X: np.ndarray, y_raw: np.ndarray,
              hp: dict, device: torch.device, scale_features: bool = False,
              ) -> tuple[dict, pd.DataFrame, nn.Module, "StandardScaler | None"]:
    """Search + retrain on the FULL dataset - produces the model that gets saved to disk."""
    return search_and_retrain(name, objective_fn, build_model_fn, X, y_raw, hp, device,
                               scale_features, n_trials=hp["n_trials"])


def predict_uM(model: nn.Module, scaler: "StandardScaler | None", X: np.ndarray, hp: dict,
               device: torch.device, name: str) -> np.ndarray:
    """Applies the same scaling/reshaping convention used during training, then
    denormalises back to uM - mirrors the val-prediction logic in train_with_pruning."""
    X_t = scaler.transform(X).astype(np.float32) if scaler is not None else X.astype(np.float32)
    X_tensor = torch.tensor(X_t, dtype=torch.float32)
    if name == "cnn":
        X_tensor = X_tensor.unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        pred_norm = model(X_tensor.to(device)).cpu().numpy()
    pred_norm = np.clip(pred_norm, -6.0, 6.0)
    return log10_conc_denormalise(pred_norm, hp["conc_min"], hp["conc_max"])


# 6b. OUTER LOO/K-FOLD LOOP - HONEST GENERALISATION-ERROR ESTIMATE
def run_nested_eval(name: str, objective_fn, build_model_fn, X: np.ndarray, y_raw: np.ndarray,
                     hp: dict, device: torch.device, scale_features: bool) -> dict:
    """Mirrors tune_ml_hparams.py's nested LOOCV/K-fold report: for every
    outer fold, runs a fresh (cheaper - hp['eval_n_trials'] instead of
    hp['n_trials']) Optuna search + retrain using only that fold's training
    rows, then scores the held-out fold. This is what makes the MLP/CNN
    numbers comparable to the sklearn models' nested_cv_mae/rmse/r2 instead
    of just an internal Optuna val-split.

    Off by default (hp['eval_cv'] == 'none') since it multiplies runtime by
    the number of outer folds - each fold re-runs its own search from scratch.
    """
    outer_cv = (LeaveOneOut() if hp["eval_cv"] == "loo"
                else KFold(n_splits=hp["eval_splits"], shuffle=True, random_state=hp["seed"]))
    n_folds = outer_cv.get_n_splits(X)
    y_true, y_pred = [], []
    t0 = time.time()

    print(f"\n  [{name}] Nested eval (outer={hp['eval_cv']}, {n_folds} folds, "
          f"{hp['eval_n_trials']} trials/fold) over {len(X)} rows ...")

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]

        _, _, model, scaler = search_and_retrain(
            name, objective_fn, build_model_fn, X_train, y_train, hp, device,
            scale_features, n_trials=hp["eval_n_trials"])
        pred = predict_uM(model, scaler, X_test, hp, device, name)

        y_true.extend(np.asarray(y_test).tolist())
        y_pred.extend(np.asarray(pred).flatten().tolist())

        if i % max(1, n_folds // 10) == 0 or i == n_folds:
            print(f"    outer fold {i:>4}/{n_folds}  ({time.time() - t0:.0f}s elapsed)")

    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    metrics = {
        "nested_cv_mae":  float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "nested_cv_rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "nested_cv_r2":   float(r2_score(y_true_arr, y_pred_arr)),
    }
    print(f"  [{name}] Nested eval done in {time.time() - t0:.0f}s | "
          f"MAE={metrics['nested_cv_mae']:.4f}  RMSE={metrics['nested_cv_rmse']:.4f}  "
          f"R2={metrics['nested_cv_r2']:.4f}")
    return metrics


# 7. ENTRY POINT
def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for MLP / 1D-CNN")
    parser.add_argument("--features-csv", type=str, default=HP["features_csv"])
    parser.add_argument("--signals-csv",  type=str, default=HP["signals_csv"])
    parser.add_argument("--models", type=str, nargs="+", default=HP["models"], choices=["mlp", "cnn"])
    parser.add_argument("--n-trials", type=int, default=HP["n_trials"])
    parser.add_argument("--max-epochs", type=int, default=HP["max_epochs"])
    parser.add_argument("--patience", type=int, default=HP["patience"])
    parser.add_argument("--val-split", type=float, default=HP["val_split"])
    parser.add_argument("--seed", type=int, default=HP["seed"])
    parser.add_argument("--output", type=str, default=HP["output_json"])
    parser.add_argument("--eval-cv", type=str, default=HP["eval_cv"], choices=["none", "loo", "kfold"])
    parser.add_argument("--eval-splits", type=int, default=HP["eval_splits"])
    parser.add_argument("--eval-n-trials", type=int, default=HP["eval_n_trials"])
    parser.add_argument("--log-csv", type=str, default=HP["log_csv"])
    parser.add_argument("--models-dir", type=str, default=HP["models_dir"])
    args = parser.parse_args()

    hp = dict(HP)
    hp.update(features_csv=args.features_csv, signals_csv=args.signals_csv, models=args.models,
               n_trials=args.n_trials, max_epochs=args.max_epochs, patience=args.patience,
               val_split=args.val_split, seed=args.seed, output_json=args.output,
               eval_cv=args.eval_cv, eval_splits=args.eval_splits, eval_n_trials=args.eval_n_trials,
               log_csv=args.log_csv, models_dir=args.models_dir)

    print("=" * 60)
    print("  MLP / 1D-CNN - Optuna Hyperparameter Search (with pruning)")
    print("=" * 60)

    set_seed(hp["seed"])
    device = get_device()
    os.makedirs(hp["models_dir"], exist_ok=True)

    results = {}
    trial_frames = []

    if "mlp" in hp["models"]:
        X, y, feature_cols = load_mlp_features(hp)
        if hp["eval_cv"] != "none":
            nested_metrics = run_nested_eval("mlp", objective_mlp, build_mlp, X, y, hp, device, scale_features=True)
        result, trials_df, model, scaler = run_study(
            "mlp", objective_mlp, build_mlp, X, y, hp, device, scale_features=True)
        result["features_csv"] = hp["features_csv"]
        result["feature_columns"] = feature_cols
        if hp["eval_cv"] != "none":
            result.update(nested_metrics)
        results["mlp"] = result
        trial_frames.append(trials_df)
        torch.save(model.state_dict(), f"{hp['models_dir']}/mlp_tuned.pt")
        joblib.dump(scaler, f"{hp['models_dir']}/mlp_scaler.joblib")
        print(f"  Saved: {hp['models_dir']}/mlp_tuned.pt")
        print(f"  Saved: {hp['models_dir']}/mlp_scaler.joblib")

    if "cnn" in hp["models"]:
        X, y = load_cnn_signals(hp)
        if hp["eval_cv"] != "none":
            nested_metrics = run_nested_eval("cnn", objective_cnn, build_cnn, X, y, hp, device, scale_features=False)
        result, trials_df, model, _ = run_study("cnn", objective_cnn, build_cnn, X, y, hp, device)
        result["signals_csv"] = hp["signals_csv"]
        result["signal_len"] = hp["signal_len"]
        if hp["eval_cv"] != "none":
            result.update(nested_metrics)
        results["cnn"] = result
        trial_frames.append(trials_df)
        torch.save(model.state_dict(), f"{hp['models_dir']}/cnn_tuned.pt")
        print(f"  Saved: {hp['models_dir']}/cnn_tuned.pt")

    with open(hp["output_json"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {hp['output_json']}")

    os.makedirs(os.path.dirname(hp["log_csv"]) or ".", exist_ok=True)
    pd.concat(trial_frames, ignore_index=True).to_csv(hp["log_csv"], index=False)
    print(f"  Saved: {hp['log_csv']}")

    print("\n  Summary:")
    for name, r in results.items():
        print(f"    {name:<4}: val MAE={r['val_mae_uM']:.4f} uM  "
              f"({r['n_pruned']}/{r['n_trials']} pruned)  | winning params: {r['best_params']}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
