"""
Nested LOOCV/K-Fold + Grid/Bayesian Hyperparameter Search for 6 ML Models

Why this script
----------------
model_training.ipynb runs a nested LeaveOneOut(outer) / KFold(inner)
search loop for six regressors on the 'core' feature suite (4 features, 40
real signals) 
Same methodology here (LOOCV or K-fold outer loop, inner
search MAE-scored)
Script for DGX.

REGRESSORS (ridge, elastic_net, decision_tree,
random_forest, svr, xgboost) + GridSearchCV grid and a
BayesSearchCV search space (--search-method).
Default experimental.


best_params.json decided by the last search pass, not by the outer folds.

SVR/Ridge/ElasticNet get a StandardScaler in front of them because they're
scale-sensitive and the raw feature magnitudes here span several orders of
magnitude (peak_current ~tens of uA, peak_FWHM ~0.08 V) 
 Tree-based models (decision_tree, random_forest, gboost) don't need it.

Usage
-----
    python training/tune_ml_hparams.py
    python training/tune_ml_hparams.py --models svr --n-iter 50
    python training/tune_ml_hparams.py --models ridge elastic_net decision_tree --search-method grid
    python training/tune_ml_hparams.py --features-csv vectorized/combined_experimental.csv \\
        --outer-cv kfold --outer-splits 5

Output
------
    models/svr_xgb_best_params.json   - winning hyperparams + nested CV metrics
    logs/svr_xgb_nested_cv.csv        - per-outer-fold predictions & best params
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBRegressor


# 0. HYPERPARAMETERS
HP = {
    "features_csv":  "vectorized/experimental.csv",
    "target_col":    "concentration",
    "drop_cols":     ["sig_id"],          # non-feature columns to ignore if present

    "outer_cv":      "loo",               # "loo" | "kfold"
    "outer_splits":  5,                   # only used when outer_cv == "kfold"
    "inner_splits":  3,
    "search_method": "bayes",             # "bayes" | "grid"
    "n_iter":        32,                  # BayesSearchCV iterations per fold
    "n_jobs":        -1,
    "seed":          42,
    "scoring":       "neg_mean_absolute_error",

    "models":        ["svr", "xgboost"],
    "output_json":   "models/svr_xgb_best_params.json",
    "log_csv":       "logs/svr_xgb_nested_cv.csv",
}


# 1. MODEL SPECS (pipeline + a Bayes search space + a Grid search grid, per model)
def build_model_specs(seed: int) -> dict:
    return {
        "ridge": {
            "pipeline": make_pipeline(StandardScaler(), Ridge()),
            "bayes_spaces": {
                "ridge__alpha": Real(1e-3, 1e3, prior="log-uniform"),
            },
            "grid_spaces": {
                "ridge__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                "ridge__fit_intercept": [True, False],
            },
        },
        "elastic_net": {
            "pipeline": make_pipeline(StandardScaler(), ElasticNet(max_iter=20000, random_state=seed)),
            "bayes_spaces": {
                "elasticnet__alpha":    Real(1e-3, 1e2, prior="log-uniform"),
                "elasticnet__l1_ratio": Real(0.0, 1.0),
            },
            "grid_spaces": {
                "elasticnet__alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
                "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },
        "decision_tree": {
            "pipeline": make_pipeline(DecisionTreeRegressor(random_state=seed)),
            "bayes_spaces": {
                "decisiontreeregressor__max_depth":        Integer(2, 20),
                "decisiontreeregressor__min_samples_split": Integer(2, 20),
                "decisiontreeregressor__min_samples_leaf":  Integer(1, 10),
                "decisiontreeregressor__ccp_alpha":         Real(1e-5, 1e-1, prior="log-uniform"),
            },
            "grid_spaces": {
                "decisiontreeregressor__max_depth":        [3, 5, 8, 12, None],
                "decisiontreeregressor__min_samples_split": [2, 5, 10],
                "decisiontreeregressor__min_samples_leaf":  [1, 2, 4],
                "decisiontreeregressor__ccp_alpha":         [0.0, 0.001, 0.01],
            },
        },
        "random_forest": {
            "pipeline": make_pipeline(RandomForestRegressor(random_state=seed, n_jobs=1)),
            "bayes_spaces": {
                "randomforestregressor__n_estimators":     Integer(50, 500),
                "randomforestregressor__max_depth":        Integer(2, 20),
                "randomforestregressor__min_samples_split": Integer(2, 20),
                "randomforestregressor__min_samples_leaf":  Integer(1, 10),
                "randomforestregressor__max_features":      Categorical(["sqrt", "log2", 1.0]),
            },
            "grid_spaces": {
                "randomforestregressor__n_estimators":     [100, 200, 300],
                "randomforestregressor__max_depth":        [5, 10, None],
                "randomforestregressor__min_samples_split": [2, 5],
                "randomforestregressor__min_samples_leaf":  [1, 2],
                "randomforestregressor__max_features":      ["sqrt", "log2"],
            },
        },
        "svr": {
            "pipeline": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            "bayes_spaces": {
                "svr__C":       Real(1e-2, 1e3,  prior="log-uniform"),
                "svr__epsilon": Real(1e-3, 2.0,  prior="log-uniform"),
                "svr__gamma":   Real(1e-4, 10.0, prior="log-uniform"),
            },
            "grid_spaces": {
                "svr__C":       [0.1, 1, 10, 100, 1000],
                "svr__epsilon": [0.001, 0.01, 0.1, 0.5, 1.0],
                "svr__gamma":   [0.0001, 0.001, 0.01, 0.1, 1.0],
            },
        },
        "xgboost": {
            "pipeline": make_pipeline(
                SimpleImputer(strategy="median"),
                XGBRegressor(objective="reg:squarederror", random_state=seed,
                             n_jobs=1, verbosity=0),
            ),
            "bayes_spaces": {
                "xgbregressor__n_estimators":     Integer(50, 500),
                "xgbregressor__max_depth":        Integer(2, 8),
                "xgbregressor__learning_rate":    Real(1e-2, 0.3,  prior="log-uniform"),
                "xgbregressor__subsample":        Real(0.6, 1.0),
                "xgbregressor__colsample_bytree": Real(0.6, 1.0),
                "xgbregressor__min_child_weight": Integer(1, 10),
                "xgbregressor__gamma":            Real(1e-8, 5.0,  prior="log-uniform"),
                "xgbregressor__reg_alpha":        Real(1e-8, 10.0, prior="log-uniform"),
                "xgbregressor__reg_lambda":       Real(1e-3, 10.0, prior="log-uniform"),
            },
            "grid_spaces": {
                "xgbregressor__n_estimators":     [100, 200, 300],
                "xgbregressor__max_depth":        [3, 5, 8],
                "xgbregressor__learning_rate":    [0.01, 0.05, 0.1, 0.3],
                "xgbregressor__subsample":        [0.7, 1.0],
                "xgbregressor__colsample_bytree": [0.7, 1.0],
            },
        },
    }


# 1b. SEARCH FACTORY (Bayes vs Grid, same call signature either way)
def make_search(spec: dict, hp: dict, cv) -> "BayesSearchCV | GridSearchCV":
    estimator = clone(spec["pipeline"])
    if hp["search_method"] == "grid":
        return GridSearchCV(
            estimator=estimator,
            param_grid=spec["grid_spaces"],
            cv=cv,
            scoring=hp["scoring"],
            n_jobs=hp["n_jobs"],
            refit=True,
        )
    return BayesSearchCV(
        estimator=estimator,
        search_spaces=spec["bayes_spaces"],
        n_iter=hp["n_iter"],
        cv=cv,
        scoring=hp["scoring"],
        n_jobs=hp["n_jobs"],
        random_state=hp["seed"],
        refit=True,
    )


# 2. DATA
def load_data(hp: dict) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(hp["features_csv"])
    drop = [c for c in hp["drop_cols"] if c in df.columns] + [hp["target_col"]]
    X = df.drop(columns=drop)
    y = df[hp["target_col"]]
    print(f"  Loaded {hp['features_csv']}  |  {X.shape[0]} rows  |  {X.shape[1]} features: {list(X.columns)}")
    return X, y


# 3. NESTED CV (generalisation-error estimate)
def run_nested_cv(name: str, spec: dict, X: pd.DataFrame, y: pd.Series, hp: dict) -> tuple[dict, list, pd.DataFrame]:
    outer_cv = (LeaveOneOut() if hp["outer_cv"] == "loo"
                else KFold(n_splits=hp["outer_splits"], shuffle=True, random_state=hp["seed"]))
    n_folds = outer_cv.get_n_splits(X)

    inner_cv = KFold(n_splits=hp["inner_splits"], shuffle=True, random_state=hp["seed"])

    y_true, y_pred, fold_params, fold_records = [], [], [], []
    t0 = time.time()

    print(f"\n  [{name}] Nested CV  (outer={hp['outer_cv']}, {n_folds} folds, inner={hp['inner_splits']}-fold, "
          f"search={hp['search_method']}, n_iter={hp['n_iter']}) over {len(X)} rows ...")

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        opt = make_search(spec, hp, inner_cv)
        opt.fit(X_train, y_train)
        pred = opt.best_estimator_.predict(X_test)

        y_true.extend(np.asarray(y_test))
        y_pred.extend(np.asarray(pred))
        fold_params.append(opt.best_params_)
        # One row per outer fold - test_idx/y_true/y_pred may hold more than
        # one sample when outer_cv == "kfold", so they're stored as JSON
        # lists rather than forced into separate same-length columns.
        fold_records.append({
            "model": name,
            "fold": i,
            "test_idx": json.dumps(list(map(int, test_idx))),
            "y_true": json.dumps(np.asarray(y_test).tolist()),
            "y_pred": json.dumps(np.asarray(pred).tolist()),
            **opt.best_params_,
        })

        if i % max(1, n_folds // 10) == 0 or i == n_folds:
            print(f"    outer fold {i:>4}/{n_folds}  ({time.time() - t0:.0f}s elapsed)")

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    metrics = {
        "mae":  float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2":   float(r2_score(y_true_arr, y_pred_arr)),
    }
    print(f"  [{name}] Nested CV done in {time.time() - t0:.0f}s | "
          f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}")

    log_df = pd.DataFrame(fold_records)
    return metrics, fold_params, log_df


# 4. FINAL SEARCH ON THE FULL DATASET (the actual "winning" config)
def final_search(name: str, spec: dict, X: pd.DataFrame, y: pd.Series, hp: dict) -> tuple[dict, float]:
    inner_cv = KFold(n_splits=hp["inner_splits"], shuffle=True, random_state=hp["seed"])
    opt = make_search(spec, hp, inner_cv)
    opt.fit(X, y)
    print(f"  [{name}] Full-data search best CV MAE: {-opt.best_score_:.4f}")
    return dict(opt.best_params_), float(opt.best_score_)


MODEL_CHOICES = ["ridge", "elastic_net", "decision_tree", "random_forest", "svr", "xgboost"]


# 5. ENTRY POINT
def main():
    parser = argparse.ArgumentParser(description="Nested LOOCV/K-Fold + Grid/BayesSearchCV for 6 ML models")
    parser.add_argument("--features-csv", type=str, default=HP["features_csv"])
    parser.add_argument("--models", type=str, nargs="+", default=HP["models"],
                        choices=MODEL_CHOICES)
    parser.add_argument("--outer-cv", type=str, default=HP["outer_cv"], choices=["loo", "kfold"])
    parser.add_argument("--outer-splits", type=int, default=HP["outer_splits"])
    parser.add_argument("--inner-splits", type=int, default=HP["inner_splits"])
    parser.add_argument("--search-method", type=str, default=HP["search_method"], choices=["bayes", "grid"])
    parser.add_argument("--n-iter", type=int, default=HP["n_iter"])
    parser.add_argument("--n-jobs", type=int, default=HP["n_jobs"])
    parser.add_argument("--seed", type=int, default=HP["seed"])
    parser.add_argument("--output", type=str, default=HP["output_json"])
    parser.add_argument("--log-csv", type=str, default=HP["log_csv"])
    args = parser.parse_args()

    hp = dict(HP)
    hp.update(
        features_csv=args.features_csv, models=args.models, outer_cv=args.outer_cv,
        outer_splits=args.outer_splits, inner_splits=args.inner_splits,
        search_method=args.search_method, n_iter=args.n_iter,
        n_jobs=args.n_jobs, seed=args.seed, output_json=args.output, log_csv=args.log_csv,
    )

    print("=" * 60)
    print("  ML Regressors - Nested LOOCV/K-Fold + Grid/Bayesian Hyperparameter Search")
    print("=" * 60)

    X, y = load_data(hp)
    specs = build_model_specs(hp["seed"])

    results = {}
    log_frames = []

    for name in hp["models"]:
        spec = specs[name]
        nested_metrics, _, log_df = run_nested_cv(name, spec, X, y, hp)
        best_params, best_cv_score = final_search(name, spec, X, y, hp)

        results[name] = {
            "search_method":          hp["search_method"],
            "best_params":            best_params,
            "final_search_cv_mae":    -best_cv_score,
            "nested_cv_mae":          nested_metrics["mae"],
            "nested_cv_rmse":         nested_metrics["rmse"],
            "nested_cv_r2":           nested_metrics["r2"],
            "n_samples":              len(X),
            "n_features":             X.shape[1],
            "feature_columns":        list(X.columns),
            "features_csv":           hp["features_csv"],
        }
        log_frames.append(log_df)

    os.makedirs(os.path.dirname(hp["output_json"]) or ".", exist_ok=True)
    with open(hp["output_json"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {hp['output_json']}")

    os.makedirs(os.path.dirname(hp["log_csv"]) or ".", exist_ok=True)
    pd.concat(log_frames, ignore_index=True).to_csv(hp["log_csv"], index=False)
    print(f"  Saved: {hp['log_csv']}")

    print("\n  Summary:")
    for name, r in results.items():
        print(f"    {name:<8}: nested MAE={r['nested_cv_mae']:.4f}  "
              f"RMSE={r['nested_cv_rmse']:.4f}  R2={r['nested_cv_r2']:.4f}  "
              f"| winning params: {r['best_params']}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
