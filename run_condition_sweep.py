"""
Orchestrates the lab / physics-augmented / GAN / combined training sweep
across four dataset conditions (lab, physics-augmented, GAN-only, and combined) 
and multiple feature suites.

Runs training/tune_ml_hparams.py and training/tune_dl_hparams.py
 once per (condition, feature suite) combination,  each as its own subprocess

 every run's output JSON into:

  - results/conditions/{lab,physics_aug,gan,combined}_best_models.csv
      one row per model = the best (suite, search-method) found for that
      condition, ranked by nested/LOO-CV MAE. These are the "4 tables of
      best models".
  - results/conditions/feature_count_accuracy_matrix.csv
      long-format: condition x feature suite (core=4, extended=7,
      experimental=23 features) x model -> MAE/RMSE/R2 - the "does more
      features help as the dataset grows" sweep, run in parallel.
  - results/conditions/all_runs_long.csv
      every individual run's full metrics.

for lab - leaveoneout (little data)
for physics aug 5-fold-cv (80/20)
for gan aug 5-fold-cv (80/20)

(GAN trained on combined excuded so no data leak)

First run prepare_condition_datasets.py to build the csvs

Usage
-----
    python run_condition_sweep.py --dry-run
    python run_condition_sweep.py --quick --conditions lab --suites core --ml-models ridge --parallel 2
    python run_condition_sweep.py --parallel 8                       # full sweep
    python run_condition_sweep.py --conditions lab gan --ml-models ridge svr
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import paths
from augmentation_pipeline import FEATURE_COLUMNS_BY_SUITE, SIGNAL_LEN

ROOT = Path(__file__).resolve().parent


def _resolve_tune_script(filename: str) -> Path:
    """Prefers ROOT/training/<filename> (this repo's layout), falls back to
    ROOT/<filename> (DGX's - tune_ml_hparams.py/tune_dl_hparams.py live at
    the repo root there, no training/ subfolder)."""
    nested = ROOT / "training" / filename
    return nested if nested.exists() else ROOT / filename


TUNE_ML_SCRIPT = _resolve_tune_script("tune_ml_hparams.py")
TUNE_DL_SCRIPT = _resolve_tune_script("tune_dl_hparams.py")

ML_MODELS = ["ridge", "elastic_net", "decision_tree", "random_forest", "svr", "xgboost"]
SEARCH_METHODS = ["grid", "bayes"]
SUITES = list(paths.FEATURE_SUITES)
SUITE_N_FEATURES = {suite: len(FEATURE_COLUMNS_BY_SUITE[suite]) for suite in SUITES}

CONDITIONS = {
    "lab": dict(
        outer_cv="loo", outer_splits=5,
        features={s: p for s, p in zip(
            SUITES, [paths.VECTORIZED_CORE_CSV, paths.VECTORIZED_EXTENDED_CSV, paths.VECTORIZED_EXPERIMENTAL_CSV])},
        signals_csv=paths.REAL_SIGNALS_CSV,
    ),
    "physics_aug": dict(
        outer_cv="kfold", outer_splits=5,
        features={s: paths.VECTORIZED_DIR / f"full_augmented_{s}.csv" for s in SUITES},
        signals_csv=paths.AUGMENTED_SIGNALS_CSV,
    ),
    "gan": dict(
        outer_cv="kfold", outer_splits=5,
        features={s: paths.gan_only_features_csv(s) for s in SUITES},
        signals_csv=paths.GAN_ONLY_RAW_SIGNALS_CSV,
    ),
    "combined": dict(
        outer_cv="kfold", outer_splits=5,
        features={s: paths.combined_all_features_csv(s) for s in SUITES},
        signals_csv=paths.COMBINED_ALL_RAW_SIGNALS_CSV,
    ),
}

# Mirrors CONDITIONS above, but with the flat relative paths training/*.py's
# own hardcoded defaults use (vectorized/..., raw/..., models/...) - for
# running this same orchestrator directly on the DGX checkout, which still
# has the pre-restructure flat layout (see training_scripts_dgx_paths /
# training_condition_sweep memory notes). Selected via --flat-paths.
# The 6 new condition CSVs (gan_only_*, combined_all_*) don't exist on DGX
# yet - copy them from this repo's data/vectorized/conditions/ into DGX's
# vectorized/ and raw/ folders (flattened, no 'conditions' subfolder) before
# running with --flat-paths there.
FLAT_CONDITIONS = {
    "lab": dict(
        outer_cv="loo", outer_splits=5,
        features={"core": "vectorized/core.csv", "extended": "vectorized/extended.csv",
                   "experimental": "vectorized/experimental.csv"},
        signals_csv="raw/raw_signals_real.csv",
    ),
    "physics_aug": dict(
        outer_cv="kfold", outer_splits=5,
        features={s: f"vectorized/full_augmented_{s}.csv" for s in SUITES},
        signals_csv="raw/raw_signals_augmented.csv",
    ),
    "gan": dict(
        outer_cv="kfold", outer_splits=5,
        features={s: f"vectorized/gan_only_{s}.csv" for s in SUITES},
        signals_csv="raw/gan_only_raw_signals.csv",
    ),
    "combined": dict(
        outer_cv="kfold", outer_splits=5,
        features={s: f"vectorized/combined_all_{s}.csv" for s in SUITES},
        signals_csv="raw/combined_all_raw_signals.csv",
    ),
}


@dataclass
class Job:
    condition: str
    model_type: str        # "ml" | "mlp" | "cnn"
    model: str
    suite: str | None       # None for cnn (raw signal, no suite)
    search_method: str      # "grid" | "bayes" | "optuna"
    cmd: list
    output_json: Path


def build_ml_jobs(conditions: list, suites: list, models: list, methods: list, hp: dict, work_dir: Path,
                   conditions_cfg: dict) -> list[Job]:
    jobs = []
    for cond in conditions:
        cfg = conditions_cfg[cond]
        for suite in suites:
            for model in models:
                for method in methods:
                    run_dir = work_dir / cond / suite
                    out_json = run_dir / f"{model}_{method}_best_params.json"
                    log_csv = run_dir / f"{model}_{method}_nested_cv.csv"
                    cmd = [
                        sys.executable, str(TUNE_ML_SCRIPT),
                        "--features-csv", str(cfg["features"][suite]),
                        "--models", model,
                        "--outer-cv", cfg["outer_cv"],
                        "--outer-splits", str(cfg["outer_splits"]),
                        "--search-method", method,
                        "--n-iter", str(hp["ml_n_iter"]),
                        "--n-jobs", str(hp["ml_n_jobs"]),
                        "--output", str(out_json),
                        "--log-csv", str(log_csv),
                    ]
                    jobs.append(Job(cond, "ml", model, suite, method, cmd, out_json))
    return jobs


def build_mlp_jobs(conditions: list, suites: list, hp: dict, work_dir: Path, conditions_cfg: dict) -> list[Job]:
    jobs = []
    for cond in conditions:
        cfg = conditions_cfg[cond]
        for suite in suites:
            run_dir = work_dir / cond / suite
            out_json = run_dir / "mlp_best_params.json"
            cmd = [
                sys.executable, str(TUNE_DL_SCRIPT),
                "--features-csv", str(cfg["features"][suite]),
                "--models", "mlp",
                "--n-trials", str(hp["dl_n_trials"]),
                "--max-epochs", str(hp["dl_max_epochs"]),
                "--patience", str(hp["dl_patience"]),
                "--eval-cv", cfg["outer_cv"],
                "--eval-splits", str(cfg["outer_splits"]),
                "--eval-n-trials", str(hp["dl_eval_n_trials"]),
                "--output", str(out_json),
                "--log-csv", str(run_dir / "mlp_studies.csv"),
                "--models-dir", str(run_dir),
            ]
            jobs.append(Job(cond, "mlp", "mlp", suite, "optuna", cmd, out_json))
    return jobs


def build_cnn_jobs(conditions: list, hp: dict, work_dir: Path, conditions_cfg: dict) -> list[Job]:
    jobs = []
    for cond in conditions:
        cfg = conditions_cfg[cond]
        run_dir = work_dir / cond
        out_json = run_dir / "cnn_best_params.json"
        cmd = [
            sys.executable, str(TUNE_DL_SCRIPT),
            "--signals-csv", str(cfg["signals_csv"]),
            "--models", "cnn",
            "--n-trials", str(hp["dl_n_trials"]),
            "--max-epochs", str(hp["dl_max_epochs"]),
            "--patience", str(hp["dl_patience"]),
            "--eval-cv", cfg["outer_cv"],
            "--eval-splits", str(cfg["outer_splits"]),
            "--eval-n-trials", str(hp["dl_eval_n_trials"]),
            "--output", str(out_json),
            "--log-csv", str(run_dir / "cnn_studies.csv"),
            "--models-dir", str(run_dir),
        ]
        jobs.append(Job(cond, "cnn", "cnn", None, "optuna", cmd, out_json))
    return jobs


def run_job(job: Job) -> tuple[Job, bool, str]:
    job.output_json.parent.mkdir(parents=True, exist_ok=True)
    log_path = job.output_json.with_suffix(".log")
    t0 = time.time()
    proc = subprocess.run(job.cmd, cwd=ROOT, capture_output=True, text=True)
    log_path.write_text(proc.stdout + "\n----- STDERR -----\n" + proc.stderr)
    ok = proc.returncode == 0
    tag = f"{job.condition}/{job.suite or 'raw'}/{job.model}/{job.search_method}"
    status = "OK" if ok else f"FAILED (see {log_path})"
    print(f"  [{tag}] {status}  ({time.time() - t0:.0f}s)")
    return job, ok, proc.stderr if not ok else ""


def parse_result(job: Job) -> dict | None:
    if not job.output_json.exists():
        return None
    data = json.loads(job.output_json.read_text())
    r = data.get(job.model)
    if not r:
        return None
    n_features = SUITE_N_FEATURES.get(job.suite) if job.suite else SIGNAL_LEN
    mae = r.get("nested_cv_mae", r.get("val_mae_uM"))
    rmse = r.get("nested_cv_rmse")
    r2 = r.get("nested_cv_r2")
    n_samples = r.get("n_samples", r.get("n_train"))
    return dict(
        condition=job.condition, model_type=job.model_type, model=job.model,
        suite=job.suite or "raw_signal", n_features=n_features, search_method=job.search_method,
        mae=mae, rmse=rmse, r2=r2, n_samples=n_samples, best_params=json.dumps(r.get("best_params", {})),
    )


def discover_jobs(work_dir: Path) -> list[Job]:
    """Reconstructs Job stubs from whatever *_best_params.json files already
    exist under work_dir, using the naming convention build_*_jobs() writes:
    {condition}/cnn_best_params.json, {condition}/{suite}/mlp_best_params.json,
    {condition}/{suite}/{model}_{method}_best_params.json.

    Lets --aggregate-only build the report tables from a directory that was
    populated by training runs done elsewhere (e.g. copied back from a DGX
    run of the plain training/tune_*_hparams.py scripts) without needing to
    re-run anything here.
    """
    jobs = []
    if not work_dir.exists():
        return jobs
    for condition_dir in sorted(p for p in work_dir.iterdir() if p.is_dir()):
        condition = condition_dir.name
        cnn_json = condition_dir / "cnn_best_params.json"
        if cnn_json.exists():
            jobs.append(Job(condition, "cnn", "cnn", None, "optuna", [], cnn_json))
        for suite_dir in sorted(p for p in condition_dir.iterdir() if p.is_dir()):
            suite = suite_dir.name
            mlp_json = suite_dir / "mlp_best_params.json"
            if mlp_json.exists():
                jobs.append(Job(condition, "mlp", "mlp", suite, "optuna", [], mlp_json))
            for json_path in suite_dir.glob("*_best_params.json"):
                if json_path.name == "mlp_best_params.json":
                    continue
                model, method = json_path.stem[: -len("_best_params")].rsplit("_", 1)
                jobs.append(Job(condition, "ml", model, suite, method, [], json_path))
    jobs += discover_jobs_flat(work_dir)
    return jobs


def discover_jobs_flat(work_dir: Path) -> list[Job]:
    """Reconstructs Job stubs from the FLAT filename convention
    dgx_run_ml_sweep.sh / dgx_run_dl_sweep.sh use (no subfolders, everything
    directly under work_dir):
      {condition}_{suite}_{method}_best_params.json   - ML; one file holds
          MULTIPLE model keys since each command groups several models
          (e.g. the "grid" file has ridge+elastic_net+decision_tree+random_forest)
      {condition}_{suite}_mlp_best_params.json        - DL MLP
      {condition}_cnn_best_params.json                - DL CNN (no suite)
    Condition names can contain underscores (physics_aug), so matching is
    done against the known CONDITIONS keys rather than a fixed split count.
    """
    jobs = []
    if not work_dir.exists():
        return jobs
    known_conditions = sorted(CONDITIONS, key=len, reverse=True)
    for json_path in sorted(work_dir.glob("*_best_params.json")):
        stem = json_path.stem[: -len("_best_params")]
        condition = next((c for c in known_conditions if stem == c or stem.startswith(c + "_")), None)
        if condition is None:
            continue
        rest = stem[len(condition):].lstrip("_")
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if rest == "cnn":
            jobs += [Job(condition, "cnn", model_key, None, "optuna", [], json_path) for model_key in data]
        elif rest.endswith("_mlp"):
            suite = rest[: -len("_mlp")]
            jobs += [Job(condition, "mlp", model_key, suite, "optuna", [], json_path) for model_key in data]
        elif "_" in rest:
            suite, _, method = rest.rpartition("_")
            jobs += [Job(condition, "ml", model_key, suite, method, [], json_path) for model_key in data]
    return jobs


def build_reports(rows: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("  No successful runs to report on.")
        return
    df = pd.DataFrame(rows).dropna(subset=["mae"])
    df.to_csv(results_dir / "all_runs_long.csv", index=False)
    print(f"  Saved: {results_dir / 'all_runs_long.csv'}  ({len(df)} rows)")

    matrix = (df.groupby(["condition", "suite", "n_features", "model"], as_index=False)["mae"]
              .min().sort_values(["condition", "n_features", "model"]))
    matrix_path = results_dir / "feature_count_accuracy_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    print(f"  Saved: {matrix_path}")

    for condition in df["condition"].unique():
        cond_df = df[df["condition"] == condition]
        best_idx = cond_df.groupby("model")["mae"].idxmin()
        best = cond_df.loc[best_idx].sort_values("mae")
        best = best[["model", "suite", "n_features", "search_method", "mae", "rmse", "r2", "n_samples", "best_params"]]
        out_path = results_dir / f"{condition}_best_models.csv"
        best.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}  ({len(best)} models)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate the lab/physics-aug/GAN/combined training sweep")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--suites", nargs="+", default=SUITES, choices=SUITES)
    parser.add_argument("--ml-models", nargs="+", default=ML_MODELS, choices=ML_MODELS)
    parser.add_argument("--search-methods", nargs="+", default=SEARCH_METHODS, choices=SEARCH_METHODS)
    parser.add_argument("--dl-models", nargs="+", default=["mlp", "cnn"], choices=["mlp", "cnn"])
    parser.add_argument("--parallel", type=int, default=4, help="max concurrent subprocesses")
    parser.add_argument("--dry-run", action="store_true", help="print the planned commands, run nothing")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip running anything - just scan --work-dir for existing *_best_params.json "
                             "files (e.g. copied back from a DGX run) and build the report tables")
    parser.add_argument("--quick", action="store_true", help="tiny search budgets, for a fast local smoke test")
    parser.add_argument("--ml-n-iter", type=int, default=None, help="BayesSearchCV iterations (grid ignores this)")
    parser.add_argument("--ml-n-jobs", type=int, default=1, help="n_jobs *within* each subprocess's own search")
    parser.add_argument("--dl-n-trials", type=int, default=None)
    parser.add_argument("--dl-eval-n-trials", type=int, default=None)
    parser.add_argument("--dl-max-epochs", type=int, default=None)
    parser.add_argument("--dl-patience", type=int, default=None)
    parser.add_argument("--flat-paths", action="store_true",
                        help="use the flat vectorized/raw/models relative-path convention "
                             "training/*.py's own defaults use, instead of paths.py's nested "
                             "data/ layout - for running this script directly on a DGX checkout "
                             "that hasn't been migrated to the new folder layout")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    conditions_cfg = FLAT_CONDITIONS if args.flat_paths else CONDITIONS
    work_dir = Path(args.work_dir) if args.work_dir else (
        Path("models/conditions") if args.flat_paths else paths.MODELS_DIR / "conditions")
    results_dir = Path(args.results_dir) if args.results_dir else (
        Path("results/conditions") if args.flat_paths else paths.CONDITIONS_RESULTS_DIR)

    defaults = dict(ml_n_iter=4, dl_n_trials=3, dl_eval_n_trials=2, dl_max_epochs=10, dl_patience=5) \
        if args.quick else dict(ml_n_iter=32, dl_n_trials=50, dl_eval_n_trials=20, dl_max_epochs=300, dl_patience=25)
    hp = dict(defaults)
    hp["ml_n_jobs"] = args.ml_n_jobs
    for key in ("ml_n_iter", "dl_n_trials", "dl_eval_n_trials", "dl_max_epochs", "dl_patience"):
        override = getattr(args, key)
        if override is not None:
            hp[key] = override

    if args.aggregate_only:
        jobs = discover_jobs(work_dir)
        print(f"Found {len(jobs)} existing result files under {work_dir}.")
        rows = [row for row in (parse_result(job) for job in jobs) if row]
        build_reports(rows, results_dir)
        print("\nDone.")
        return

    jobs = []
    jobs += build_ml_jobs(args.conditions, args.suites, args.ml_models, args.search_methods, hp, work_dir,
                          conditions_cfg)
    if "mlp" in args.dl_models:
        jobs += build_mlp_jobs(args.conditions, args.suites, hp, work_dir, conditions_cfg)
    if "cnn" in args.dl_models:
        jobs += build_cnn_jobs(args.conditions, hp, work_dir, conditions_cfg)

    print(f"Planned {len(jobs)} runs across {len(args.conditions)} conditions.")
    if args.dry_run:
        for job in jobs:
            print(" ", " ".join(job.cmd))
        return

    rows = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job, ok, _ = future.result()
            if ok:
                row = parse_result(job)
                if row:
                    rows.append(row)

    build_reports(rows, results_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
