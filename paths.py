"""

Single source of truth for every path used in the project. 
This is a convenience to avoid hardcoding paths in multiple places, 
and to make it easy to change the directory structure.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Source data (lab measurements)
DATASETS_DIR = ROOT / 'data'/ 'datasets'
CALIBRATION_WORKBOOK = DATASETS_DIR / 'Standard calibration in culture media_extended.xlsx'

# Raw signal batches
RAW_DIR = ROOT / 'data' / 'raw'
POTENTIAL_GRID_CSV = RAW_DIR / 'raw_potential_grid.csv'
REAL_SIGNALS_CSV = RAW_DIR / 'raw_signals_real.csv'
AUGMENTED_SIGNALS_CSV = RAW_DIR / 'raw_signals_augmented.csv'
COMBINED_SIGNALS_CSV = RAW_DIR / 'raw_signals_combined.csv'

# Small-interval variant (data_augmentation.ipynb)
POTENTIAL_GRID_SMALL_INTERVAL_CSV = RAW_DIR / 'raw_potential_grid_small_interval.csv'
AUGMENTED_SIGNALS_SMALL_INTERVAL_CSV = RAW_DIR / 'raw_signals_augmented_small_interval.csv'

# Engineered feature vectors
VECTORIZED_DIR = ROOT / 'data' / 'vectorized'
VECTORIZED_EXPERIMENTAL_CSV = VECTORIZED_DIR / 'experimental.csv'
VECTORIZED_CORE_CSV = VECTORIZED_DIR / 'core.csv'
VECTORIZED_EXTENDED_CSV = VECTORIZED_DIR / 'extended.csv'

# trained regressor bundles (export_models.py output, used in
# pages/Model_Training_Testing.py)
MODELS_DIR = ROOT / 'models'
REGRESSORS_DIR = MODELS_DIR / 'regressors'
FEATURE_METADATA_JSON = REGRESSORS_DIR / 'feature_metadata.json'
TUNED_MLP_STATE_DICT = REGRESSORS_DIR / 'mlp_tuned.pt'
TUNED_MLP_SCALER = REGRESSORS_DIR / 'mlp_scaler.joblib'
TUNED_CNN_STATE_DICT = REGRESSORS_DIR / 'cnn_tuned.pt'

# GAN checkpoints (models)
GAN_MODELS_DIR = MODELS_DIR / 'gan'
GAN_MODEL_DIR_REAL = GAN_MODELS_DIR / 'real'
GAN_MODEL_DIR_COMBINED = GAN_MODELS_DIR / 'combined'

# GAN-generated sample batches (data)
GAN_SAMPLES_DIR = ROOT / 'data' / 'generated' / 'gan'
TIMEGAN_SIGNALS_CSV = GAN_SAMPLES_DIR / 'real' / 'timegan_signals.csv'
WGANGP_SIGNALS_CSV = GAN_SAMPLES_DIR / 'real' / 'wgangp_signals.csv'
TIMEGAN_SIGNALS_TRAINED_COMBINED_CSV = GAN_SAMPLES_DIR / 'combined' / 'timegan_signals.csv'
WGANGP_SIGNALS_TRAINED_COMBINED_CSV = GAN_SAMPLES_DIR / 'combined' / 'wgangp_signals.csv'

# GAN training logs
GAN_LOGS_DIR = ROOT / 'logs' / 'gan'
TIMEGAN_LOG_REAL_CSV = GAN_LOGS_DIR / 'real' / 'timegan_training_history.csv'
WGANGP_LOG_REAL_CSV = GAN_LOGS_DIR / 'real' / 'wgangp_training_history.csv'
TIMEGAN_LOG_COMBINED_CSV = GAN_LOGS_DIR / 'combined' / 'timegan_training_history.csv'
WGANGP_LOG_COMBINED_CSV = GAN_LOGS_DIR / 'combined' / 'wgangp_training_history.csv'

# Ablation / evaluation results (data_augmentation.ipynb / full_range_data_augmentation.ipynb)
RESULTS_DIR = ROOT / 'results'
MODEL_EVALUATION_RESULTS_CSV = RESULTS_DIR / 'model_evaluation_results.csv'
AUGMENTATION_ABLATION_RESULTS_CSV = RESULTS_DIR / 'augmentation_ablation_results.csv'
FULL_RANGE_ABLATION_RESULTS_CSV = RESULTS_DIR / 'full_range_ablation_results_refined.csv'

# Per-condition datasets built by prepare_condition_datasets.py for the
# run_condition_sweep.py.
CONDITIONS_DIR = VECTORIZED_DIR / 'conditions'
FEATURE_SUITES = ('core', 'extended', 'experimental')
GAN_ONLY_RAW_SIGNALS_CSV = CONDITIONS_DIR / 'gan_only_raw_signals.csv'
COMBINED_ALL_RAW_SIGNALS_CSV = CONDITIONS_DIR / 'combined_all_raw_signals.csv'


def gan_only_features_csv(suite: str) -> Path:
    return CONDITIONS_DIR / f'gan_only_{suite}.csv'


def combined_all_features_csv(suite: str) -> Path:
    return CONDITIONS_DIR / f'combined_all_{suite}.csv'


# Results tables produced by run_condition_sweep.py
CONDITIONS_RESULTS_DIR = RESULTS_DIR / 'conditions'
CONDITION_NAMES = ('lab', 'physics_aug', 'gan', 'combined')
FEATURE_COUNT_ACCURACY_MATRIX_CSV = CONDITIONS_RESULTS_DIR / 'feature_count_accuracy_matrix.csv'


def condition_best_models_csv(condition: str) -> Path:
    return CONDITIONS_RESULTS_DIR / f'{condition}_best_models.csv'

# logs
LOGS_DIR = ROOT / 'logs'
TUNE_DL_HPARAMS_STUDIES_CSV = LOGS_DIR / 'tune_dl_hparams_studies.csv'
SVR_XGB_NESTED_CV_CSV = LOGS_DIR / 'svr_xgb_nested_cv.csv'

