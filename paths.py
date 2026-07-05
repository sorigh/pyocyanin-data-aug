"""

Single source of truth for every path used in the project. 
This is a convenience to avoid hardcoding paths in multiple places, 
and to make it easy to change the directory structure.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Source data (lab measurements)
DATASETS_DIR = ROOT / 'datasets'
CALIBRATION_WORKBOOK = DATASETS_DIR / 'Standard calibration in culture media_extended.xlsx'

# Raw signal batches
RAW_DIR = ROOT / 'raw'
POTENTIAL_GRID_CSV = RAW_DIR / 'raw_potential_grid.csv'
REAL_SIGNALS_CSV = RAW_DIR / 'raw_signals_real.csv'
AUGMENTED_SIGNALS_CSV = RAW_DIR / 'raw_signals_augmented.csv'
COMBINED_SIGNALS_CSV = RAW_DIR / 'raw_signals_combined.csv'

# Small-interval variant (data_augmentation.ipynb)
POTENTIAL_GRID_SMALL_INTERVAL_CSV = RAW_DIR / 'raw_potential_grid_small_interval.csv'
AUGMENTED_SIGNALS_SMALL_INTERVAL_CSV = RAW_DIR / 'raw_signals_augmented_small_interval.csv'

# Engineered feature vectors
VECTORIZED_DIR = ROOT / 'vectorized'
VECTORIZED_EXPERIMENTAL_CSV = VECTORIZED_DIR / 'experimental.csv'
VECTORIZED_CORE_CSV = VECTORIZED_DIR / 'core.csv'
VECTORIZED_EXTENDED_CSV = VECTORIZED_DIR / 'extended.csv'

# GAN checkpoints/samples/logs
GAN_OUTPUT_DIR = ROOT / 'gan_output'
TIMEGAN_SIGNALS_CSV = GAN_OUTPUT_DIR / 'samples_trained_on_real' / 'timegan_signals.csv'
WGANGP_SIGNALS_CSV = GAN_OUTPUT_DIR / 'samples_trained_on_real' / 'wgangp_signals.csv'
TIMEGAN_SIGNALS_TRAINED_COMBINED_CSV = GAN_OUTPUT_DIR / 'samples_trained_on_combined' / 'timegan_signals.csv'
WGANGP_SIGNALS_TRAINED_COMBINED_CSV = GAN_OUTPUT_DIR / 'samples_trained_on_combined' / 'wgangp_signals.csv'

# trained regressor bundles (export_models.py output, used in
# pages/Model_Training_Testing.py)
MODELS_DIR = ROOT / 'models'
FEATURE_METADATA_JSON = MODELS_DIR / 'feature_metadata.json'

# Tuning script outputs (training/tune_ml_hparams.py,
# training/tune_dl_hparams.py) used in
# model_registry.py and export_models.py
RESULTS_DIR = ROOT / 'results'
RESULTS_MODELS_DIR = RESULTS_DIR / 'models'
TUNED_MLP_STATE_DICT = RESULTS_MODELS_DIR / 'mlp_tuned.pt'
TUNED_MLP_SCALER = RESULTS_MODELS_DIR / 'mlp_scaler.joblib'
TUNED_CNN_STATE_DICT = RESULTS_MODELS_DIR / 'cnn_tuned.pt'

# Ablation study results (data_augmentation.ipynb / full_range_data_augmentation.ipynb)
AUGMENTATION_ABLATION_RESULTS_CSV = RESULTS_DIR / 'augmentation_ablation_results.csv'
FULL_RANGE_ABLATION_RESULTS_CSV = RESULTS_DIR / 'full_range_ablation_results_refined.csv'

# logs
LOGS_DIR = ROOT / 'logs'

