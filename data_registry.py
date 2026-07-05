"""

Single source of truth for every dataset the two Streamlit pages can see.

Two kinds of sources are exposed through the same `get_available_sources()`
catalog:

- Static, always-available sources loaded from disk (`real`,
  `stable_augmented`) - cached with `st.cache_data`.
- Session-scoped sources (`custom_generated`, `gan_generated`) written by the
  "Data Generation" page into `st.session_state['datasets']` and read back
  here. They only appear once the user generates them.

  
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from augmentation_pipeline import AugmentedDataset, FEATURE_COLUMNS_BY_SUITE, FeatureVectorizer, FullRangeDataAugmentor

# Section 0 - configuration
import paths

DATASET_PATH = paths.CALIBRATION_WORKBOOK
POTENTIAL_GRID_PATH = paths.POTENTIAL_GRID_CSV
REAL_SIGNALS_PATH = paths.REAL_SIGNALS_CSV
STABLE_AUGMENTED_PATH = paths.AUGMENTED_SIGNALS_CSV

# Replicate labels for the calibration workbook columns (same list used by
# full_range_data_augmentation.ipynb and data_analysis.ipynb).
CONCENTRATIONS_uM = [
    100, 100, 100, 50, 50, 50, 25, 25, 25, 15, 15, 15,
    10, 10, 10, 7.5, 7.5, 7.5, 5, 5, 5, 2.5, 2.5,
    1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.25, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1,
]

REPRESENTATIVE_CONCENTRATIONS_uM = [0.1, 0.5, 2.5, 10.0, 25.0, 75.0]

# Pre-generated static sample batches - used only as the Page 1 comparison
# overlay against the physics-augmented batch (unchanged legacy behaviour).
GAN_SAMPLE_DIRS = {
    'Trained on Real data': paths.GAN_OUTPUT_DIR / 'samples_trained_on_real',
    'Trained on Combined (real + physics-aug) data': paths.GAN_OUTPUT_DIR / 'samples_trained_on_combined',
}

# Saved generator checkpoints - used by `gan_inference.py` to draw fresh
# signals on demand from the "Generate GAN data" menu.
GAN_MODEL_DIRS = {
    'Trained on Real data': paths.GAN_OUTPUT_DIR / 'models_trained_on_real',
    'Trained on Combined (real + physics-aug) data': paths.GAN_OUTPUT_DIR / 'models_trained_on_combined',
}

SOURCE_LABELS = {
    'real': 'Real (lab-measured)',
    'stable_augmented': 'Stable augmented (physics-informed, recommended settings)',
    'custom_generated': 'Custom generated (Data Generation page)',
    'gan_generated': 'GAN synthetic (Data Generation page)',
}


# Section 1 - cached, expensive, rarely-changing resources
@st.cache_resource(show_spinner='Loading calibration workbook and fitting the noise model...')
def load_augmentor() -> FullRangeDataAugmentor:
    return FullRangeDataAugmentor.from_excel(DATASET_PATH, concentrations_uM=CONCENTRATIONS_uM)


@st.cache_data(show_spinner=False)
def load_real_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E = pd.read_csv(POTENTIAL_GRID_PATH).values.flatten()
    df = pd.read_csv(REAL_SIGNALS_PATH)
    y = df['concentration'].to_numpy()
    X = df.drop(columns=['concentration']).to_numpy()
    return E, X, y


@st.cache_data(show_spinner=False)
def load_stable_augmented_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E = pd.read_csv(POTENTIAL_GRID_PATH).values.flatten()
    df = pd.read_csv(STABLE_AUGMENTED_PATH)
    y = df['concentration'].to_numpy()
    X = df.drop(columns=['concentration']).to_numpy()
    return E, X, y


@st.cache_data(show_spinner=False)
def load_gan_sample_batch(gan_dir: str, architecture: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(f'{gan_dir}/{architecture}_signals.csv')
    return df.drop(columns=['concentration']).to_numpy(), df['concentration'].to_numpy()


# Section 2 - session-scoped dataset cache (written by the Data Generation page)
def save_dataset_to_session(key: str, E: np.ndarray, X: np.ndarray, y: np.ndarray, meta: dict) -> None:
    st.session_state.setdefault('datasets', {})[key] = dict(
        E=np.asarray(E), X=np.asarray(X), y=np.asarray(y), n=len(y), meta=meta)


def clear_dataset_from_session(key: str) -> None:
    st.session_state.get('datasets', {}).pop(key, None)


# Section 3 - the unified dataset catalog
def get_available_sources() -> dict[str, dict]:
    """key -> {label, E, X, y, n, meta, available} for every known dataset."""
    sources: dict[str, dict] = {}

    E, X, y = load_real_batch()
    sources['real'] = dict(label=SOURCE_LABELS['real'], E=E, X=X, y=y, n=len(y),
                            meta={'origin': REAL_SIGNALS_PATH}, available=True)

    E, X, y = load_stable_augmented_batch()
    sources['stable_augmented'] = dict(
        label=SOURCE_LABELS['stable_augmented'], E=E, X=X, y=y, n=len(y),
        meta={'origin': STABLE_AUGMENTED_PATH, 'note': 'Generated once with AugmentationConfig() defaults.'},
        available=True)

    session_datasets = st.session_state.get('datasets', {})
    for key in ('custom_generated', 'gan_generated'):
        entry = session_datasets.get(key)
        if entry:
            sources[key] = dict(label=SOURCE_LABELS[key], E=entry['E'], X=entry['X'], y=entry['y'],
                                 n=entry['n'], meta=entry['meta'], available=True)
        else:
            sources[key] = dict(label=SOURCE_LABELS[key], E=None, X=None, y=None, n=0,
                                 meta={}, available=False)

    return sources


def format_source_option(key: str, sources: dict) -> str:
    s = sources[key]
    if not s['available']:
        return f"{s['label']} - not generated yet"
    return f"{s['label']} (n={s['n']})"


def combine_sources(keys: Sequence[str], sources: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate the chosen sources into one (E, X, y) batch.

    All sources share the same 229-point potential grid, so the first
    available one is reused as-is.
    """
    Xs, ys, E = [], [], None
    for key in keys:
        s = sources[key]
        if not s['available']:
            continue
        if E is None:
            E = s['E']
        Xs.append(s['X'])
        ys.append(s['y'])
    if not Xs:
        raise ValueError('No available sources were selected.')
    return E, np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# Section 4 - feature extraction shared by both pages
def featurize(E: np.ndarray, X: np.ndarray, y: np.ndarray, suite: str = 'core') -> pd.DataFrame:
    """Vectorize a raw (already baseline-subtracted) signal batch into a
    tidy feature DataFrame for the given suite ('core' / 'extended' /
    'experimental' / 'raw_signal'), plus a 'concentration' column.

    Works uniformly for real, physics-augmented, GAN and user-generated
    batches since they all share the same on-disk row format.

    'raw_signal' (torch_models.CNN1DRegressor's input) is a pass-through,
    not an engineered suite: it skips FeatureVectorizer/Signal entirely and
    returns the I(E) samples as-is. This matters because Signal's
    constructor Savitzky-Golay-smooths its input for peak detection - fine
    for the engineered features, but the CNN was trained directly on
    training/tune_dl_hparams.py's raw/raw_signals_real.csv values, so
    smoothing here would be a train/inference distribution mismatch.
    """
    if suite == 'raw_signal':
        cols = FEATURE_COLUMNS_BY_SUITE['raw_signal']
        df = pd.DataFrame(np.asarray(X, dtype=float), columns=cols)
        df['concentration'] = np.asarray(y, dtype=float)
        return df

    E = np.asarray(E, dtype=float)
    dataset = AugmentedDataset(potential_grid_V=E, concentrations_uM=np.asarray(y, dtype=float),
                                signals_uA=np.asarray(X, dtype=float))
    vectorizer = FeatureVectorizer(E, blank_baseline_uA=np.zeros_like(E))
    return vectorizer.synthetic_feature_dataframe(dataset, suite=suite)
