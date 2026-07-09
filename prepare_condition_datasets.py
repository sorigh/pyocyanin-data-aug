"""
Build per-condition datasets used by run_condition_sweep.py.

Makes two new training conditions not present on disk:
a 'gan_only' dataset of 600 generated signals and
a 'combined_all' pool (940 signals) (real (40) + physics-augmented (300) + gan_only (600)).
-
Excludes generators trained on physics-augmented data from the GAN-only 
condition to strictly prevent circular training data leakage.

runs the exact same FeatureVectorizer/AugmentedDataset pipeline
data_registry.featurize() uses for the Streamlit app, just without the
streamlit dependency  (this needs to run standalone, e.g. before handing the
data off to training/ scripts).
Usage
-----
    python prepare_condition_datasets.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import paths
from augmentation_pipeline import AugmentedDataset, FeatureVectorizer


def load_raw_signals(csv_path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    I_cols = [c for c in df.columns if c.startswith('I_')]
    return df[I_cols].to_numpy(dtype=float), df['concentration'].to_numpy(dtype=float)


def featurize(E: np.ndarray, X: np.ndarray, y: np.ndarray, suite: str) -> pd.DataFrame:
    dataset = AugmentedDataset(potential_grid_V=E, concentrations_uM=y, signals_uA=X)
    vectorizer = FeatureVectorizer(E, blank_baseline_uA=np.zeros_like(E))
    return vectorizer.synthetic_feature_dataframe(dataset, suite=suite)


def load_lab_features(suite: str) -> pd.DataFrame:
    path = {'core': paths.VECTORIZED_CORE_CSV, 'extended': paths.VECTORIZED_EXTENDED_CSV,
            'experimental': paths.VECTORIZED_EXPERIMENTAL_CSV}[suite]
    df = pd.read_csv(path)
    return df.drop(columns=['sig_id'], errors='ignore')


def load_physics_aug_features(suite: str) -> pd.DataFrame:
    return pd.read_csv(paths.VECTORIZED_DIR / f'full_augmented_{suite}.csv')


def main() -> None:
    paths.CONDITIONS_DIR.mkdir(parents=True, exist_ok=True)

    E = pd.read_csv(paths.POTENTIAL_GRID_CSV).to_numpy(dtype=float).flatten()
    X_wgangp, y_wgangp = load_raw_signals(paths.WGANGP_SIGNALS_CSV)
    X_timegan, y_timegan = load_raw_signals(paths.TIMEGAN_SIGNALS_CSV)
    print(f'Loaded GAN sources: wgangp_real n={len(y_wgangp)}, timegan_real n={len(y_timegan)}')

    X_gan_only = np.concatenate([X_wgangp, X_timegan], axis=0)
    y_gan_only = np.concatenate([y_wgangp, y_timegan], axis=0)

    gan_raw_df = pd.DataFrame(X_gan_only, columns=[f'I_{i}' for i in range(X_gan_only.shape[1])])
    gan_raw_df.insert(0, 'concentration', y_gan_only)
    gan_raw_df.to_csv(paths.GAN_ONLY_RAW_SIGNALS_CSV, index=False)
    print(f'Saved: {paths.GAN_ONLY_RAW_SIGNALS_CSV}  (n={len(gan_raw_df)})')

    X_real, y_real = load_raw_signals(paths.REAL_SIGNALS_CSV)
    X_physics_aug, y_physics_aug = load_raw_signals(paths.AUGMENTED_SIGNALS_CSV)
    X_combined_raw = np.concatenate([X_real, X_physics_aug, X_gan_only], axis=0)
    y_combined_raw = np.concatenate([y_real, y_physics_aug, y_gan_only], axis=0)
    combined_raw_df = pd.DataFrame(X_combined_raw, columns=[f'I_{i}' for i in range(X_combined_raw.shape[1])])
    combined_raw_df.insert(0, 'concentration', y_combined_raw)
    combined_raw_df.to_csv(paths.COMBINED_ALL_RAW_SIGNALS_CSV, index=False)
    print(f'Saved: {paths.COMBINED_ALL_RAW_SIGNALS_CSV}  (n={len(combined_raw_df)})')

    for suite in paths.FEATURE_SUITES:
        gan_only_df = featurize(E, X_gan_only, y_gan_only, suite)
        gan_only_df.to_csv(paths.gan_only_features_csv(suite), index=False)
        print(f'Saved: {paths.gan_only_features_csv(suite)}  '
              f'(n={len(gan_only_df)}, features={list(gan_only_df.columns[:-1])})')

        lab_df = load_lab_features(suite)
        physics_aug_df = load_physics_aug_features(suite)
        combined_df = pd.concat([lab_df, physics_aug_df, gan_only_df], ignore_index=True)
        combined_df.to_csv(paths.combined_all_features_csv(suite), index=False)
        print(f'Saved: {paths.combined_all_features_csv(suite)}  (n={len(combined_df)})')

    print('\nDone.')


if __name__ == '__main__':
    main()
