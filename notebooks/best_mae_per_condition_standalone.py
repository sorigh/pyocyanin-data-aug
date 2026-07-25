# Standalone re-plot of "Best MAE per model, across conditions" (originally
# notebooks/condition_sweep_analysis.ipynb, section 5), as a horizontal bar
# chart so the condition labels read normally instead of at an angle, with
# the legend moved off the title.
#
# Reads the already-persisted per-condition result tables from
# results/conditions/{condition}_best_models.csv (written by section 6 of
# condition_sweep_analysis.ipynb) - no need to re-run the sweep or rebuild
# `df`/`best_tables` from the raw `models/` JSON files.

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# --- locate the repo root from wherever this notebook lives -----------------
here = Path().resolve()
ROOT_DIR = next(p for p in [here, *here.parents] if (p / 'paths.py').exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import paths
from plot_style import apply_default_plotly_layout

CONDITIONS = ['lab', 'physics_aug', 'gan', 'combined']
CONDITION_LABELS = {
    'lab': 'Lab only (n=40, LOO)',
    'physics_aug': 'Physics-augmented only (n=300, 5-fold)',
    'gan': 'GAN only (n=600, 5-fold)',
    'combined': 'Combined: real+physics-aug+GAN (n=940, 5-fold)',
}
ML_MODELS = ['ridge', 'elastic_net', 'decision_tree', 'random_forest', 'svr', 'xgboost']
ALL_MODELS = ML_MODELS + ['mlp', 'cnn']
MODEL_COLORS = {
    'ridge': '#17becf', 'elastic_net': '#bcbd22', 'decision_tree': '#8c564b',
    'random_forest': '#e34a1a', 'svr': '#2ca02c', 'xgboost': '#5a23c4',
    'mlp': '#0b5fa5', 'cnn': '#a83279',
}

# --- reload the persisted per-condition "best model" tables -----------------
best_tables = {}
for condition in CONDITIONS:
    csv_path = paths.condition_best_models_csv(condition)
    if not csv_path.exists():
        print(f"missing: {csv_path}")
        continue
    best_tables[condition] = pd.read_csv(csv_path)

best_long = pd.concat(
    [t.assign(condition=c) for c, t in best_tables.items() if not t.empty],
    ignore_index=True,
) if best_tables else pd.DataFrame()

# --- horizontal grouped bar chart -------------------------------------------
condition_labels_ordered = [CONDITION_LABELS[c] for c in CONDITIONS]

fig = go.Figure()
if not best_long.empty:
    for model in ALL_MODELS:
        sub = best_long[best_long['model'] == model].set_index('condition').reindex(CONDITIONS)
        if sub['mae'].isna().all():
            continue
        fig.add_trace(go.Bar(
            y=condition_labels_ordered, x=sub['mae'], name=model, orientation='h',
            marker_color=MODEL_COLORS.get(model, '#999999'),
            text=[f'{v:.2f}' if pd.notna(v) else '' for v in sub['mae']],
            textposition='outside',
        ))

fig = apply_default_plotly_layout(
    fig, title_text='Best MAE per model, across conditions (growing data pool ->)',
    xaxis_title='MAE (uM), lower = better', yaxis_title=None)
fig.update_layout(
    barmode='group',
    height=600, width=1000,
    # legend was overlapping the centered title as a wide horizontal strip
    # (8 models); moved to a vertical list on the right of the plot instead.
    legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02),
    margin=dict(l=220, r=160, t=70, b=50),  # l: room for the long condition labels
)
fig.update_yaxes(categoryarray=condition_labels_ordered, categoryorder='array', autorange='reversed')
fig.show()
