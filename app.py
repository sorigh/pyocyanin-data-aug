"""Entry point / router for the Pyocyanin Synthetic Data Studio.

This file intentionally contains no data-generation or model logic - it only
declares the two pages and hands control to Streamlit's navigation. Shared
state (`st.set_page_config`) is set here once because Streamlit forbids
calling it more than once per run; the individual page files must not call
it again.

- Data generation, physics-informed controls and the GAN generation menu
  live in `pages/1_Data_Generation.py`.
- Model training, testing and transparency live in
  `pages/2_Model_Training_Testing.py`.
- Shared, cross-page state (datasets, trained model bundles) is read/written
  through `data_registry.py`, not through direct imports between pages.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title='Pyocyanin Synthetic Data Studio', layout='wide', page_icon='🧫')

data_generation_page = st.Page(
    'pages/1_Data_Generation.py', title='Data Generation', icon='🧬', default=True)
model_training_testing_page = st.Page(
    'pages/2_Model_Training_Testing.py', title='Model Training & Testing', icon='🧪')

navigation = st.navigation([data_generation_page, model_training_testing_page])
navigation.run()
