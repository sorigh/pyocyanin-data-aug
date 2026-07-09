"""Entry point / router for the Pyocyanin Synthetic Data Studio.


Declares the two pages and hands control to Streamlit's navigation. Shared
state (`st.set_page_config`) is set here once because Streamlit forbids
calling it more than once per run.

- Data generation, physics-informed controls and the GAN generation menu
  live in `pages/1_Data_Generation.py`.
- Model training, testing and transparency live in
  `pages/2_Model_Training_Testing.py`.
- Shared, cross-page state (datasets, trained model bundles) is read/written
  through `data_registry.py`.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title='Pyocyanin Synthetic Data Studio', layout='wide')

data_generation_page = st.Page(
    'pages/1_Data_Generation.py', title='Data Generation', default=True)
model_training_testing_page = st.Page(
    'pages/2_Model_Training_Testing.py', title='Model Training & Testing')

navigation = st.navigation([data_generation_page, model_training_testing_page])
navigation.run()
