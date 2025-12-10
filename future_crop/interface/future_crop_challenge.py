import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from future_crop.data_viz.class_data_visualization import DataVisualization
from pathlib import Path

# --- Constants & Configuration ---

# Define the API URL
API_URL = 'https://future-crop-464940631020.northamerica-northeast2.run.app'

# Model list
model_list = ['knn', 'xgb', 'rf', 'lgbm'] ## à modifier
crop_list = ['wheat', 'maize']

world = None ## Indeed

europe = {
    'scope': 'europe',
    'center': {'lat': 50, 'lon': 15},
    'lataxis_range': [35, 70],
    'lonaxis_range': [-20, 40],
    'projection': 'mercator'
}

north_america = {
    'scope': 'north america',
    'center': {'lat': 40, 'lon': -100},
    'lataxis_range': [20, 60],
    'lonaxis_range': [-130, -70],
    'projection': 'albers usa' # Projection spécifique pour l'Amérique du Nord (USA)
}

south_america = {
    'scope': 'south america',
    'center': {'lat': -20, 'lon': -60},
    'lataxis_range': [-55, 10],
    'lonaxis_range': [-90, -30],
    'projection': 'mercator'
}

south_asia = {
    'scope': 'asia',
    'center': {'lat': 25, 'lon': 78},
    'lataxis_range': [0, 40],
    'lonaxis_range': [60, 100],
    'projection': 'natural earth'
}

sub_saharan_africa = {
    'scope': 'africa',
    'center': {'lat': 5, 'lon': 20},
    'lataxis_range': [-35, 25],
    'lonaxis_range': [-20, 55],
    'projection': 'natural earth'
}

zone_list = ['world', 'europe', 'north_america', 'south_america', 'south_asia', 'sub_saharan_africa']

# -------------------------------------------------------------
# 1. Page Configuration (MUST BE THE FIRST COMMAND)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Future Crop",
    layout="wide",
    page_icon="🌾"
)

# Configuration of project root (for consistency)
# project_root = Path(__file__).resolve().parents[3]

# -------------------------------------------------------------
# 2. Application Header & Introduction
# -------------------------------------------------------------
st.title("🌾 Future Crop Yield Analysis 🌾")
st.caption("Geographical visualization of predicted yields under a climate change scenario.")

# Introduction to the challenge
st.markdown('''
## The Future Crop Challenge
This challenge is dedicated to predicting **maize** and **wheat** yields using
soil and daily weather data under a high-emission climate change scenario.
''')

st.divider()

# -------------------------------------------------------------
# 3. User Controls (Harmonized Layout)
# -------------------------------------------------------------
st.header("Prediction Visualization")

# Create a container for controls
control_container = st.container()

custom_width = 300

with control_container:
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # Model Selector
    model_selection = col1.selectbox(
        '**Select the Prediction Model**',
        model_list,
        key='model_select_key', width=custom_width
    )

    # Crop Selector
    crop_selection = col2.selectbox(
        '**Select a Crop Type**',
        crop_list,
        key='crop_select_key', width=custom_width
    )

    # Zone Selector
    zone_selection = col3.selectbox(
        '**Select a Zone**',
        zone_list,
        key='zone_select_key', width=custom_width
    )

    # Launch Button (Vertical Alignment)
    col4.markdown("""
    <div style="height: 28px;"></div>
    """, unsafe_allow_html=True)

    launch_button = col4.button('Launch Prediction', icon="🚀", type="primary", width=custom_width)

# -------------------------------------------------------------
# 4. Processing and Rendering (API STRICTLY UNTOUCHED)
# -------------------------------------------------------------
if launch_button:
    with st.spinner('Loading data and generating map...'):
        try:
            # # --- API Logic  ---
            # url = f'{API_URL}/yield?model={model_selection}'

            # response = requests.get(url, timeout=30)
            # response.raise_for_status()

            # data = response.json()
            # # The original code's logic using eval() is preserved:
            # y_yield = pd.DataFrame(eval(data))
            # # --- End API Logic ---

            # --- API Logic  ---
            url = f'{API_URL}/yield?model={model_selection}&crop={crop_selection}' # ajouter la crop
            y_pred = pd.read_csv(url, compression='gzip')

            # --- End API Logic ---

            # Rendering
            data_viz = DataVisualization()
            fig = data_viz.geo_plot_non_diff(y_pred[y_pred.real_year <2099], zoom_area=eval(zone_selection)) # car 2099 n'a pas de points

            # Harmonized plot layout
            fig.update_layout(
                title=f'Predicted Yield Map ({model_selection.upper()} - {crop_selection.upper()})',
                height=700, # Harmonized height
                margin=dict(t=50, b=0, l=0, r=0)
            )

            # Native Streamlit rendering
            # st.subheader(f"Predicted Yield Map ({model_selection.upper()})")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during data processing or API call: {e}")

# -------------------------------------------------------------
# 5. Data and Hypotheses Section
# -------------------------------------------------------------
st.divider()
with st.expander("🔍 Detailed Data Description and Hypotheses"):
    st.markdown("""
    ## Available Data:

    ### Training data (1982-2020)
    For each crop (maize and wheat): Climate variables, yield solutions, and soil data (texture, real year, nitrogen, CO2).

    ### Test data (2021-2098)
    Same input data structure as training, but without yield solutions.

    ### Columns and Units
    * All files contain an **ID** column, `lat`, `lon`, `year`, and `crop`.
    * **rsds** (short-wave radiation): $[W\\ m^{-2}]$
    * **pr** (precipitation): $[kg\\ m^{-2}\\ s^{-1}]$
    * **tas, tmax, tmin** (temperatures): $[^{\\circ}C]$
    * **yield** (target): $[t\\ ha^{-1}]$

    ## Hypotheses

    * **Nitrogen fertilization rates and soil texture** remain **constant** across longitude and latitude over the years.
    * **CO2 concentration** is constant per year but **increases** over the years.
    """)
