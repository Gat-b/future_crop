import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from future_crop.data_viz.class_data_visualization import DataVisualization
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# --- Constants & Configuration ---

# Define the API URL (assuming the same as Page 0 if API is used)
API_URL = 'https://future-crop-464940631020.northamerica-northeast2.run.app'
# Model list (TO BE MODIFIED)
model_list = ['knn_val', 'xgb_val', 'rf_val', 'lgbm_val']
crop_list = ['wheat', 'maize']
number_location_list = np.arange(3,7,1)

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

ZONES_CONFIG = {
    'world': world,
    'europe': europe,
    'north_america': north_america,
    'south_america': south_america,
    'south_asia': south_asia,
    'sub_saharan_africa': sub_saharan_africa
}

zone_list = list(ZONES_CONFIG.keys())

# zone_list = ['world', 'europe', 'north_america', 'south_america', 'south_asia', 'sub_saharan_africa']

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
# 2. Application Header
# -------------------------------------------------------------
st.title("🌾 Long Term Future Crop Yield Prediction Model Estimates 🌾")
st.caption("Visual representation of long term future crop yield prediction errors between 2010 and 2020.")

st.divider()

# -------------------------------------------------------------
# 3. User Controls (Harmonized Layout)
# -------------------------------------------------------------
st.header("Prediction Visualization")

# Create a container for controls
control_container = st.container()

custom_width = 300

with control_container:
    col1, col2, col3, col4, col5= st.columns([1, 1, 1, 1, 1])

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

    number_location = col4.selectbox(
        '**Select a number of locations**',
        number_location_list,
        key='number_location_select_key', width=custom_width
    )

    # Launch Button (Vertical Alignment)
    col5.markdown("""
    <div style="height: 28px;"></div>
    """, unsafe_allow_html=True)

    launch_button = col5.button('Launch Prediction', icon="🚀", type="primary", width=custom_width)

# -------------------------------------------------------------
# 4. Processing and Rendering (API STRICTLY UNTOUCHED)
# -------------------------------------------------------------
if launch_button:
    with st.spinner('Loading data...'):
        try:
            # # --- API Logic (STRICTLY UNTOUCHED) ---
            # url = f'{API_URL}/yield?model={model_selection}'

            # response = requests.get(url, timeout=30)
            # response.raise_for_status()

            # # The original code's logic using eval() is preserved:
            # y_yield = pd.DataFrame(eval(response.json()))
            # # --- End API Logic ---

            # --- API Logic  ---
            url = f'{API_URL}/yield?model={model_selection}&crop={crop_selection}' # ajouter la crop
            y_pred = pd.read_csv(url, compression='gzip')

            ################# ATTENTE DE DISPO DES FICHIERS ####################
            url = f'{API_URL}/yield?model=train' # ajouter la crop
            train_df = pd.read_csv(url, compression='gzip')

            #  masque de zonage
            selected_zone_config = ZONES_CONFIG[zone_selection]

            if zone_selection == 'world':
                # Si 'world', on ne filtre pas (on garde tout)
                y_pred_zone = y_pred.copy()
                train_df_zone = train_df.copy()
            else:
                # Si une zone spécifique est choisie, on applique le filtre lat/lon
                lon_range = selected_zone_config['lonaxis_range']
                lat_range = selected_zone_config['lataxis_range']

                # Filtre pour y_pred
                y_pred_zone = y_pred[
                    (y_pred.lon_orig >= lon_range[0]) & (y_pred.lon_orig <= lon_range[1]) &
                    (y_pred.lat_orig >= lat_range[0]) & (y_pred.lat_orig <= lat_range[1])
                ]

                # Filtre pour train_df (Correction de la coquille lat_range[1])
                train_df_zone = train_df[
                    (train_df.lon_orig >= lon_range[0]) & (train_df.lon_orig <= lon_range[1]) &
                    (train_df.lat_orig >= lat_range[0]) & (train_df.lat_orig <= lat_range[1])
                ]


            # --- End API Logic ---

            # Rendering
            data_viz = DataVisualization()
            fig = data_viz.geo_plot(y_pred, zoom_area=eval(zone_selection)) # plot les différences entre y_pred et y_val



            # Harmonized plot layout
            fig.update_layout(
                title=f'Predicted Yield Map Errors({model_selection.upper()} - {crop_selection.upper()})',
                height=700, # Harmonized height
                margin=dict(t=50, b=0, l=0, r=0)
            )

            # Native Streamlit rendering
            st.subheader(f"Map of predicted yields ({model_selection.upper()})")
            st.plotly_chart(fig, use_container_width=True)

            ################# ATTENTE DE DISPO DES FICHIERS ####################


            fig_2 = data_viz.plotting_forecast(train_df_zone,y_pred_zone, n_loc=number_location)
            st.pyplot(fig_2, use_container_width=True)




        except Exception as e:
            st.error(f"An error occurred: {e}")
