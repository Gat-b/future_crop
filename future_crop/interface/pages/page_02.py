import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from future_crop.data_viz.class_data_visualization import DataVisualization
from future_crop.ml_logic.baseline import dummy_baseline
import matplotlib.pyplot as plt
import plotly.express as px

st.markdown('''
##### Visualisation of the yield differences per year between 2010 and 2020
This page represents the diff ...
            ''')

project_root = Path(__file__).resolve().parents[3] # sub folder

data_viz = DataVisualization()

############ CSV UPLOADER ##############

uploaded_file = st.file_uploader(label = 'upload your csv file here',
                                 type = 'csv')

########################################

model_list = ['model_1', 'model_2', 'model_3']

model_selection = st.selectbox('Select a model' ,
                               model_list,
                               width = 160)

st.markdown(f'''
            you selected {model_selection}
            ''')

# X_new = pd.read_csv(uploaded_file)

####
### appel à l'API avec X_new
####

# y_pred = np. array(len(10))

# à modifier quand plugged avec l'API
y_yield = pd.read_csv(project_root / "dummy_data/y_yield.csv")
# y_yield = data_viz.create_results_df(X_val, y_val, y_pred)


fig = data_viz.geo_plot_non_diff(y_yield)

st.markdown('''
#### Please find hereunder the prediction of you crop
            ''')

### A noter qu'il faut modifier / ajouter une fonction
# de plot pour le yield pur et non le diff de yield

########## BLOC DE RENDU DANS STREAMLIT #####################
# Forcez l'affichage de toutes les traces de données
for trace in fig.data:
    trace.visible = True
    if hasattr(trace, 'marker') and trace.marker:
        trace.marker.size = 8 # Force une taille visible

fig_html = fig.to_html(
    include_plotlyjs='cdn',
    full_html=True
)

st.html(fig_html, unsafe_allow_javascript=True)

############################ FIN ############################
