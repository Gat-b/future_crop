import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from future_crop.data_viz.class_data_visualization import DataVisualization
from future_crop.ml_logic.baseline import dummy_baseline
import matplotlib.pyplot as plt
import plotly.express as px


project_root = Path(__file__).resolve().parents[2]
st.markdown('''
## Welcome to Future Crop

This projet is aming at ....
....
....
            ''')

st.image(image= project_root / 'image.png')
st.image(image= project_root / 'Disco Cow GIF by CLAAS.gif')
