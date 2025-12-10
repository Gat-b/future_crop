import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from future_crop.data_viz.class_data_visualization import DataVisualization

# --- Constants & Configuration ---

# Model list (Harmonized)
model_list = ['model_1', 'model_2', 'model_3']
# Assuming project_root is correctly set for local access
project_root = Path(__file__).resolve().parents[3]

# -------------------------------------------------------------
# 1. Page Configuration (MUST BE THE FIRST COMMAND)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Future Crop",
    layout="wide",
    page_icon="🌾"
)

# -------------------------------------------------------------
# 2. Application Header & Introduction
# -------------------------------------------------------------
st.title("📈 Yield Difference Visualization 📈")
st.caption("Visualization of the yield differences per year (e.g., between 2010 and 2020).")

st.divider()

# -------------------------------------------------------------
# 3. User Controls (Harmonized Layout)
# -------------------------------------------------------------
st.header("⚙️ Data Input and Visualization Controls")

control_container = st.container()
custom_width = 300

with control_container:
    col1, col2= st.columns([1, 1])

    # File Uploader (in col1)
    uploaded_file = col1.file_uploader(
        label='**1. Upload your CSV file here**',
        type='csv',
        key='csv_uploader'
    )

    # Model Selector (in col2)
    model_selection = col2.selectbox(
        '**2. Select a Model (for reference)**',
        model_list,
        key='model_select_key', width=custom_width
    )

    # Placeholder for the button / Action message
    col2.markdown("""
    <div style="height: 28px;"></div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. Processing and Rendering (Adapted for local data/upload)
# -------------------------------------------------------------
st.subheader(f"Map of Yield Difference ({model_selection})")

data_viz = DataVisualization()

if uploaded_file is not None:
    try:
        # Read uploaded data
        y_yield = pd.read_csv(uploaded_file)
        st.success("File successfully loaded. Displaying results...")

    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        y_yield = None
else:
    # --- Fallback/Dummy Data ---
    st.info("Please upload a CSV file to view your results. Using dummy data for demonstration.")
    try:
        # Read dummy data for visualization placeholder
        # NOTE: Adjust path based on your actual project structure!
        y_yield = pd.read_csv(project_root / "dummy_data/y_yield.csv")
    except FileNotFoundError:
        st.error(f"Dummy data not found at {project_root / 'dummy_data/y_yield.csv'}")
        y_yield = None
    # --- End Fallback ---


if y_yield is not None:
    with st.spinner('Generating difference plot...'):

        # Use the appropriate plot function (assuming geo_plot_non_diff is for difference)
        fig = data_viz.geo_plot_non_diff(y_yield)


        # Harmonized plot layout
        fig.update_layout(
            title=f'Yield Difference Visualization ({model_selection.upper()})',
            height=700,
            margin=dict(t=50, b=0, l=0, r=0)
        )

        # Native Streamlit rendering (harmonized)
        st.plotly_chart(fig, use_container_width=True)
