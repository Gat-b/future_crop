# # import streamlit as st
# # import numpy as np
# # import pandas as pd
# # from pathlib import Path
# # from future_crop.data_viz.class_data_visualization import DataVisualization
# # from future_crop.ml_logic.baseline import dummy_baseline
# # import matplotlib.pyplot as plt
# # import plotly.express as px

# # import requests





# # st.markdown('''
# # ##### Visualisation of the yield differences per year between 2010 and 2020
# #             ''')

# # project_root = Path(__file__).resolve().parents[3] # sub folder

# # model_list = ['knn', 'xgb_regressor']



# # ######## LAYOUT ############
# # custom_width = 160
# # columns = st.columns(2)

# # ############################

# # model_selection = columns[0].selectbox('' ,
# #                                model_list,
# #                                width=custom_width)

# # ########### API CALL ###########

# # columns[1].markdown('bla')
# # if columns[1].button('Call API', icon= "🔥", width=custom_width):

# #     url = f'https://future-crop-464940631020.northamerica-northeast2.run.app/yield?model={model_selection}'
# #     branch = 'yield'

# #     response = requests.get(url)

# #     y_yield = pd.DataFrame(eval(response.json()))

# # ################################

# # ########## BLOC DE RENDU DANS STREAMLIT #####################
# #     data_viz = DataVisualization()
# #     fig = data_viz.geo_plot(y_yield)

# #     # Forcez l'affichage de toutes les traces de données
# #     for trace in fig.data:
# #         trace.visible = True
# #         if hasattr(trace, 'marker') and trace.marker:
# #             trace.marker.size = 8 # Force une taille visible

# #     fig_html = fig.to_html(
# #         include_plotlyjs='cdn',
# #         full_html=True
# #     )

# #     st.html(fig_html, unsafe_allow_javascript=True)

# # ############################ FIN ############################

# import streamlit as st
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from future_crop.data_viz.class_data_visualization import DataVisualization
# from future_crop.ml_logic.baseline import dummy_baseline
# import matplotlib.pyplot as plt
# import plotly.express as px

# import requests

# st.markdown('''
# ##### Visualisation of the yield differences per year entre 2010 et 2020
#             ''')

# project_root = Path(__file__).resolve().parents[3] # sub folder

# model_list = ['knn', 'xgb_regressor']

# ## LAYOUT ##
# custom_width = 160
# columns = st.columns(2)

# # --- Colonne 1 : Selecteur ---
# model_selection = columns[0].selectbox('' ,
#                                model_list,
#                                width=custom_width)

# # --- Colonne 2 : Bouton avec ajustement d'alignement ---

# # 1. Ajoutez l'espace pour compenser la hauteur du sélecteur sans étiquette.
# # L'étiquette (même vide) du selectbox ajoute une certaine hauteur.
# # Un <br> (saut de ligne) ou &nbsp; (espace) peut aider.

# columns[1].markdown(
#     """
#     <div style="height: 28px;"></div>
#     """,
#     unsafe_allow_html=True)


# if columns[1].button('Call API', icon= "🔥", width=custom_width):

#     url = f'https://future-crop-464940631020.northamerica-northeast2.run.app/yield?model={model_selection}'
#     branch = 'yield'

#     response = requests.get(url)

#     y_yield = pd.DataFrame(eval(response.json()))

# ## BLOC DE RENDU DANS STREAMLIT ##
#     data_viz = DataVisualization()
#     fig = data_viz.geo_plot(y_yield)

#     # Forcez l'affichage de toutes les traces de données
#     for trace in fig.data:
#         trace.visible = True
#         if hasattr(trace, 'marker') and trace.marker:
#             trace.marker.size = 8 # Force une taille visible

#     fig_html = fig.to_html(
#         include_plotlyjs='cdn',
#         full_html=True
#     )

#     st.html(fig_html, unsafe_allow_javascript=True)

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from future_crop.data_viz.class_data_visualization import DataVisualization
from pathlib import Path
# Supprimé les imports inutilisés pour la clarté (numpy, matplotlib, dummy_baseline)

# -------------------------------------------------------------
# 1. Configuration de la page (DOIT ÊTRE LA PREMIÈRE COMMANDE)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Future Crop Yield Analysis",
    layout="wide", # Utilise toute la largeur de l'écran
    page_icon="🌾"
)

project_root = Path(__file__).resolve().parents[3] # sub folder
model_list = ['knn', 'xgb_regressor']

# -------------------------------------------------------------
# 2. En-tête de l'application
# -------------------------------------------------------------
st.title("🌾 Analyse des Rendements Agricoles Futurs 🌾")
st.caption("Visualisation cartographique des rendements prédits (2010 - 2020) selon différents modèles de Machine Learning.")

st.divider() # Séparateur visuel

# -------------------------------------------------------------
# 3. Contrôles Utilisateur (Disposition Améliorée)
# -------------------------------------------------------------
# Crée un conteneur pour les contrôles
control_container = st.container()

custom_width = 300

with control_container:
    col1, col2 = st.columns([1, 1]) # Utilisation de la flexibilité des colonnes

    # **Modification Clé : Utilisation d'une étiquette (label) explicite**
    model_selection = col1.selectbox(
        '**Sélectionnez le Modèle de Prédiction**', # Étiquette claire
        model_list,
        key='model_select_key', width = custom_width
    )

    # Bouton d'appel
    # Pas besoin de hacks d'alignement, l'étiquette du selectbox fait le travail.

    col2.markdown(
    """
    <div style="height: 28px;"></div>
    """,
    unsafe_allow_html=True)

    if col2.button('Lancer la Prédiction', icon="🚀", type="primary", width=custom_width):

        # -------------------------------------------------------------
        # 4. Traitement et Rendu (Avec Retour Visuel)
        # -------------------------------------------------------------
        with st.spinner('Chargement des données et génération de la carte...'):
            try:
                url = f'https://future-crop-464940631020.northamerica-northeast2.run.app/yield?model={model_selection}'

                response = requests.get(url, timeout=30)
                response.raise_for_status() # Lève une exception si le statut est une erreur (4xx ou 5xx)

                # Assurez-vous que eval() est sécurisé si les données viennent d'une source externe
                # Il est fortement recommandé d'utiliser json.loads() si possible, ou d'encapsuler eval.
                y_yield = pd.DataFrame(eval(response.json()))

                # Rendu de la visualisation
                data_viz = DataVisualization()
                fig = data_viz.geo_plot(y_yield)

                fig.update_layout(
                height=600, # Définir la hauteur en pixels (ex: 800px)
                margin=dict(t=0, b=0, l=0, r=0) # Optionnel: Réduire les marges
)

                # Utilisation de st.plotly_chart pour un rendu natif
                st.subheader(f"Carte de Rendement Prédit ({model_selection.upper()})")
                st.plotly_chart(fig, use_container_width=True)

            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de l'appel à l'API : {e}")
            except Exception as e:
                st.error(f"Une erreur s'est produite lors du traitement des données : {e}")

# -------------------------------------------------------------
# 5. Conclusion (Si le bouton n'a pas été cliqué)
# -------------------------------------------------------------
if 'y_yield' not in locals():
    st.info("Sélectionnez un modèle de prédiction et cliquez sur 'Lancer la Prédiction' pour afficher la carte interactive.")
