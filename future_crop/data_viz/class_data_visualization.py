import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import random
from pathlib import Path
from sklearn.model_selection import learning_curve


class DataVisualization:
    """
    Utility class for visualizing data
    """
    def __init__(self, geo_coding: Path = None, yield_forecasts: Path = None):
        current_file_path = Path(__file__).resolve()
        project_root = current_file_path.parents[2] # Remonte de 2 crans pour atteindre la racine du projet

        # Si aucun chemin n'est fourni, on construit le chemin absolu par défaut
        if geo_coding is None:
            self.geo_coding = project_root / "geo_coding"
        else:
            self.geo_coding = Path(geo_coding)

        if yield_forecasts is None:
            self.yield_forecasts = project_root / "yield_forecasts"
        else:
            self.yield_forecasts = Path(yield_forecasts)

        # Création des dossiers si inexistants

        # self.geo_coding.mkdir(parents=True, exist_ok=True)
        # self.yield_forecasts.mkdir(parents=True, exist_ok=True)
        # self.geo_coding = Path(geo_coding)
        # self.geo_coding.mkdir(parents=True, exist_ok=True)
        # self.yield_forecasts = Path(yield_forecasts)
        # self.yield_forecasts.mkdir(parents=True, exist_ok=True)

    def plot_forecast(self, y_pred: pd.Series, y_train: pd.Series,
                      y_test: pd.Series, upper: np.ndarray = None,
                      lower: np.ndarray = None):
        """
        Plot training, test and forecast time series with an optional confidence interval.

        Parameters
        ----------
        y_pred : array-like or pandas.Series
            Predicted values for the test period. If not a pandas.Series, it will be
            converted and aligned to y_test.index.
        y_train : pandas.Series
            Training time series (expected to have a datetime or comparable index).
        y_test : pandas.Series
            Test time series. Its index is used for plotting and aligning y_pred and bounds.
        upper : array-like or None, optional
            Upper bound of the confidence interval (same length as y_test). The shaded
            interval is drawn only if both upper and lower are numpy.ndarray.
        lower : array-like or None, optional
            Lower bound of the confidence interval.

        Returns
        -------
        None
            The function displays a matplotlib figure and does not return a value.

        Notes
        -----
        - upper and lower are considered valid confidence bounds only when both are
        instances of numpy.ndarray (checked via isinstance).
        - If lengths or indices do not match between y_pred and y_test, pandas will raise
        an error when creating the aligned Series.
        """
        is_conf = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)

        ##### ADDED BY MLS #####

        y_test.index = range(len(y_train), len(y_train) + len(y_test))

        ########################

        y_pred_series = pd.Series(y_pred, index=y_test.index)
        y_train_series = y_train.copy()
        y_test_series = y_test.copy()

        plt.figure(figsize=(10, 4))
        plt.plot(y_train_series, label="Train")
        plt.plot(y_test_series, label="Test", color="black")
        plt.plot(y_pred_series, label="Prediction", color="orange")

        if is_conf:
            plt.fill_between(
                y_test.index,
                lower,
                upper,
                color="tab:orange",
                alpha=0.2,
                label="confidence interval"
            )

        plt.title("Predictions vs Actuals")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def create_results_df(self, X_val: pd.DataFrame = None, y_val: pd.DataFrame = None, y_pred: pd.Series = None, model: str = None, crop: str = 'wheat') -> pd.DataFrame:
        yield_df = X_val[["ID", "real_year", "lon_orig", "lat_orig"]].copy()
        
        if y_val is not None:
            yield_df = pd.merge(yield_df, y_val, how='inner', on='ID')

        if y_pred is not None:
            yield_df["pred"] = y_pred

        if y_val is not None and y_pred is not None:
            yield_df["yield_diff"] = yield_df["pred"] - yield_df["yield"]
        
        yield_df["lon_lat"] = yield_df["lon_orig"].astype(str) + "_" + yield_df["lat_orig"].astype(str)

        yield_df.to_csv(self.yield_forecasts / f"{crop}_{model}_yield_pred.csv", index=False)
        return yield_df

    def geo_plot(self, yield_df: pd.DataFrame = None, X_val: pd.DataFrame = None, y_val: pd.DataFrame = None,
                 y_pred: pd.Series = None, zoom_area : dict = None):
        """
        Plots the yield difference.
        Can be called with a pre-computed yield_df OR with raw data (X, y, pred) to compute it on the fly.
        """

        # 1. Automatic Execution: Create DataFrame if not provided
        if yield_df is None:
            if all(v is not None for v in [X_val, y_val, y_pred]):
                print("Computing results DataFrame automatically...")
                yield_df = self.create_results_df(X_val, y_val, y_pred)
            else:
                raise ValueError("You must provide either 'yield_df' OR 'X_val', 'y_val', and 'y_pred'.")

        # 2. Plotting Logic (unchanged)
        fig = px.scatter_geo(
            data_frame=yield_df,
            lat="lat_orig",
            lon="lon_orig",
            color="yield_diff",
            range_color=[-5, 5],
            size_max=0.001,
            hover_name="yield_diff",
            animation_frame="real_year",
            projection="natural earth",
            color_continuous_scale="Turbo_r"
        )

        # 3. Application du Zoom (Nouveauté)
        if zoom_area is not None:
            # Utilise update_geos pour définir la zone de la carte
            fig.update_geos(
                # Définir le scope (monde, europe, usa, asia, etc.)
                scope=zoom_area.get('scope', 'world'),
                # Définir le centre de la carte
                center=zoom_area.get('center', None),
                # Définir le niveau de zoom (par défaut 1, >1 zoome plus)
                lataxis_range=zoom_area.get('lataxis_range', None),
                lonaxis_range=zoom_area.get('lonaxis_range', None),
                # Zoom pour ajuster les données si 'fitbounds' est fourni
                fitbounds=zoom_area.get('fitbounds', False)
        )
        # Si aucun zoom n'est spécifié, on conserve une vue globale par défaut
        else:
            fig.update_geos(scope='world')

            fig.update_layout(title="diff vs. valuation yield")
            # fig.show()

        path = self.geo_coding / f"geo_plot.html"
        fig.write_html(path)
        return fig

    def geo_plot_non_diff(self, yield_df: pd.DataFrame = None, X_val: pd.DataFrame = None, y_val: pd.DataFrame = None,
                 y_pred: pd.Series = None, zoom_area : str = None):
        """
        Plots the yield difference.
        Can be called with a pre-computed yield_df OR with raw data (X, y, pred) to compute it on the fly.
        """

        # 1. Automatic Execution: Create DataFrame if not provided
        if yield_df is None:
            if all(v is not None for v in [X_val, y_val, y_pred]):
                print("Computing results DataFrame automatically...")
                yield_df = self.create_results_df(X_val, y_val, y_pred)
            else:
                raise ValueError("You must provide either 'yield_df' OR 'X_val', 'y_val', and 'y_pred'.")

        # 2. Plotting Logic (unchanged)
        fig = px.scatter_geo(
            data_frame=yield_df,
            lat="lat_orig",
            lon="lon_orig",
            color='pred',
            range_color=[0, 10],
            size_max=0.001,
            hover_name='pred',  # modification par MLS
            animation_frame="real_year",
            projection="natural earth",
            color_continuous_scale="Turbo_r"
        )

        # 3. Application du Zoom (Nouveauté)
        if zoom_area is not None:
            # Utilise update_geos pour définir la zone de la carte
            fig.update_geos(
                # Définir le scope (monde, europe, usa, asia, etc.)
                scope=zoom_area.get('scope', 'world'),
                # Définir le centre de la carte
                center=zoom_area.get('center', None),
                # Définir le niveau de zoom (par défaut 1, >1 zoome plus)
                lataxis_range=zoom_area.get('lataxis_range', None),
                lonaxis_range=zoom_area.get('lonaxis_range', None),
                # Zoom pour ajuster les données si 'fitbounds' est fourni
                fitbounds=zoom_area.get('fitbounds', False)
        )
        # Si aucun zoom n'est spécifié, on conserve une vue globale par défaut
        else:
            fig.update_geos(scope='world')

            fig.update_layout(title="expected yield")
            # fig.show()

        path = self.geo_coding / f"geo_plot.html"
        fig.write_html(path)
        return fig

    def error_distribution(self, yield_df: pd.DataFrame = None, X_val: pd.DataFrame = None, y_val: pd.DataFrame = None,
                 y_pred: pd.Series = None):

        # 1. Automatic Execution: Create DataFrame if not provided
        if yield_df is None:
            if all(v is not None for v in [X_val, y_val, y_pred]):
                print("Computing results DataFrame automatically...")
                yield_df = self.create_results_df(X_val, y_val, y_pred)
            else:
                raise ValueError("You must provide either 'yield_df' OR 'X_val', 'y_val', and 'y_pred'.")

        #2. Plotting
        sns.histplot(yield_df["yield_diff"], bins=100)

    def plotting_forecast(self, train_df: pd.DataFrame = None, yield_df: pd.DataFrame = None, n_loc: int = 5,
                          X_train: pd.DataFrame = None, y_train: pd.DataFrame = None,
                          X_val: pd.DataFrame = None, y_val: pd.DataFrame = None, y_pred: pd.Series = None):

        # 1. Automatic Execution: Create DataFrame if not provided
        if train_df is None:
            if all(v is not None for v in [X_train, y_train]):
                print("Computing results DataFrame automatically...")
                train_df = self.create_results_df(X_val = X_train, y_val = y_train)
            else:
                raise ValueError("You must provide either 'train_df' OR 'X_train', 'y_train'.")

        if yield_df is None:
            if all(v is not None for v in [X_val, y_val, y_pred]):
                print("Computing results DataFrame automatically...")
                yield_df = self.create_results_df(X_val = X_val, y_val = y_val, y_pred = y_pred)
            else:
                raise ValueError("You must provide either 'yield_df' OR 'X_val', 'y_val', and 'y_pred'.")

        #2. Plotting

        unique_locs = list(set(train_df["lon_lat"].unique()) | set(yield_df["lon_lat"].unique()))
        selected_locs = random.sample(unique_locs, min(n_loc, len(unique_locs)))

        fig, ax = plt.subplots(figsize=(15, 6))
        palette = sns.color_palette("husl", len(selected_locs))

        for i, loc in enumerate(selected_locs):

            color = palette[i]
            last_train = train_df[train_df["lon_lat"] == loc].iloc[[-1]].copy()
            last_train[0] = last_train['yield']
            val_data = pd.concat([last_train, yield_df[yield_df["lon_lat"] == loc]], axis=0)

            sns.lineplot(train_df[train_df["lon_lat"]==loc],x="real_year",y='yield', ax=ax, color=color, alpha=0.4, legend=False)
            sns.lineplot(yield_df[yield_df["lon_lat"]==loc], x= "real_year",y="yield", ax=ax, color=color, label=str(loc), linewidth=2)
            if 0 in val_data.columns:
                sns.lineplot(yield_df[yield_df["lon_lat"]==loc], x= "real_year",y='pred', ax=ax, color=color, linestyle='--', linewidth=2, legend=False)

        ax.set_title(f"Yield History & Forecast ({n_loc} Random Locations)")
        ax.set_ylabel("Yield")
        ax.legend(title="Location", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # plt.show()
        return fig

    def plot_feature_importance(self, model, feature_names: list[str], top_n: int =20):

        # 1. Extraction et tri des importances
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False).head(top_n)

        # 2. Création du graphique
        plt.figure(figsize=(10, max(6, top_n * 0.3))) # Hauteur adaptative selon le nb de features

        # Barplot horizontal (x=importance, y=feature)
        sns.barplot(data=feat_imp, x='importance', y='feature', palette="viridis", hue='feature', legend=False)

        # 3. Esthétique
        plt.title(f"Top {top_n} Feature Importance", fontsize=15)
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self, model, X, y):
        '''
        uses the sklearn function 'learning_curve' to plot
        the learning curves of a model while training.
        Does not return values, just plots.
        '''
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 100),
            scoring="r2"
        )

        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)

        plt.figure(figsize=(6, 4))
        plt.plot(train_sizes, train_mean, marker="o", label="Train")
        plt.plot(train_sizes, val_mean, marker="s", label="Validation")
        plt.xlabel("Taille de l'échantillon")
        plt.ylabel("Score (R²)")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
