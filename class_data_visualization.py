import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataVisualization:
    """
    Utility class for visualizing data
    """
    def __init__(self):
        pass


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

        y_test.index = y_test.index +len(y_train)

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
