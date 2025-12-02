import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def compute_baseline(y_train : pd.Series , length : int) -> pd.Series:
    '''
    function tales a pd.Series and a length to compute
    the baseline of our future_crop's prediction with
    an ARIMA (2,1,1) model

    it should be used as follows :
    compute_baseline(y_train, len(y_test))
    '''


    arima = ARIMA(y_train, order=(2,1,1), trend='t')
    arima = arima.fit()

    forecast = arima.forecast(length, alpha = 0.05)

    return forecast
