import glassboxml.losses.mse as mse
import numpy as np

#Later might add r2 score, mae, mape, etc.

def rmse(y_true, y_pred):
    return np.sqrt(mse.mse_loss(y_true, y_pred))