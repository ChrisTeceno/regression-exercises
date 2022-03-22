import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")
from matplotlib import style

style.use("ggplot")


def plot_residuals(y, yhat):
    """plot residuals given y and yhat"""
    residuals = y - yhat
    plt.hlines(0, y.min(), y.max(), ls="--")
    plt.scatter(y, residuals, color="blue")
    plt.ylabel("residual ($y - \hat{y}$)")
    plt.xlabel("y value ($y$)")
    plt.title("Actual vs Residual")
    plt.show()


def regression_errors(y, yhat):
    """return metrics"""
    residuals = y - yhat
    return pd.Series(
        {
            "SSE": (residuals ** 2).sum(),
            "ESS": ((yhat - y.mean()) ** 2).sum(),
            "TSS": ((y - yhat.mean()) ** 2).sum(),
            "MSE": mean_squared_error(y, yhat),
            "RMSE": mean_squared_error(y, yhat) ** 0.5,
        }
    )


def baseline_mean_errors(y):
    """return baseline metrics"""
    # make a series of the baseline value
    mean = pd.Series([y.mean()])
    # repeat the value to make a correctly sized series to match y
    mean = mean.repeat(len(y))
    residuals = y - mean
    return pd.Series(
        {
            "SSE": (residuals ** 2).sum(),
            "MSE": mean_squared_error(y, mean),
            "RMSE": mean_squared_error(y, mean) ** 0.5,
        }
    )


def better_than_baseline(y, yhat):
    """compare model results to baseline based on mean"""
    # make a series of the baseline value
    mean = pd.Series([y.mean()])
    # repeat the value to make a correctly sized series to match y
    mean = mean.repeat(len(y))
    rmse_baseline = (mean_squared_error(y, mean) ** 0.5,)
    rmse_model = (mean_squared_error(y, yhat) ** 0.5,)
    is_better = rmse_model < rmse_baseline
    # print result
    print(f"based on RMSE, is the model better: {is_better}")
    # return a boolean to be used in a df
    return is_better
