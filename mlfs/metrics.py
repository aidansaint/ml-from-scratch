import numpy as np


# Compute the mean squared error between true and predicted values.
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# Calculate the accuracy of predicted labels.
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred == y_true))
