import numpy as np
from mlfs.metrics import mean_squared_error


def test_mean_squared_error() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([4.0, 5.0, 6.0])

    assert mean_squared_error(y_true, y_pred) == 9.0
