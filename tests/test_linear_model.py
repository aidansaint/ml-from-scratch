import numpy as np
from mlfs.linear_model import LinearRegression


def test_linear_regression() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([3.0, 5.0, 7.0])

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.coef_ is not None
    assert model.intercept_ is not None

    assert np.allclose(model.coef_, [2.0], atol=0.1)
    assert np.allclose(model.intercept_, 1.0, atol=0.1)
    assert np.allclose(y_pred, y, atol=0.1)
