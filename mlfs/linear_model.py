import numpy as np
from mlfs.metrics import mean_squared_error


class LinearRegression:
    def __init__(
        self,
        learning_rate: float = 0.01,  # Step size for gradient descent updates.
        max_iter: int = 1_000,  # Number of iterations for training.
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    # Train the model using gradient descent.
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        m, n = X.shape  # m = number of examples, n = number of features

        # Initialise weights and bias.
        w, b = np.zeros(n), 0.0

        for _ in range(self.max_iter):
            y_hat = X @ w + b  # Predictions

            # Compute gradients and update weights and bias.
            w -= (self.learning_rate / m) * X.T @ (y_hat - y)
            b -= (self.learning_rate / m) * np.sum(y_hat - y)

        self.coef_, self.intercept_ = w, b
        return self

    # Make predictions using the learned weights and bias.
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    # Return mean squared error on predictions.
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return mean_squared_error(y, self.predict(X))
