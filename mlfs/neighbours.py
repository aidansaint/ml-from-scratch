import numpy as np
from collections import Counter
from mlfs.metrics import accuracy


class KNeighboursClassifier:
    def __init__(self, n_neighbours=5) -> None:
        self.n_neighbours = n_neighbours  # Number of neighbours to use for vote.

    # Store training data in memory.
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighboursClassifier":
        self.X_train = X
        self.y_train = y

        return self

    # Make predictions using squared euclidean distances.
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Compute squared L2 norms.
        X_train_norms = np.sum(self.X_train**2, axis=1).reshape(1, -1)
        X_norms = np.sum(X**2, axis=1).reshape(-1, 1)

        # Calculate squared distances.
        D_squared = X_train_norms + X_norms - 2 * (X @ self.X_train.T)

        # Sort squared distances and get top k neighbour labels.
        neighbour_indices = np.argpartition(D_squared, self.n_neighbours - 1, axis=1)[
            :, : self.n_neighbours
        ]
        labels = self.y_train[neighbour_indices]

        # Perform majority vote to determine predictions.
        y_preds = []
        for row in labels:
            counter = Counter(row)
            max_freq = max(counter.values())

            for val, freq in counter.items():
                if freq == max_freq:
                    y_preds.append(val)
                    break

        return np.array(y_preds)

    # Return accuracy of predictions.
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy(y, self.predict(X))
