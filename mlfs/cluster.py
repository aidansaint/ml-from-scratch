import numpy as np


class KMeans:
    def __init__(self, n_clusters: int = 8, max_iter: int = 300):
        self.n_clusters = n_clusters  # Number of clusters (k).
        self.max_iter = max_iter  # Maximum iterations of algorithm.
        self.centroids: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "KMeans":
        # Initalise centroids randomly from data points.
        centroid_indices = np.random.choice(
            X.shape[0], size=self.n_clusters, replace=False
        )
        self.centroids = X[centroid_indices]

        for _ in range(self.max_iter):
            # Calculate squared distances.
            assert self.centroids is not None  # For type checker
            distances = self._distance(X, self.centroids)

            # Assign each point to its nearest cluster.
            labels = np.argmin(distances, axis=1)

            # Update centroids as mean of assigned points.
            one_hot = np.eye(self.n_clusters)[labels]
            label_counts = one_hot.sum(axis=0).reshape(-1, 1)
            self.centroids = (one_hot.T @ X) / np.maximum(label_counts, 1.0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Calculate squared distances.
        assert self.centroids is not None  # For type checker
        distances = self._distance(X, self.centroids)

        # Assign each point to its nearest cluster.
        labels = np.argmin(distances, axis=1)

        return labels

    def score(self, X: np.ndarray) -> float:
        # Calculate squared distances.
        assert self.centroids is not None  # For type checker
        distances = self._distance(X, self.centroids)

        # Compute sum of squared distances from each point to closest centroid
        inertia = np.sum(np.min(distances, axis=1))

        return inertia

    def _distance(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Compute squared L2 norms.
        A_norms = np.sum(A**2, axis=1).reshape(-1, 1)
        B_norms = np.sum(B**2, axis=1).reshape(1, -1)

        # Calculate squared distances.
        D_squared = A_norms + B_norms - 2 * (A @ B.T)

        return D_squared
