import time
import numpy as np
from sklearn.linear_model import SGDClassifier


class Agent:
    def __init__(
        self,
        input_dim: int = 784,
        output_dim: int = 10,
        seed: int | None = None,
        learning_rate: float = 0.02,
        epochs: int = 30,
        batch_size: int = 4096,
        n_components: int = 512,
        gamma: float = 0.03,
        alpha: float = 1e-5,
        time_limit_seconds: float | None = 60.0,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_components = n_components
        self.gamma = gamma
        self.alpha = alpha
        self.time_limit_seconds = time_limit_seconds
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # Random Fourier feature parameters for approximate RBF kernel
        # W ~ N(0, 2*gamma)
        self.W = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 * self.gamma),
            size=(self.input_dim, self.n_components),
        ).astype(np.float32)

        self.b = self.rng.uniform(
            low=0.0,
            high=2.0 * np.pi,
            size=(self.n_components,),
        ).astype(np.float32)

        self.scale = np.sqrt(2.0 / self.n_components).astype(np.float32)

        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=self.alpha,
            learning_rate="constant",
            eta0=self.learning_rate,
            average=True,
            fit_intercept=True,
            random_state=seed,
        )

        self._is_fitted = False

    def _prepare_inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = X.reshape(len(X), -1) / 255.0
        return X

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y).ravel().astype(np.int64)

    def _rff(self, X: np.ndarray) -> np.ndarray:
        # Fast random Fourier features for approximate RBF kernel
        Z = X @ self.W
        Z += self.b
        np.cos(Z, out=Z)
        Z *= self.scale
        return Z

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X = self._prepare_inputs(X_train)
        y = self._prepare_targets(y_train)

        n_samples = X.shape[0]
        classes = np.arange(self.output_dim, dtype=np.int64)

        start_time = time.perf_counter()
        first_batch = True

        for _ in range(self.epochs):
            if self.time_limit_seconds is not None:
                if time.perf_counter() - start_time >= self.time_limit_seconds:
                    break

            perm = self.rng.permutation(n_samples)

            for i in range(0, n_samples, self.batch_size):
                if self.time_limit_seconds is not None:
                    if time.perf_counter() - start_time >= self.time_limit_seconds:
                        self._is_fitted = True
                        return

                idx = perm[i:i + self.batch_size]
                X_batch = X[idx]
                y_batch = y[idx]

                Z_batch = self._rff(X_batch)

                if first_batch:
                    self.model.partial_fit(Z_batch, y_batch, classes=classes)
                    first_batch = False
                else:
                    self.model.partial_fit(Z_batch, y_batch)

        self._is_fitted = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Agent must be trained before calling predict().")

        X = self._prepare_inputs(X_test)
        Z = self._rff(X)
        return self.model.predict(Z)