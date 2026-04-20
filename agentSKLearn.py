import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sympy import xthreaded


class Agent:
    def __init__(
        self,
        input_dim: int = 784,
        output_dim: int = 10,
        seed: int | None = None,
        learning_rate: float = 0.15,
        epochs: int = 1500,
        batch_size: int = 1024,
        hidden_layer_sizes: list[int] = [1024, 256],
        #[768],
        alpha: float = 1e-4,
        time_limit_seconds: float | None = 55.0,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = alpha
        self.time_limit_seconds = time_limit_seconds

        if seed is not None:
            np.random.seed(seed)

        self.scaler = StandardScaler(with_mean=True, with_std=True)

        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="sgd",
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate="constant",
            learning_rate_init=self.learning_rate,
            momentum=0.9,
            nesterovs_momentum=True,
            shuffle=True,
            #max_iter=1,          # one internal pass per partial_fit call
            max_iter = self.epochs,
            warm_start=False,
            early_stopping=False,
            random_state=self.seed,
            tol=0.0,
            verbose=True,
        )

        self._is_fitted = False
        self._classes = np.arange(self.output_dim, dtype=np.int64)

    def _prepare_inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = X.reshape(len(X), -1) / 255.0
        return X

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y).ravel().astype(np.int64)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X = self._prepare_inputs(X_train)
        y = self._prepare_targets(y_train)

        #X = self.scaler.fit_transform(X)

        # X = X_train
        # y = y_train

        # start_time = time.time()
        # first = True

        # for _ in range(self.epochs):
        #     if self.time_limit_seconds is not None:
        #         if time.time() - start_time >= self.time_limit_seconds:
        #             break

        #     if first:
        #         self.model.partial_fit(X, y, classes=self._classes)
        #         first = False
        #     else:
        #         self.model.partial_fit(X, y)

        self.model.fit(X,y)

        self._is_fitted = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Agent must be trained before calling predict().")

        X = self._prepare_inputs(X_test)
        #X=X_test
        #X = self.scaler.transform(X)
        return self.model.predict(X)