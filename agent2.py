import time
import numpy as np
import torch
import torch.nn as nn


class Agent:
    def __init__(
        self,
        input_dim: int = 784,
        output_dim: int = 10,
        seed: int | None = None,
        learning_rate: float = 0.12,
        epochs: int = 200,
        batch_size: int = 2048,
        hidden_layer_sizes: list[int] = [3072, 3072],
        weight_decay: float = 1e-4,
        time_limit_seconds: float | None = 54.0,
        num_threads: int = 2,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.weight_decay = weight_decay
        self.time_limit_seconds = time_limit_seconds
        self.num_threads = num_threads
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.device = torch.device("cpu")
        self.model = self._build_model().to(self.device)
        self._init_weights()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.95,
            nesterov=True,
            weight_decay=self.weight_decay,
        )

        self.x_mean = None
        self.x_std = None

    def _build_model(self) -> nn.Module:
        layers: list[nn.Module] = []
        in_features = self.input_dim

        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, self.output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]

        for layer in linear_layers[:-1]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        last = linear_layers[-1]
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last.bias)

    def _prepare_inputs_train(self, X: np.ndarray) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32).reshape(len(X), -1) / 255.0

        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True)
        self.x_std = np.where(self.x_std < 1e-6, 1.0, self.x_std)

        X = (X - self.x_mean) / self.x_std
        return torch.from_numpy(X).to(self.device)

    def _prepare_inputs_test(self, X: np.ndarray) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32).reshape(len(X), -1) / 255.0
        X = (X - self.x_mean) / self.x_std
        return torch.from_numpy(X).to(self.device)

    def _prepare_targets(self, y: np.ndarray) -> torch.Tensor:
        y = np.asarray(y).ravel().astype(np.int64)
        return torch.from_numpy(y).to(self.device)

    def _set_lr_by_progress(self, progress: float) -> None:
        if progress < 0.35:
            lr = self.learning_rate
        elif progress < 0.70:
            lr = self.learning_rate * 0.25
        elif progress < 0.90:
            lr = self.learning_rate * 0.05
        else:
            lr = self.learning_rate * 0.01

        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X = self._prepare_inputs_train(X_train)
        y = self._prepare_targets(y_train)

        n_samples = X.shape[0]
        self.model.train()

        start_time = time.perf_counter()

        for _ in range(self.epochs):
            if self.time_limit_seconds is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed >= self.time_limit_seconds:
                    break
                self._set_lr_by_progress(elapsed / self.time_limit_seconds)

            perm = torch.randperm(n_samples, device=self.device)

            for i in range(0, n_samples, self.batch_size):
                if self.time_limit_seconds is not None:
                    elapsed = time.perf_counter() - start_time
                    if elapsed >= self.time_limit_seconds:
                        return
                    self._set_lr_by_progress(elapsed / self.time_limit_seconds)

                idx = perm[i:i + self.batch_size]
                X_batch = X[idx]
                y_batch = y[idx]

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X = self._prepare_inputs_test(X_test)
        self.model.eval()
        logits = self.model(X)
        return logits.argmax(dim=1).cpu().numpy()