"""
Softmax Linear Classification Agent for Permuted MNIST
Uses PyTorch for a single linear layer with softmax (logistic regression).
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int = 784, output_dim: int = 10) -> None:
        super(NeuralNetwork, self).__init__()

        #self.flatten = nn.Flatten()
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            
            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
            
        )

        layer1features= 512
        layer2features = 256

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_dim, out_features=layer1features),
            nn.ReLU(),
            nn.Linear(in_features=layer1features, out_features=layer2features),
            nn.ReLU(),
            nn.Linear(in_features=layer2features, out_features=output_dim),
        )

    def forward(self, x):
        #x = self.flatten(x)
        #x = x.reshape(-1, 1, 28, 28)
        logits = self.classifier(x)
        return logits

class Agent:
    """Linear softmax classifier using PyTorch SGD."""

    def __init__(self, input_dim: int = 784, output_dim: int = 10, seed: int = None,
                 learning_rate: float = 0.05, epochs: int = 45, batch_size: int = 256):        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.model = NeuralNetwork().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the linear model on the provided data."""
        X = torch.from_numpy(X_train.reshape(len(X_train), -1) / 255.).float().to(self.device)
        y = torch.from_numpy(y_train.ravel().astype(np.int64)).to(self.device)

        n_samples = X.shape[0]
        self.model.train()

        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, self.batch_size):
                idx = perm[i:i + self.batch_size]
                X_batch = X[idx]
                y_batch = y[idx]

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for test data."""
        X = torch.from_numpy(X_test.reshape(len(X_test), -1) / 255.).float().to(self.device)
        self.model.eval()
        logits = self.model(X)
        return logits.argmax(dim=1).cpu().numpy()
