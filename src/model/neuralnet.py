import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),  # output: single logit
        )

    def forward(self, x):
        return self.net(x)  # logits


class NeuralNetClassifier:
    def __init__(self, input_dim: int, seed: int = 42, pos_weight: float = 5.0):
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = MLP(input_dim)
        self.scaler = StandardScaler()
        self.pos_weight = pos_weight

    def _prepare_data(self, X: np.ndarray, y: np.ndarray):
        # Scale the data and convert to tensors
        X_scaled = self.scaler.fit_transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 10):
        train_loader = self._prepare_data(X, y)

        # Define loss and optimizer
        pos_weight_tensor = torch.tensor([self.pos_weight], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor
        )  # for binary classification
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.model(X_batch).squeeze(1)  # shape [batch]
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            avg_loss = epoch_loss / len(train_loader.dataset)
            logger.info(f"Epoch [{epoch+1}/{n_epochs}] Loss: {avg_loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_tensor).squeeze(1)  # shape [n_samples]
            probs = torch.sigmoid(logits).numpy()  # convert logits to probabilities
        return np.vstack([1 - probs, probs]).T  # shape [n_samples, 2]
