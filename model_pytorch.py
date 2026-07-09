"""
PyTorch counterpart to the TensorFlow/Keras pooled LSTM in main.py.

Same architecture (learned stock embedding concatenated onto each timestep's
features, feeding a two-layer stacked LSTM), but implemented with a custom
training loop (manual forward pass, backward pass, optimizer step, and early
stopping) instead of a high-level `.fit()` call — for a direct side-by-side
comparison of the two frameworks' ergonomics.

Reuses StockDataFetcher and build_pooled_dataset from main.py as-is: that
code is plain NumPy/pandas with nothing TensorFlow-specific about it, so
there's no reason to duplicate the data pipeline here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from main import StockDataFetcher, build_pooled_dataset


# ---------------------------------------------------------------------------
# 1. Model: same stock-embedding + stacked-LSTM architecture as main.py
# ---------------------------------------------------------------------------
class PooledLSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        embedding_dim: int = 8,
        hidden_size1: int = 48,
        hidden_size2: int = 24,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim)

        self.lstm1 = nn.LSTM(num_features + embedding_dim, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size2, 16)
        self.relu = nn.ReLU()
        self.output = nn.Linear(16, 1)

    def forward(self, x_features: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        window = x_features.shape[1]

        stock_vec = self.stock_embedding(stock_ids)               # (batch, embedding_dim)
        stock_vec = stock_vec.unsqueeze(1).expand(-1, window, -1)  # repeat across every timestep
        x = torch.cat([x_features, stock_vec], dim=-1)             # (batch, window, num_features+embedding_dim)

        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        _, (h_n, _) = self.lstm2(x)      # h_n: final hidden state == Keras' return_sequences=False output
        x = self.dropout2(h_n[-1])

        x = self.relu(self.dense(x))
        return self.output(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. Custom training loop (the PyTorch/Keras contrast point)
# ---------------------------------------------------------------------------
def train_pytorch_model(
    X_train, id_train, y_train, X_val, id_val, y_val,
    num_stocks: int, epochs: int = 20, batch_size: int = 64, patience: int = 4,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on device: {device}")

    model = PooledLSTM(num_features=X_train.shape[-1], num_stocks=num_stocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.HuberLoss()  # same loss choice as the Keras version, for a fair comparison

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(id_train).long(),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val).float().to(device)
    id_val_t = torch.from_numpy(id_val).long().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, idb, yb in train_loader:
            xb, idb, yb = xb.to(device), idb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb, idb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t, id_val_t), y_val_t).item()

        print(f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Manual early stopping + best-weights restore (Keras' EarlyStopping does this
        # for you automatically; here it's spelled out explicitly).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    return model, device


# ---------------------------------------------------------------------------
# 3. Main: same data pipeline as main.py, PyTorch model + training loop
# ---------------------------------------------------------------------------
def main():
    WINDOW = 60
    HORIZON = 1

    fetcher = StockDataFetcher(start="2018-01-01")
    print(f"Fetching data for {len(fetcher.tickers)} tickers...")
    raw_data = fetcher.fetch_raw()
    print(f"Successfully fetched {len(raw_data)} tickers.")
    market_return = fetcher.fetch_market_return()

    feature_data = fetcher.engineer_features(raw_data, market_return)
    print(f"{len(feature_data)} tickers have enough data after feature engineering.")

    train_data, val_data = fetcher.chronological_split(feature_data, train_frac=0.8)
    scaled_train, scaled_val, scaler = fetcher.scale_features(train_data, val_data)

    stock_to_idx = {ticker: i for i, ticker in enumerate(scaled_train.keys())}
    X_train, id_train, y_train = build_pooled_dataset(scaled_train, stock_to_idx, window=WINDOW, horizon=HORIZON)
    X_val, id_val, y_val = build_pooled_dataset(scaled_val, stock_to_idx, window=WINDOW, horizon=HORIZON)
    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    model, device = train_pytorch_model(
        X_train, id_train, y_train, X_val, id_val, y_val, num_stocks=len(stock_to_idx),
    )

    # Same evaluation as main.py: directional accuracy vs. a persistence baseline.
    model.eval()
    with torch.no_grad():
        val_preds = model(
            torch.from_numpy(X_val).float().to(device),
            torch.from_numpy(id_val).long().to(device),
        ).cpu().numpy()

    model_mse = float(np.mean((val_preds - y_val) ** 2))
    model_dir_acc = float(np.mean(np.sign(val_preds) == np.sign(y_val)))
    last_day_return = X_val[:, -1, 0]
    persistence_dir_acc = float(np.mean(np.sign(last_day_return) == np.sign(y_val)))

    print("\n--- Validation evaluation (PyTorch model, full val set) ---")
    print(f"Model MSE:                          {model_mse:.4f}")
    print(f"Model directional accuracy:          {model_dir_acc:.2%}")
    print(f"Baseline (persistence) directional accuracy: {persistence_dir_acc:.2%}")


if __name__ == "__main__":
    main()
