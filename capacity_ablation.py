"""
Capacity/regularization ablation experiment.

Tests whether the pooled LSTM's early overfitting onset (val_loss diverging
from train_loss around epoch 4-6, as seen in main.py's training runs) is
caused by this specific model being too large or too weakly regularized, or
whether it reflects a genuine ceiling on how much signal exists in the input
features regardless of model size.

Trains several model sizes on the exact same data/split (fetched once and
reused) and compares where each one's validation loss stops improving. If
the divergence point lands at roughly the same epoch across very different
model sizes, that is evidence the ceiling is a property of the data, not a
tuning artifact of the baseline architecture used in main.py.
"""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from main import StockDataFetcher, build_pooled_dataset, build_pooled_lstm_model


def train_variant(
    name, X_train, id_train, y_train, X_val, id_val, y_val, num_stocks,
    hidden_size1, hidden_size2, embedding_dim, dropout,
    epochs=15, batch_size=64, patience=4,
):
    model = build_pooled_lstm_model(
        window=X_train.shape[1],
        num_features=X_train.shape[-1],
        num_stocks=num_stocks,
        embedding_dim=embedding_dim,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        dropout=dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.Huber(),
        metrics=["mse"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        [X_train, id_train], y_train,
        validation_data=([X_val, id_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    val_losses = history.history["val_loss"]
    train_losses = history.history["loss"]
    best_epoch = int(np.argmin(val_losses)) + 1
    stopped_epoch = len(val_losses)
    total_params = model.count_params()

    print(f"\n=== RESULT: {name} ===")
    print(f"Total params: {total_params}")
    print(f"Train loss at best epoch: {train_losses[best_epoch - 1]:.4f}")
    print(f"Best val_loss: {val_losses[best_epoch - 1]:.4f} (epoch {best_epoch})")
    print(f"Stopped at epoch: {stopped_epoch}")

    return {
        "name": name,
        "params": total_params,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": val_losses[best_epoch - 1],
        "train_loss_at_best": train_losses[best_epoch - 1],
    }


def main():
    WINDOW = 60
    HORIZON = 1

    fetcher = StockDataFetcher(start="2018-01-01")
    print("Fetching data once — reused across all model-size variants...")
    raw_data = fetcher.fetch_raw()
    market_return = fetcher.fetch_market_return()
    feature_data = fetcher.engineer_features(raw_data, market_return)
    train_data, val_data = fetcher.chronological_split(feature_data, train_frac=0.8)
    scaled_train, scaled_val, scaler = fetcher.scale_features(train_data, val_data)

    stock_to_idx = {ticker: i for i, ticker in enumerate(scaled_train.keys())}
    X_train, id_train, y_train = build_pooled_dataset(scaled_train, stock_to_idx, window=WINDOW, horizon=HORIZON)
    X_val, id_val, y_val = build_pooled_dataset(scaled_val, stock_to_idx, window=WINDOW, horizon=HORIZON)
    num_stocks = len(stock_to_idx)
    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}\n")

    variants = [
        dict(name="Small (~4x fewer units)", hidden_size1=16, hidden_size2=8, embedding_dim=4, dropout=0.3),
        dict(name="Baseline (main.py's current model)", hidden_size1=48, hidden_size2=24, embedding_dim=8, dropout=0.3),
        dict(name="Baseline + heavy dropout (0.6)", hidden_size1=48, hidden_size2=24, embedding_dim=8, dropout=0.6),
        dict(name="Large (~2x more units)", hidden_size1=96, hidden_size2=48, embedding_dim=16, dropout=0.3),
    ]

    results = []
    for v in variants:
        result = train_variant(
            v["name"], X_train, id_train, y_train, X_val, id_val, y_val, num_stocks,
            hidden_size1=v["hidden_size1"], hidden_size2=v["hidden_size2"],
            embedding_dim=v["embedding_dim"], dropout=v["dropout"],
        )
        results.append(result)

    print("\n\n=== ABLATION SUMMARY ===")
    header = f"{'Variant':<36}{'Params':>10}{'Best Epoch':>12}{'Stopped':>10}{'Best val_loss':>15}"
    print(header)
    for r in results:
        print(f"{r['name']:<36}{r['params']:>10}{r['best_epoch']:>12}{r['stopped_epoch']:>10}{r['best_val_loss']:>15.4f}")

    # Persist results in the repo (not just console output) so they survive
    # beyond this process and can be reused later, e.g. by generate_report_plots.py,
    # without needing to re-run this ~20-minute experiment.
    with open("capacity_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to capacity_ablation_results.json")


if __name__ == "__main__":
    main()
