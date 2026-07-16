"""
Small, "reliable" universe experiment.

Tests a question raised in Q&A: if we train the same pooled LSTM on a much
smaller set of large, highly liquid "reliable" stocks (5, instead of ~100),
does directional accuracy or validation loss change?

Reuses StockDataFetcher, build_pooled_dataset, and build_pooled_lstm_model
from main.py as-is — no duplicated pipeline logic, same architecture and
hyperparameters as the full-universe model, so this is a controlled
comparison that isolates the effect of universe size.

Also reports UNSCALED metrics (real % returns), not just the scaled MSE,
since StandardScaler re-normalizes to ~unit variance for whatever stocks are
in the training set — a naive scaled-MSE comparison between a 5-stock and a
98-stock run isn't actually apples-to-apples on its own.

Usage:
    python small_universe_experiment.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from main import StockDataFetcher, build_pooled_dataset, build_pooled_lstm_model

# Five large, highly liquid, well-covered mega-caps — a reasonable stand-in
# for "very reliable" stocks, in contrast to the ~100-ticker broad universe.
RELIABLE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]

WINDOW = 60
HORIZON = 1
EMBEDDING_DIM = 8
EPOCHS = 20
BATCH_SIZE = 64

# Known reference numbers from the full ~98-stock model (LIMITATIONS.md /
# README.md), for direct comparison against this smaller-universe run.
BASELINE_98_STOCK = {
    "scaled_mse": 1.0770,
    "directional_accuracy": 0.5012,
    "best_epoch_range": "4-6",
}


def main():
    fetcher = StockDataFetcher(tickers=RELIABLE_TICKERS, start="2018-01-01")
    print(f"Fetching data for {len(fetcher.tickers)} tickers: {RELIABLE_TICKERS}")
    raw_data = fetcher.fetch_raw()
    print(f"Successfully fetched {len(raw_data)} tickers.")
    market_return = fetcher.fetch_market_return()

    feature_data = fetcher.engineer_features(raw_data, market_return)
    train_data, val_data = fetcher.chronological_split(feature_data, train_frac=0.8)
    scaled_train, scaled_val, scaler = fetcher.scale_features(train_data, val_data)

    stock_to_idx = {ticker: i for i, ticker in enumerate(scaled_train.keys())}
    X_train, id_train, y_train = build_pooled_dataset(scaled_train, stock_to_idx, window=WINDOW, horizon=HORIZON)
    X_val, id_val, y_val = build_pooled_dataset(scaled_val, stock_to_idx, window=WINDOW, horizon=HORIZON)
    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    model = build_pooled_lstm_model(
        window=WINDOW,
        num_features=X_train.shape[-1],
        num_stocks=len(stock_to_idx),
        embedding_dim=EMBEDDING_DIM,
        dropout=0.3,
        hidden_size1=48,
        hidden_size2=24,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.Huber(),
        metrics=["mae", "mse"],
    )
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    history = model.fit(
        [X_train, id_train], y_train,
        validation_data=([X_val, id_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
    )

    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1
    stopped_epoch = len(val_losses)

    # --- Evaluation: scaled metrics (comparable to the 98-stock numbers) ---
    preds = model.predict([X_val, id_val], verbose=0).flatten()
    scaled_mse = float(np.mean((preds - y_val) ** 2))
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y_val)))

    # Zero-predictor baseline, computed on THIS run's validation set. Since
    # StandardScaler is fit on THIS training set only, the "expected" scaled
    # MSE of a zero-predictor isn't guaranteed to be ~1.0 here the way it was
    # for the 98-stock run — this checks whether scaled_mse actually beats a
    # trivial baseline, or whether the 0.93 vs 1.08 difference is just a
    # variance/regime artifact of a different, smaller stock set.
    zero_mse = float(np.mean(y_val ** 2))

    # --- Evaluation: UNSCALED metrics (real % returns) — this is the fair,
    # trustworthy comparison, since StandardScaler re-normalizes to ~unit
    # variance for whichever stocks are in THIS training set, making raw
    # scaled-MSE comparisons across different universes potentially
    # misleading. Both the model's error AND a zero-predictor baseline are
    # computed in this same real-percentage space, so neither side is
    # distorted by rescaling — this is the number that actually answers
    # "did fewer, more reliable stocks help." ---
    return_mean = scaler.mean_[0]
    return_scale = scaler.scale_[0]
    actual_pct = y_val * return_scale + return_mean
    pred_pct = preds * return_scale + return_mean
    unscaled_mse = float(np.mean((pred_pct - actual_pct) ** 2))
    unscaled_mae_pct = float(np.mean(np.abs(pred_pct - actual_pct)) * 100)
    zero_unscaled_mse = float(np.mean(actual_pct ** 2))
    zero_unscaled_mae_pct = float(np.mean(np.abs(actual_pct)) * 100)

    print("\n" + "=" * 60)
    print("TRUSTWORTHY METRICS (not distorted by per-run rescaling)")
    print("=" * 60)
    print(f"Directional accuracy:              {dir_acc:.2%}")
    print(f"Model MAE (real % daily return):   {unscaled_mae_pct:.3f}%")
    print(f"Zero-baseline MAE (real % return):  {zero_unscaled_mae_pct:.3f}%  <- is the model actually beating this?")
    print(f"Model MSE (real % return, x1e-4):  {unscaled_mse * 1e4:.4f}")
    print(f"Zero-baseline MSE (real %, x1e-4): {zero_unscaled_mse * 1e4:.4f}")

    print("\n" + "=" * 60)
    print("SCALED METRICS (use with caution — NOT comparable across")
    print("different stock universes; see README/notes on why)")
    print("=" * 60)
    print(f"Tickers: {RELIABLE_TICKERS}")
    print(f"Train sequences: {X_train.shape[0]}, Val sequences: {X_val.shape[0]}")
    print(f"Best epoch: {best_epoch}, stopped at: {stopped_epoch}")
    print(f"Scaled val MSE (model):          {scaled_mse:.4f}")
    print(f"Scaled val MSE (zero baseline):  {zero_mse:.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON: 5-stock vs. 98-stock (full universe)")
    print("=" * 60)
    print(f"{'Metric':<28}{'5 reliable stocks':>20}{'98 stocks (full)':>20}")
    print(f"{'Directional accuracy':<28}{dir_acc:>19.2%}{BASELINE_98_STOCK['directional_accuracy']:>19.2%}")
    print(f"{'Best epoch':<28}{str(best_epoch):>20}{BASELINE_98_STOCK['best_epoch_range']:>20}")
    print(f"{'Scaled val MSE (caution)':<28}{scaled_mse:>20.4f}{BASELINE_98_STOCK['scaled_mse']:>20.4f}")


if __name__ == "__main__":
    main()
