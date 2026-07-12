#!/usr/bin/env python3
"""
Trains the pooled LSTM model and saves model, scaler, and stock mappings to disk
so the Flask app can load them without retraining.

Usage:
    python train_and_save.py
"""

import os
import json
import pickle
import tensorflow as tf
from main import StockDataFetcher, build_pooled_dataset, build_pooled_lstm_model


def main():
    WINDOW = 60
    HORIZON = 1
    EMBEDDING_DIM = 8
    EPOCHS = 20
    BATCH_SIZE = 64
    SAVE_DIR = "saved_model"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Step 1: fetch data ---
    fetcher = StockDataFetcher(start="2018-01-01")
    print(f"Fetching data for {len(fetcher.tickers)} tickers...")
    raw_data = fetcher.fetch_raw()
    print(f"Successfully fetched {len(raw_data)} tickers.")
    market_return = fetcher.fetch_market_return()

    # --- Step 2: engineer features ---
    feature_data = fetcher.engineer_features(raw_data, market_return)
    print(f"{len(feature_data)} tickers after feature engineering.")

    # --- Step 3: chronological split + scale ---
    train_data, val_data = fetcher.chronological_split(feature_data, train_frac=0.8)
    scaled_train, scaled_val, scaler = fetcher.scale_features(train_data, val_data)

    # --- Step 4: build pooled dataset ---
    stock_to_idx = {ticker: i for i, ticker in enumerate(scaled_train.keys())}
    X_train, id_train, y_train = build_pooled_dataset(
        scaled_train, stock_to_idx, window=WINDOW, horizon=HORIZON
    )
    X_val, id_val, y_val = build_pooled_dataset(
        scaled_val, stock_to_idx, window=WINDOW, horizon=HORIZON
    )
    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    # --- Step 5: build and train model ---
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

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    model.fit(
        [X_train, id_train], y_train,
        validation_data=([X_val, id_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
    )

    # --- Step 6: save artifacts ---
    model_path = os.path.join(SAVE_DIR, "model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    stock_to_idx_path = os.path.join(SAVE_DIR, "stock_to_idx.json")
    with open(stock_to_idx_path, "w") as f:
        json.dump(stock_to_idx, f, indent=2)
    print(f"Stock mapping saved to {stock_to_idx_path}")

    tickers_path = os.path.join(SAVE_DIR, "tickers.json")
    with open(tickers_path, "w") as f:
        json.dump(list(scaled_train.keys()), f, indent=2)
    print(f"Tickers list saved to {tickers_path}")

    print("\nDone. All artifacts saved to 'saved_model/'. You can now run: python app.py")


if __name__ == "__main__":
    main()
