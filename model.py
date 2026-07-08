"""
Approach 2: Pooled LSTM with Stock Embeddings (TensorFlow / Keras version)
============================================================================
One shared model trained across many stocks. Each stock is identified by a
learned embedding vector, concatenated onto the per-timestep feature vector,
so the model shares weights across all stocks but can still tell them apart.

Pipeline:
  1. Build per-stock feature sequences (returns, technical indicators)
  2. Attach a stock_id to every sequence
  3. Pool all stocks' sequences into one training set
  4. Model: Embedding(stock_id) + LSTM(features) -> Dense -> scalar prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# 1. Data prep: build sliding-window sequences across MULTIPLE stocks
# ---------------------------------------------------------------------------
def build_pooled_dataset(stock_data: dict[str, pd.DataFrame], window: int = 60, horizon: int = 1):
    """
    stock_data: {ticker: DataFrame} where each DataFrame has already-scaled
    columns like ['return', 'rsi', 'macd', 'volatility'], sorted chronologically.
    Assumes 'return' is column 0 (the prediction target).

    Returns:
        X          -> (num_samples, window, num_features)  float32
        stock_ids  -> (num_samples,)                        int32
        y          -> (num_samples,)                        float32
        stock_to_idx -> dict mapping ticker -> integer id
    """
    stock_to_idx = {ticker: i for i, ticker in enumerate(stock_data.keys())}

    X_list, id_list, y_list = [], [], []

    for ticker, df in stock_data.items():
        arr = df.values.astype(np.float32)
        stock_idx = stock_to_idx[ticker]

        max_start = len(arr) - window - horizon
        for start in range(max_start):
            X_list.append(arr[start : start + window])
            y_list.append(arr[start + window + horizon - 1, 0])  # next return
            id_list.append(stock_idx)

    X = np.stack(X_list)                       # (N, window, num_features)
    stock_ids = np.array(id_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.float32)

    return X, stock_ids, y, stock_to_idx


# ---------------------------------------------------------------------------
# 2. Model: LSTM + learned stock embedding, shared weights across all stocks
# ---------------------------------------------------------------------------
def build_pooled_lstm_model(
    window: int,
    num_features: int,
    num_stocks: int,
    embedding_dim: int = 8,
    hidden_size1: int = 64,
    hidden_size2: int = 32,
    dropout: float = 0.2,
) -> Model:
    # Two inputs: the feature sequence, and the stock's integer ID
    feature_input = layers.Input(shape=(window, num_features), name="features")
    stock_input = layers.Input(shape=(), dtype="int32", name="stock_id")

    # Embedding: (batch,) -> (batch, embedding_dim)
    stock_vec = layers.Embedding(input_dim=num_stocks, output_dim=embedding_dim, name="stock_embedding")(stock_input)

    # Repeat the embedding across every timestep, then concat onto features
    stock_vec = layers.RepeatVector(window)(stock_vec)              # (batch, window, embedding_dim)
    x = layers.Concatenate(axis=-1)([feature_input, stock_vec])     # (batch, window, num_features+embedding_dim)

    # Stacked LSTMs — first returns full sequence, second returns final state only
    x = layers.LSTM(hidden_size1, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(hidden_size2, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)

    # Dense head -> scalar predicted return
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="linear", name="predicted_return")(x)

    model = Model(inputs=[feature_input, stock_input], outputs=output)
    return model


