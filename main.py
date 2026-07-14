"""
End-to-End Pipeline: Yahoo Finance Data -> Pooled LSTM -> Predictions
========================================================================
1. StockDataFetcher: pulls OHLCV data for ~100 large-cap tickers via yfinance,
   computes engineered features (returns, RSI, MACD, volatility), and scales
   them using train-set-only statistics.
2. Reuses the pooled LSTM + stock-embedding architecture (TensorFlow/Keras)
   from the earlier script.
3. main(): fetches data, builds the pooled dataset, trains the model, and
   prints predicted next-day returns for a sample of stocks.

Requirements:
    pip install yfinance pandas numpy tensorflow scikit-learn

NOTE: This script needs outbound internet access to Yahoo Finance to run.
If you're executing it inside a sandboxed/offline environment, the fetch
step will fail — run it locally instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


# ---------------------------------------------------------------------------
# 1. Data fetching + feature engineering
# ---------------------------------------------------------------------------
class StockDataFetcher:
    """
    Pulls historical OHLCV data for a list of top companies from Yahoo Finance
    and turns it into scaled feature DataFrames ready for the pooled LSTM.
    """

    # Top ~100 large-cap US tickers (S&P 100 style list). Swap this out for
    # a dynamically-fetched list if you want it to stay current.
    # Actual S&P 100 constituents (OEX) as of September 2025
    TOP_100_TICKERS = [
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMAT", "AMD", "AMGN", "AMT", "AMZN",
        "AVGO", "AXP", "BA", "BAC", "BKNG", "BLK", "BMY", "BNY", "BRK-B", "C",
        "CAT", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
        "DE", "DHR", "DIS", "DUK", "EMR", "FDX", "GD", "GE", "GEV", "GILD",
        "GM", "GOOG", "GOOGL", "GS", "HD", "HONA", "IBM", "INTC", "INTU", "ISRG",
        "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD",
        "MDLZ", "MDT", "META", "MMM", "MO", "MRK", "MS", "MSFT", "MU", "NEE",
        "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM",
        "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TMO", "TMUS", "TSLA",
        "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
        "XOM",
    ]

    # Broad-market proxy used to compute each stock's market-relative return feature.
    MARKET_TICKER = "SPY"

    def __init__(self, tickers: list[str] | None = None, start: str = "2018-01-01", end: str | None = None):
        self.tickers = tickers or self.TOP_100_TICKERS
        self.start = start
        self.end = end  # None -> yfinance defaults to today

    def fetch_raw(self) -> dict[str, pd.DataFrame]:
        """Downloads raw OHLCV data per ticker. Skips tickers that fail (delisted, no data, etc.)."""
        raw_data = {}
        for ticker in self.tickers:
            try:
                df = yf.Ticker(ticker).history(start=self.start, end=self.end, auto_adjust=True)
                if df.empty or len(df) < 120:  # need enough rows for indicators + windows
                    print(f"[skip] {ticker}: insufficient data")
                    continue
                raw_data[ticker] = df[["Open", "High", "Low", "Close", "Volume"]]
            except Exception as e:
                print(f"[skip] {ticker}: fetch failed ({e})")
        return raw_data

    def fetch_market_return(self) -> pd.Series:
        """Fetches SPY close price and returns its daily pct-change series (the market's own return),
        used as a market-relative feature so the model can see broad market moves, not just each stock in isolation."""
        df = yf.Ticker(self.MARKET_TICKER).history(start=self.start, end=self.end, auto_adjust=True)
        return df["Close"].pct_change()

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line  # MACD histogram

    def engineer_features(
        self, raw_data: dict[str, pd.DataFrame], market_return: pd.Series
    ) -> dict[str, pd.DataFrame]:
        """Builds ['return', 'rsi', 'macd', 'volatility', 'market_return'] columns per ticker (unscaled).
        'market_return' is SPY's same-day return, aligned by date, giving the model visibility into
        whether the broad market was up or down that day (not just this one stock in isolation)."""
        feature_data = {}
        for ticker, df in raw_data.items():
            close = df["Close"]

            out = pd.DataFrame(index=df.index)
            out["return"] = close.pct_change()
            out["rsi"] = self._compute_rsi(close)
            out["macd"] = self._compute_macd(close)
            out["volatility"] = out["return"].rolling(20).std()
            out["market_return"] = market_return.reindex(out.index)

            out = out.dropna()
            if len(out) < 90:  # need room for a 60-day window + a few targets
                print(f"[skip] {ticker}: not enough rows after feature engineering")
                continue

            feature_data[ticker] = out
        return feature_data

    @staticmethod
    def chronological_split(
        feature_data: dict[str, pd.DataFrame], train_frac: float = 0.8
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Splits each stock's timeline by date BEFORE scaling/pooling, to avoid lookahead leakage."""
        train_data, val_data = {}, {}
        for ticker, df in feature_data.items():
            split_idx = int(len(df) * train_frac)
            train_data[ticker] = df.iloc[:split_idx]
            val_data[ticker] = df.iloc[split_idx:]
        return train_data, val_data

    @staticmethod
    def scale_features(
        train_data: dict[str, pd.DataFrame], val_data: dict[str, pd.DataFrame]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], StandardScaler]:
        """Fits a scaler on TRAIN data only (across all stocks pooled), applies it to both splits."""
        all_train_rows = pd.concat(train_data.values(), axis=0)
        scaler = StandardScaler()
        scaler.fit(all_train_rows.values)

        scaled_train = {t: pd.DataFrame(scaler.transform(df.values), columns=df.columns, index=df.index)
                         for t, df in train_data.items()}
        scaled_val = {t: pd.DataFrame(scaler.transform(df.values), columns=df.columns, index=df.index)
                      for t, df in val_data.items()}

        return scaled_train, scaled_val, scaler


# ---------------------------------------------------------------------------
# 2. Sliding-window dataset builder (pools all stocks together)
# ---------------------------------------------------------------------------
def build_pooled_dataset(stock_data: dict[str, pd.DataFrame], stock_to_idx: dict[str, int],
                          window: int = 60, horizon: int = 1, return_dates: bool = False):
    X_list, id_list, y_list, date_list = [], [], [], []

    for ticker, df in stock_data.items():
        if ticker not in stock_to_idx:
            continue
        arr = df.values.astype(np.float32)
        stock_idx = stock_to_idx[ticker]

        max_start = len(arr) - window - horizon
        for start in range(max_start):
            target_idx = start + window + horizon - 1
            X_list.append(arr[start: start + window])
            y_list.append(arr[target_idx, 0])  # 'return' is column 0
            id_list.append(stock_idx)
            if return_dates:
                date_list.append(df.index[target_idx])

    if not X_list:
        raise ValueError("No sequences built — check that stocks have enough history for the given window.")

    X = np.stack(X_list)
    stock_ids = np.array(id_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.float32)

    if return_dates:
        return X, stock_ids, y, np.array(date_list)
    return X, stock_ids, y


# ---------------------------------------------------------------------------
# 3. Model: pooled LSTM with stock embeddings (same architecture as before)
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
    feature_input = layers.Input(shape=(window, num_features), name="features")
    stock_input = layers.Input(shape=(), dtype="int32", name="stock_id")

    stock_vec = layers.Embedding(input_dim=num_stocks, output_dim=embedding_dim, name="stock_embedding")(stock_input)
    stock_vec = layers.RepeatVector(window)(stock_vec)
    x = layers.Concatenate(axis=-1)([feature_input, stock_vec])

    x = layers.LSTM(hidden_size1, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(hidden_size2, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="linear", name="predicted_return")(x)

    return Model(inputs=[feature_input, stock_input], outputs=output)


# ---------------------------------------------------------------------------
# 4. Main: fetch data -> engineer features -> split -> scale -> train -> predict
# ---------------------------------------------------------------------------
def main():
    WINDOW = 60
    HORIZON = 1
    EMBEDDING_DIM = 8
    EPOCHS = 20
    BATCH_SIZE = 64

    # --- Step 1: fetch top 100 companies' data ---
    fetcher = StockDataFetcher(start="2018-01-01")
    print(f"Fetching data for {len(fetcher.tickers)} tickers...")
    raw_data = fetcher.fetch_raw()
    print(f"Successfully fetched {len(raw_data)} tickers.")
    market_return = fetcher.fetch_market_return()

    # --- Step 2: engineer features (return, rsi, macd, volatility, market_return) ---
    feature_data = fetcher.engineer_features(raw_data, market_return)
    print(f"{len(feature_data)} tickers have enough data after feature engineering.")

    # --- Step 3: chronological split (per stock, BEFORE scaling/pooling) ---
    train_data, val_data = fetcher.chronological_split(feature_data, train_frac=0.8)

    # --- Step 4: scale using train-set statistics only ---
    scaled_train, scaled_val, scaler = fetcher.scale_features(train_data, val_data)

    # --- Step 5: build stock ID mapping and pooled sequences ---
    stock_to_idx = {ticker: i for i, ticker in enumerate(scaled_train.keys())}

    X_train, id_train, y_train = build_pooled_dataset(scaled_train, stock_to_idx, window=WINDOW, horizon=HORIZON)
    X_val, id_val, y_val = build_pooled_dataset(scaled_val, stock_to_idx, window=WINDOW, horizon=HORIZON)

    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    # --- Step 6: build and train the model ---
    num_features = X_train.shape[-1]
    model = build_pooled_lstm_model(
        window=WINDOW,
        num_features=num_features,
        num_stocks=len(stock_to_idx),
        embedding_dim=EMBEDDING_DIM,
        dropout=0.3,
        hidden_size1=48,
        hidden_size2=24,
    )
    # Huber loss: quadratic for small errors, linear for large ones — less dominated by
    # outlier return days (earnings surprises, etc.) than plain MSE.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.Huber(),
        metrics=["mae", "mse"],
    )
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="logs/pooled_lstm", histogram_freq=1)

    model.fit(
        [X_train, id_train], y_train,
        validation_data=([X_val, id_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, tensorboard_cb],
    )

    # --- Step 6.5: full validation-set evaluation — directional accuracy + baselines ---
    # Raw loss numbers are meaningless in isolation. These comparisons answer the real
    # question: is this model actually better than doing nothing (predicting zero),
    # doing the simplest possible thing (assuming tomorrow repeats today's direction),
    # or a much simpler non-sequence model (gradient boosting on that day's features alone)?
    val_preds = model.predict([X_val, id_val], verbose=0).flatten()

    model_mse = float(np.mean((val_preds - y_val) ** 2))
    model_dir_acc = float(np.mean(np.sign(val_preds) == np.sign(y_val)))

    # Baseline A: always predict zero return (no model at all).
    zero_mse = float(np.mean(y_val ** 2))

    # Baseline B: "persistence" — assume tomorrow's return has the same sign as today's
    # (the last day in the input window). The cheapest possible non-trivial baseline.
    last_day_return = X_val[:, -1, 0]  # 'return' is feature column 0
    persistence_dir_acc = float(np.mean(np.sign(last_day_return) == np.sign(y_val)))

    # Baseline C: gradient-boosted trees on the SAME final day's feature vector — no
    # 60-day window, no learned stock embedding. Tells us whether the LSTM's extra
    # architectural complexity is earning its keep over a much simpler tabular model.
    gbm = HistGradientBoostingRegressor(random_state=0)
    gbm.fit(X_train[:, -1, :], y_train)
    gbm_preds = gbm.predict(X_val[:, -1, :])
    gbm_mse = float(np.mean((gbm_preds - y_val) ** 2))
    gbm_dir_acc = float(np.mean(np.sign(gbm_preds) == np.sign(y_val)))

    print("\n--- Validation evaluation (full val set, scaled 'return' space) ---")
    print(f"{'Model':<30}{'MSE':>10}{'Directional Acc':>20}")
    print(f"{'Pooled LSTM':<30}{model_mse:>10.4f}{model_dir_acc:>19.2%}")
    print(f"{'Baseline: predict 0':<30}{zero_mse:>10.4f}{'n/a':>20}")
    print(f"{'Baseline: persistence':<30}{'n/a':>10}{persistence_dir_acc:>19.2%}")
    print(f"{'Baseline: GBM (no sequence)':<30}{gbm_mse:>10.4f}{gbm_dir_acc:>19.2%}")

    # --- Step 7: get predictions (next-day return) for each stock's latest window ---
    print("\nPredicted next-day returns (scaled space, sample of stocks):")
    for ticker in list(stock_to_idx.keys())[:10]:
        df = scaled_val.get(ticker)
        if df is None or len(df) < WINDOW:
            continue
        x_latest = df.values[-WINDOW:].astype(np.float32)[np.newaxis, :, :]
        stock_id = np.array([stock_to_idx[ticker]])
        pred = model.predict([x_latest, stock_id], verbose=0)[0, 0]

        direction = "UP" if pred > 0 else "DOWN" if pred < 0 else "FLAT"
        print(f"  {ticker:6s}  predicted_return={pred:+.5f}  ({direction})")

    print(
        "\nNote: predictions are in SCALED feature space (StandardScaler units), "
        "not raw percentage returns. To interpret as an actual % move, inverse-transform "
        "using `scaler` on the 'return' column, or refit a scaler on the return column alone."
    )


if __name__ == "__main__":
    main()