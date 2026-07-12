#!/usr/bin/env python3
"""
Flask web app that loads the trained pooled LSTM model and displays
next-day stock predictions, split into UP and DOWN columns.

Usage:
    python app.py
"""

import os
import json
import pickle
import threading
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from flask import Flask, render_template

from main import StockDataFetcher

app = Flask(__name__)

SAVE_DIR = "saved_model"
WINDOW = 60
CACHE_REFRESH_MINUTES = 15

# --- Load saved artifacts at startup ---
if not os.path.isdir(SAVE_DIR):
    print(f"ERROR: '{SAVE_DIR}/' not found. Run 'python train_and_save.py' first.")
    raise SystemExit(1)

print("Loading saved model and artifacts...")
model = tf.keras.models.load_model(os.path.join(SAVE_DIR, "model.keras"))
with open(os.path.join(SAVE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(SAVE_DIR, "stock_to_idx.json"), "r") as f:
    stock_to_idx = json.load(f)
with open(os.path.join(SAVE_DIR, "tickers.json"), "r") as f:
    tickers = json.load(f)
print(f"Loaded model, scaler, and mapping for {len(tickers)} tickers.")

# --- Cached predictions ---
_cache = {"results": None, "timestamp": None, "ready": False}
_lock = threading.Lock()


def fetch_and_predict():
    """Fetch latest data, engineer features, and predict next-day returns for all tickers."""
    fetcher = StockDataFetcher(tickers=tickers, start="2023-01-01")
    raw_data = fetcher.fetch_raw()
    market_return = fetcher.fetch_market_return()
    feature_data = fetcher.engineer_features(raw_data, market_return)

    results = []
    for ticker in tickers:
        if ticker not in feature_data or ticker not in stock_to_idx:
            continue
        df = feature_data[ticker]
        if len(df) < WINDOW:
            continue

        latest = df.iloc[-WINDOW:].values.astype(np.float32)
        scaled = scaler.transform(latest)
        scaled = scaled[np.newaxis, :, :]

        stock_id = np.array([stock_to_idx[ticker]], dtype=np.int32)
        pred_scaled = model.predict([scaled, stock_id], verbose=0)[0, 0]

        dummy = np.zeros((1, 5))
        dummy[0, 0] = pred_scaled
        pred_raw = float(scaler.inverse_transform(dummy)[0, 0])

        results.append({
            "ticker": ticker,
            "prediction": pred_raw,
            "direction": "UP" if pred_raw > 0 else "DOWN",
        })

    return results


def refresh_cache():
    """Background task: fetch predictions and update the cache."""
    global _cache
    print("[cache] Refreshing predictions...")
    try:
        results = fetch_and_predict()
        with _lock:
            _cache["results"] = results
            _cache["timestamp"] = time.time()
            _cache["ready"] = True
        print(f"[cache] Refresh complete: {len(results)} stocks")
    except Exception as e:
        print(f"[cache] Refresh failed: {e}")


def cache_worker():
    """Run initial fetch, then refresh periodically."""
    refresh_cache()
    while True:
        time.sleep(CACHE_REFRESH_MINUTES * 60)
        refresh_cache()


@app.route("/")
def index():
    with _lock:
        ready = _cache["ready"]
        results = _cache["results"]
        ts = _cache["timestamp"]

    if not ready:
        return render_template("index.html", up_stocks=None, down_stocks=None, now=""), 200

    now_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    up_stocks = [r for r in results if r["direction"] == "UP"]
    down_stocks = [r for r in results if r["direction"] == "DOWN"]
    up_stocks.sort(key=lambda x: abs(x["prediction"]), reverse=True)
    down_stocks.sort(key=lambda x: abs(x["prediction"]), reverse=True)

    return render_template(
        "index.html",
        up_stocks=up_stocks,
        down_stocks=down_stocks,
        now=now_str,
    )


if __name__ == "__main__":
    thread = threading.Thread(target=cache_worker, daemon=True)
    thread.start()
    app.run(host="127.0.0.1", port=8080, debug=False)
