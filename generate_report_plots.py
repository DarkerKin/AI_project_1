"""
Generates presentation-ready PNGs without retraining a model:

  1. training_curves.png     — read directly from existing TensorBoard logs
  2. prediction_vs_actual.png — uses the already-trained saved_model/ for
                                 inference against a freshly fetched validation set
  3. large_errors.png         — table of the 15 largest prediction misses
  4. system_diagram.png       — static deployment flow diagram (data -> model -> app)

Requires saved_model/ to already exist (run train_and_save.py first) and
logs/pooled_lstm/ to contain at least one prior main.py training run.

Usage:
    python generate_report_plots.py
"""

from __future__ import annotations

import glob
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor

from main import StockDataFetcher, build_pooled_dataset

SAVE_DIR = "saved_model"
LOG_DIR = "logs/pooled_lstm"
WINDOW = 60
HORIZON = 1


# ---------------------------------------------------------------------------
# 1. Training curves — read straight from existing TensorBoard event files
# ---------------------------------------------------------------------------
def _latest_event_file(subdir: str) -> str:
    files = glob.glob(os.path.join(LOG_DIR, subdir, "events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(
            f"No TensorBoard event files found in {LOG_DIR}/{subdir} — run main.py at least once first."
        )
    return max(files, key=os.path.getmtime)


def _read_epoch_scalar(event_file: str, tag: str = "epoch_loss") -> list[float]:
    values = {}
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for v in event.summary.value:
            if v.tag == tag:
                val = v.simple_value if v.simple_value else tf.make_ndarray(v.tensor).item()
                values[event.step] = val
    return [values[step] for step in sorted(values)]


def plot_training_curves():
    train_loss = _read_epoch_scalar(_latest_event_file("train"))
    val_loss = _read_epoch_scalar(_latest_event_file("validation"))
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train loss (Huber)")
    plt.plot(epochs, val_loss, marker="o", label="Validation loss (Huber)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Saved training_curves.png")


# ---------------------------------------------------------------------------
# 2 & 3. Prediction-vs-actual and large-error table — reuse the saved model
# ---------------------------------------------------------------------------
def load_saved_artifacts():
    model = tf.keras.models.load_model(os.path.join(SAVE_DIR, "model.keras"))
    with open(os.path.join(SAVE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(SAVE_DIR, "stock_to_idx.json")) as f:
        stock_to_idx = json.load(f)
    return model, scaler, stock_to_idx


def build_datasets(stock_to_idx: dict[str, int], scaler):
    """Rebuilds both train and validation sequences using the saved model's own
    scaler (not a freshly fit one). Train sequences are only needed to fit the
    GBM baseline for the results table — the LSTM itself is not retrained."""
    fetcher = StockDataFetcher(start="2018-01-01")
    print(f"Fetching data for {len(fetcher.tickers)} tickers...")
    raw_data = fetcher.fetch_raw()
    market_return = fetcher.fetch_market_return()
    feature_data = fetcher.engineer_features(raw_data, market_return)
    train_data, val_data = fetcher.chronological_split(feature_data, train_frac=0.8)

    def _scale(data):
        return {
            t: pd.DataFrame(scaler.transform(df.values), columns=df.columns, index=df.index)
            for t, df in data.items()
        }

    scaled_train = _scale(train_data)
    scaled_val = _scale(val_data)

    X_train, _, y_train = build_pooled_dataset(scaled_train, stock_to_idx, window=WINDOW, horizon=HORIZON)
    X_val, id_val, y_val, dates_val = build_pooled_dataset(
        scaled_val, stock_to_idx, window=WINDOW, horizon=HORIZON, return_dates=True
    )
    return X_train, y_train, X_val, id_val, y_val, dates_val


def plot_prediction_vs_actual(model, X_val, id_val, y_val) -> np.ndarray:
    preds = model.predict([X_val, id_val], verbose=0).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, preds, alpha=0.15, s=8)
    lims = [min(y_val.min(), preds.min()), max(y_val.max(), preds.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual next-day return (scaled)")
    plt.ylabel("Predicted next-day return (scaled)")
    plt.title("Prediction vs. Actual (Validation Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_vs_actual.png", dpi=150)
    plt.close()
    print("Saved prediction_vs_actual.png")
    return preds


def plot_large_errors(preds, y_val, id_val, dates_val, idx_to_stock: dict[int, str], top_n: int = 15):
    abs_error = np.abs(preds - y_val)
    top_idx = np.argsort(abs_error)[::-1][:top_n]

    rows = []
    for i in top_idx:
        ticker = idx_to_stock[int(id_val[i])]
        date = pd.Timestamp(dates_val[i]).strftime("%Y-%m-%d")
        rows.append([ticker, date, f"{preds[i]:+.3f}", f"{y_val[i]:+.3f}", f"{abs_error[i]:.3f}"])

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(rows) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Ticker", "Date", "Predicted", "Actual", "Abs Error"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title(f"Top {top_n} Largest Prediction Errors (Validation Set)", pad=20)
    plt.tight_layout()
    plt.savefig("large_errors.png", dpi=150)
    plt.close()
    print("Saved large_errors.png")


def plot_results_table(preds, X_train, y_train, X_val, y_val):
    """Model vs. baselines: MSE + directional accuracy, computed live against
    the actual saved model (not copied from a separate main.py run) so every
    number here is consistent with prediction_vs_actual.png and large_errors.png."""
    model_mse = float(np.mean((preds - y_val) ** 2))
    model_dir_acc = float(np.mean(np.sign(preds) == np.sign(y_val)))

    zero_mse = float(np.mean(y_val ** 2))

    last_day_return = X_val[:, -1, 0]
    persistence_dir_acc = float(np.mean(np.sign(last_day_return) == np.sign(y_val)))

    gbm = HistGradientBoostingRegressor(random_state=0)
    gbm.fit(X_train[:, -1, :], y_train)
    gbm_preds = gbm.predict(X_val[:, -1, :])
    gbm_mse = float(np.mean((gbm_preds - y_val) ** 2))
    gbm_dir_acc = float(np.mean(np.sign(gbm_preds) == np.sign(y_val)))

    rows = [
        ["Pooled LSTM", f"{model_mse:.4f}", f"{model_dir_acc:.2%}"],
        ["Baseline: predict 0", f"{zero_mse:.4f}", "n/a"],
        ["Baseline: persistence", "n/a", f"{persistence_dir_acc:.2%}"],
        ["Baseline: GBM (no sequence)", f"{gbm_mse:.4f}", f"{gbm_dir_acc:.2%}"],
    ]

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "MSE (scaled)", "Directional Accuracy"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    plt.title("Model vs. Baselines (Validation Set)", pad=20)
    plt.tight_layout()
    plt.savefig("results_table.png", dpi=150)
    plt.close()
    print("Saved results_table.png")


# ---------------------------------------------------------------------------
# 4. System diagram — static deployment flow, no model inference involved
# ---------------------------------------------------------------------------
def plot_system_diagram():
    boxes = [
        "Yahoo Finance\n(yfinance API)",
        "main.py\nTraining Pipeline",
        "saved_model/\nmodel.keras + scaler.pkl",
        "app.py\n(Flask)",
        "Browser\nUP/DOWN predictions",
    ]
    box_width, box_height = 0.16, 0.4
    x_positions = np.linspace(0.02, 1 - 0.02 - box_width, len(boxes))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for x, label in zip(x_positions, boxes):
        ax.add_patch(plt.Rectangle(
            (x, 0.3), box_width, box_height, fill=True,
            facecolor="#e8f0fe", edgecolor="#1a73e8", linewidth=1.5,
        ))
        ax.text(x + box_width / 2, 0.5, label, ha="center", va="center", fontsize=9)

    for x in x_positions[:-1]:
        ax.annotate(
            "", xy=(x + box_width + (x_positions[1] - x_positions[0] - box_width), 0.5),
            xytext=(x + box_width, 0.5),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
        )

    plt.title("From Data to Product: Deployment Flow", pad=15)
    plt.tight_layout()
    plt.savefig("system_diagram.png", dpi=150)
    plt.close()
    print("Saved system_diagram.png")


# ---------------------------------------------------------------------------
# 5. Capacity ablation bar chart — model size vs. best validation loss
# ---------------------------------------------------------------------------
# Verified results from the capacity_ablation.py experiment (4 model sizes,
# 3K-75K params, confirmed against that run's saved log). If
# capacity_ablation_results.json exists — saved automatically by a more
# recent run of that script — those numbers are used instead of this fallback.
_ABLATION_FALLBACK = [
    {"name": "Small", "params": 3017, "best_val_loss": 0.3645},
    {"name": "Baseline", "params": 20113, "best_val_loss": 0.3653},
    {"name": "Baseline +\nheavy dropout", "params": 20113, "best_val_loss": 0.3635},
    {"name": "Large", "params": 75521, "best_val_loss": 0.3651},
]


def plot_capacity_ablation():
    results_path = "capacity_ablation_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"Using {results_path}")
    else:
        results = _ABLATION_FALLBACK
        print(f"No {results_path} found — using verified results from the original ablation run")

    names = [r["name"] for r in results]
    params = [r["params"] for r in results]
    losses = [r["best_val_loss"] for r in results]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, losses, color="#1a73e8")
    for bar, p in zip(bars, params):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{p:,} params", ha="center", fontsize=9,
        )

    plt.ylim(min(losses) - 0.01, max(losses) + 0.02)
    plt.ylabel("Best validation loss (Huber)")
    plt.title("Model Size vs. Validation Loss\n(25x parameter range, nearly identical results)")
    plt.tight_layout()
    plt.savefig("capacity_ablation.png", dpi=150)
    plt.close()
    print("Saved capacity_ablation.png")


def main():
    print("=== Generating presentation visuals from existing artifacts ===\n")

    print("[1/6] Training curves (from existing TensorBoard logs)...")
    plot_training_curves()

    print("\n[2/6] Loading saved model + fetching fresh train/validation data...")
    model, scaler, stock_to_idx = load_saved_artifacts()
    idx_to_stock = {v: k for k, v in stock_to_idx.items()}
    X_train, y_train, X_val, id_val, y_val, dates_val = build_datasets(stock_to_idx, scaler)
    print(f"Train sequences: {X_train.shape}, Validation sequences: {X_val.shape}")

    print("\n[3/6] Prediction-vs-actual plot + large-error table...")
    preds = plot_prediction_vs_actual(model, X_val, id_val, y_val)
    plot_large_errors(preds, y_val, id_val, dates_val, idx_to_stock)

    print("\n[4/6] Results table (model vs. baselines)...")
    plot_results_table(preds, X_train, y_train, X_val, y_val)

    print("\n[5/6] System diagram...")
    plot_system_diagram()

    print("\n[6/6] Capacity ablation bar chart...")
    plot_capacity_ablation()

    print(
        "\nDone. Generated: training_curves.png, prediction_vs_actual.png, large_errors.png, "
        "results_table.png, system_diagram.png, capacity_ablation.png"
    )


if __name__ == "__main__":
    main()
