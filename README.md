# Pooled LSTM for Multi-Stock Return Forecasting

A pooled LSTM network that predicts next-day returns across ~100 large-cap US equities using a single shared model with learned per-stock embeddings, benchmarked against baseline models and evaluated with an honest read on real-world predictive value.

---

## Tech Stack
- **Language:** Python 3.12
- **Frameworks:** TensorFlow / Keras, PyTorch
- **Libraries:** pandas, numpy, scikit-learn, yfinance, matplotlib, jupyterlab, tensorboard

---

## Project Structure
```
AI_project_1/
  main.py               # TensorFlow/Keras implementation — full pipeline:
                         #   data fetch, feature engineering, training, evaluation
  model.py               # Standalone model-definition module (architecture only)
  model_pytorch.py        # PyTorch counterpart — same architecture, custom
                         #   training loop, reuses main.py's data pipeline
  requirements.txt       # Python dependencies
  LIMITATIONS.md         # Results, architectural limitations, proposed extensions
  README.md
  .venv312/               # Python 3.12 virtual environment (gitignored)
  logs/                   # TensorBoard training logs (gitignored)
```

---

## Data Flow
```
Yahoo Finance (yfinance API)
    ↓
StockDataFetcher.fetch_raw()        — pulls OHLCV per ticker, skips delisted/failed symbols
StockDataFetcher.fetch_market_return() — pulls SPY daily return (market benchmark)
    ↓
engineer_features()                 — computes return, RSI, MACD, volatility, market_return
    ↓
chronological_split()               — splits each stock's timeline 80/20 by date (no lookahead leakage)
    ↓
scale_features()                    — StandardScaler fit on TRAIN data only, applied to both splits
    ↓
build_pooled_dataset()              — builds 60-day sliding windows, pools all stocks together
    ↓
Pooled LSTM + Stock Embedding       — TensorFlow/Keras (main.py) or PyTorch (model_pytorch.py)
    ↓
Evaluation                          — MSE/MAE + directional accuracy vs. 3 baseline models
```

---

## Data Sources
| Name | Format | Location |
|------|--------|----------|
| Stock OHLCV | API | Yahoo Finance via `yfinance` (~100 large-cap US tickers, 2018–present) |
| Market benchmark (SPY) | API | Yahoo Finance via `yfinance` |

**Universe:** ~100 large-cap US tickers (S&P 100-style list). 98 of 100 return usable data — see Known Data Gaps below.

---

## Feature Engineering
| Feature | Description |
|---------|--------------|
| `return` | Daily percentage price change (also the prediction target) |
| `rsi` | 14-day Relative Strength Index |
| `macd` | MACD histogram (12/26/9 EMA) |
| `volatility` | 20-day rolling standard deviation of returns |
| `market_return` | SPY's same-day return — gives the model visibility into broad market moves, not just one stock in isolation |

---

## Model Architecture
| Layer | Purpose |
|-------|---------|
| `stock_embedding` (Embedding, dim=8) | Learns a dense vector per stock — enables weight-sharing across all stocks while preserving stock-specific identity, without one-hot encoding |
| `LSTM` (48 units, `return_sequences=True`) | Captures short- and long-term temporal patterns across the 60-day input window |
| `Dropout` (0.3) | Regularization |
| `LSTM` (24 units) | Compresses the sequence down to a final hidden state |
| `Dropout` (0.3) | Regularization |
| `Dense` (16, ReLU) | Nonlinear projection head |
| `Dense` (1, linear) | Output: predicted next-day return |

- **Loss:** Huber loss — quadratic for small errors, linear for large ones, robust to outlier return days (e.g. earnings surprises)
- **Optimizer:** Adam, learning rate `1e-4`
- **Callbacks:** `EarlyStopping` (patience=4, restores best weights), `TensorBoard`

---

## Evaluation Results
Directional accuracy (% of correct up/down calls) and MSE on the full validation set, benchmarked against three baselines:

| Model | MSE (scaled) | Directional Accuracy |
|-------|-------------:|----------------------:|
| Pooled LSTM + stock embedding | ~1.07 | ~50–51% |
| Baseline: predict zero return | ~1.08 | n/a |
| Baseline: persistence (repeat yesterday's sign) | n/a | ~49% |
| Baseline: gradient-boosted trees (no sequence/embedding) | ~1.10–1.12 | ~51% |

The LSTM does not show a meaningful directional edge over the baselines — see [LIMITATIONS.md](LIMITATIONS.md) for full numbers, discussion, and why this is the expected, honest result for this problem rather than a pipeline defect.

---

## Known Data Gaps
Two tickers are skipped automatically during data fetching (handled gracefully in `fetch_raw()` — logs a skip message, continues with the remaining 98):

| Ticker | Reason |
|--------|--------|
| `MMC` | Insufficient/empty data returned by Yahoo Finance |
| `FI` | Yahoo Finance's `quoteSummary` API intermittently 404s for this symbol (Fiserv's current ticker, post-rebrand from `FISV`) |

---

## Future Plans
- **GNN + LSTM hybrid** — model stocks as nodes in a graph (sector membership, historical correlation) so the model can use cross-stock relationships, not just each stock in isolation
- **Multivariate cross-stock input** — feed all/correlated stocks' features per timestep instead of one stock per sequence
- **Per-sector or per-stock scaling** instead of one global `StandardScaler`
- **Longer forecast horizons** (5-day, 20-day forward return) where technical signals historically show a better signal-to-noise ratio
- **Richer feature set** — order-flow/volume-imbalance, options implied volatility, earnings-calendar proximity, cross-sectional rank features
- **Training curves + prediction-vs-actual plots** — static matplotlib visuals for presentation/reporting
- **Presentation deck** — problem statement, dataset, architecture, results, and limitations slides

---

## Game Plan — Progress Tracker
- [x] Step 1: Python 3.12 virtual environment set up (`.venv312`) for TensorFlow compatibility
- [x] Step 2: `requirements.txt` (tensorflow, tensorboard, numpy, pandas, matplotlib, jupyterlab, yfinance, scikit-learn, torch)
- [x] Step 3: Data pipeline — `StockDataFetcher` (OHLCV fetch, feature engineering, chronological split, train-only scaling)
- [x] Step 4: Pooled LSTM + stock embedding architecture (TensorFlow/Keras)
- [x] Step 5: Sliding-window pooled dataset builder across all stocks
- [x] Step 6: Model training with `EarlyStopping` + `TensorBoard` callbacks
- [x] Step 7: Added market-relative (SPY) return feature
- [x] Step 8: Switched loss function from MSE to Huber loss
- [x] Step 9: Full validation-set evaluation — directional accuracy vs. baselines (zero-predictor, persistence, gradient-boosted trees)
- [x] Step 10: PyTorch counterpart (`model_pytorch.py`) with custom training loop
- [x] Step 11: `LIMITATIONS.md` — documented findings, architectural limitations, proposed extensions
- [ ] Step 12: Training curves plot (matplotlib)
- [ ] Step 13: Prediction-vs-actual plot
- [ ] Step 14: Examples of large prediction errors
- [ ] Step 15: Presentation slide deck

---

## Setup
This project requires **Python 3.9–3.12** (TensorFlow does not yet support 3.13+), plus Graphviz for the model architecture diagram (`plot_model` needs the actual Graphviz binary, not just the `pydot` Python package).

```bash
# Install Python 3.12 if needed
brew install python@3.12

# Install Graphviz (required for tf.keras.utils.plot_model)
brew install graphviz

# Create and activate a virtual environment
/opt/homebrew/bin/python3.12 -m venv .venv312
source .venv312/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Project
Both scripts fetch live data from Yahoo Finance and require outbound internet access.

```bash
# TensorFlow/Keras version
python main.py

# PyTorch version (same architecture, custom training loop)
python model_pytorch.py
```

Each run fetches ~100 tickers, engineers features, builds a leakage-free chronological train/validation split, trains with early stopping, then prints a full evaluation against baseline models plus sample next-day predictions. `main.py` also writes `model_architecture.png` — a diagram of the model's layers and shapes, generated directly from the model object via `tf.keras.utils.plot_model`.

---

## Notes
No environment variables or credentials are required — all data comes from Yahoo Finance's public API via `yfinance`.
