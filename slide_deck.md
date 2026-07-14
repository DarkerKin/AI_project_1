# Pooled LSTM for Multi-Stock Return Forecasting — Slide Deck Outline

Source document for Gamma (or manual slide building). 10 slides max (Gamma free tier).
Each slide lists: the text/bullets to generate, which real PNG to manually place
(do not let Gamma generate an AI image for these — insert the actual file), and
speaker notes for the live talk (5-7 minutes total, every team member speaks).

---

## Slide 1 — Title & Team
**Text:**
- Pooled LSTM for Multi-Stock Return Forecasting
- [Team member names]
- [Course / cohort name] — Project 1: Neural Network Families

**Image:** None

**Speaker notes:** Quick intro — who's presenting, one sentence on what the project is (predicting next-day stock returns for ~100 companies with one shared model).

---

## Slide 2 — Problem Statement
**Text:**
- What: Predict next-day returns for ~100 large-cap US stocks using a single shared LSTM model instead of training one model per stock
- Why it matters: Tests whether pooling data across many related time series improves generalization versus modeling each stock in isolation
- Who would use this: Illustrates a realistic quant-research workflow — the same pattern (pooled sequence models, honest baseline comparison) applies to any multi-entity forecasting problem, not just stocks

**Image:** None

**Speaker notes:** Be upfront here that this is a research/engineering exercise in the ML workflow, not a claim that the output is a profitable trading signal — sets expectations honestly before results are shown later.

---

## Slide 3 — Dataset Overview
**Text:**
- Source: Yahoo Finance (via `yfinance`), 2018–present
- ~100 large-cap US tickers (S&P 100-style universe) + SPY as a market benchmark
- 98 of 100 tickers returned usable data (2 skipped automatically — delisted/API gaps)
- ~160K training sequences, ~36K validation sequences (60-day sliding windows)

**Image:** None (optional: simple stat callout, no chart needed)

**Speaker notes:** Mention the two skipped tickers (MMC, FI) are handled gracefully in code, not manual cleanup — shows the pipeline is robust to real-world API messiness.

---

## Slide 4 — Preprocessing
**Text:**
- Feature engineering: daily return, RSI, MACD, 20-day volatility, market-relative return (SPY)
- Chronological (not random) 80/20 train/validation split — per stock, by date
- Feature scaling (`StandardScaler`) fit ONLY on training data, applied to both splits
- Both choices exist specifically to prevent lookahead bias / data leakage in financial time series

**Image:** None

**Speaker notes:** This is worth emphasizing even though it's not visually exciting — a random split or scaler fit on all data would silently leak future information into training, a common mistake in financial ML. Getting this right is a core part of the methodology.

---

## Slide 5 — Model Architecture
**Text:**
- Learned stock embedding (dim 8) — lets all stocks share one model while preserving stock-specific identity
- Two stacked LSTM layers (48 → 24 units) over 60-day windows
- Dense head → single output: predicted next-day return
- Loss: Huber (robust to outlier return days); Optimizer: Adam (lr=1e-4)

**Image:** `model_architecture.png`

**Speaker notes:** Explain the embedding idea simply: instead of one-hot encoding which stock this is, the model learns a compact numeric "fingerprint" per stock during training.

---

## Slide 6 — Training & Results
**Text:**
- Trained with `EarlyStopping` (patience=4, restores best weights) — training halts automatically once validation loss stops improving
- Directional accuracy (correct up/down calls): **~50–51%** — statistically no better than a coin flip
- Benchmarked against 3 baselines: predict-zero, persistence (repeat yesterday), gradient-boosted trees — all land in the same ~49–51% range

**Images:** `training_curves.png` (top/left) + `prediction_vs_actual.png` (bottom/right)

**Speaker notes — this is the key talking point of the whole presentation, say it explicitly:**
*"Predictions cluster in a flat band near zero regardless of what actually happened — and that's the mathematically correct behavior when there's very little reliable signal to act on. If we'd instead seen these dots hug the diagonal 'perfect prediction' line, that would be the red flag, not this — reliably predicting daily stock direction from price data alone essentially doesn't happen legitimately. A result that clean would point to a data leak, not a working model. What we're seeing here is the honest, expected outcome for one of the hardest prediction problems in finance."*

---

## Slide 7 — Error Analysis
**Text:**
- Table: the 15 largest prediction misses on the validation set (ticker, date, predicted vs. actual)
- Pattern: worst misses cluster around large real-world moves the model had no way to anticipate (e.g., earnings surprises)
- Confirms the model isn't "broken" on specific stocks — it's structurally blind to event-driven moves, since it only sees price/volume history

**Image:** `large_errors.png`

**Speaker notes:** Pick 1-2 rows to narrate live, e.g. "ORCL on 2025-09-10 — model predicted almost no change, actual return was enormous, almost certainly a specific news event the model has no visibility into."

---

## Slide 8 — Improvement Attempt: Capacity Ablation
**Text:**
- Question tested: is the performance ceiling caused by this specific model being too big, too small, or badly tuned?
- Trained 4 model sizes spanning a **25x parameter range** (3,017 to 75,521 params) on identical data
- All four converged to nearly the same validation loss (0.3635–0.3653) — larger models just overfit faster, smaller ones took longer to hit the same wall

**Image:** `capacity_ablation.png`

**Speaker notes:** *"This confirms the ceiling belongs to the data's limited signal content, not our architecture. A 25x larger model should have clearly outperformed a 25x smaller one if capacity were the bottleneck — it didn't."* This is your strongest evidence-backed moment — say it plainly, don't hedge.

---

## Slide 9 — Limitations & Next Steps
**Text:**
- Limitations: only 5 features, no cross-stock correlation modeling, single global scaler, daily horizon is close to the noise floor
- Proposed extensions: GNN + LSTM hybrid for cross-stock relationships, longer forecast horizons, richer feature set (order flow, fundamentals)
- Despite the modest predictive accuracy, the full pipeline was shipped as a real product

**Image:** `system_diagram.png`

**Speaker notes:** Use the diagram to pivot from "the model's accuracy is limited" to "and yet we still built and deployed a complete, working system end-to-end" — sets up the live demo.

---

## Slide 10 — Questions / Live Demo
**Text:**
- Thank you — happy to answer questions
- [Optional: switch to live browser demo of the Flask app here]

**Image:** None (live demo covers this instead of a screenshot)

**Speaker notes:** This is where you open the actual running web app in a browser and show real predictions — no need for a static image since you're demonstrating it live.
