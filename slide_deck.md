# Pooled LSTM for Multi-Stock Return Forecasting — Slide Deck Outline

Source document for Gamma (or manual slide building). 11 slides (exceeds the
Gamma free-tier cap of 10 — plan to build/export on a paid tier or manually).
Each slide lists: the text/bullets to generate, which real PNG to manually place
(do not let Gamma generate an AI image for these — insert the actual file), and
speaker notes for the live talk (5-7 minutes total, 2 presenters).

**Suggested split:** Presenter 1 covers Slides 1-4 (setup → architecture),
Presenter 2 covers Slides 5-7 (training → results → published-research
comparison), Presenter 1 covers Slides 8-9 (error analysis → improvement
attempt), Presenter 2 closes with Slide 10 (limitations), and either presenter
can take Slide 11 (questions/demo). The back-and-forth keeps energy up over
11 slides rather than one long unbroken block per person.

---

## Slide 1 — Title & Team
**Text:**
- Pooled LSTM for Multi-Stock Return Forecasting
- [Team member names]
- [Course / cohort name] — Project 1: Neural Network Families

**Image:** None

**Speaker (1):** Quick intro — who's presenting, one sentence on what the project is (predicting next-day stock returns for ~100 companies with one shared model).

---

## Slide 2 — Problem Statement
**Text:**
- What: Predict next-day returns for ~100 large-cap US stocks using a single shared LSTM model instead of training one model per stock
- Why it matters: Tests whether pooling data across many related time series improves generalization versus modeling each stock in isolation
- Who would use this: Illustrates a realistic quant-research workflow — the same pattern (pooled sequence models, honest baseline comparison) applies to any multi-entity forecasting problem, not just stocks

**Image:** None

**Speaker (1):** Be upfront here that this is a research/engineering exercise in the ML workflow, not a claim that the output is a profitable trading signal — sets expectations honestly before results are shown later.

---

## Slide 3 — Dataset & Preprocessing
**Text:**
- Source: Yahoo Finance (via `yfinance`), 2018–present — ~100 large-cap US tickers + SPY as market benchmark
- 98 of 100 tickers returned usable data (2 skipped automatically, handled gracefully in code)
- ~160K training sequences, ~36K validation sequences (60-day sliding windows)
- Chronological (not random) 80/20 train/validation split — scaler fit ONLY on training data — both choices prevent lookahead bias in financial time series

**Image:** None

**Speaker (1):** Emphasize the chronological split and train-only scaling even though it's not visually exciting — a random split or scaler fit on all data would silently leak future information into training, a common mistake in financial ML.

---

## Slide 4 — Model Architecture
**Text:**
- Learned stock embedding (dim 8) — lets all stocks share one model while preserving stock-specific identity
- Two stacked LSTM layers (48 → 24 units) over 60-day windows
- Dense head → single output: predicted next-day return
- Loss: Huber (robust to outlier return days); Optimizer: Adam (lr=1e-4)

**Image:** `model_architecture.png`

**Speaker (1):** Explain the embedding idea simply: instead of one-hot encoding which stock this is, the model learns a compact numeric "fingerprint" per stock during training. Hand off to Presenter 2 here.

---

## Slide 5 — Training
**Text:**
- `EarlyStopping` (patience=4, restores best weights) — halts automatically once validation loss stops improving
- Train loss keeps dropping while validation loss stays flat — the two curves never converge
- This is early evidence the model finds whatever weak signal exists almost immediately, then starts fitting noise

**Image:** `training_curves.png`

**Speaker (2):** Point directly at the gap between the two lines — this single chart previews the "signal ceiling" story that the rest of the talk builds on.

---

## Slide 6 — Results
**Text:**
- Benchmarked against 3 baselines — predict-zero, persistence, gradient-boosted trees
- LSTM's error (MSE 1.0770) is barely better than guessing no change at all (1.0835) — about a 0.6% improvement
- Directional accuracy (50.12%) is statistically indistinguishable from persistence (49.09%), GBM (50.66%), and a coin flip

**Image:** `results_table.png`

**Callout box:** Guessing nothing, copying yesterday, and our full LSTM all score about the same — roughly 50%. Daily stock moves are famously close to random, so it makes sense that no method here found a real pattern to exploit.

**Speaker (2) — this is the key talking point of the whole presentation, give it room, say it explicitly:**
*"Whether we guessed nothing, copied yesterday's move, or used our full LSTM, we all landed around the same score — about 50%. That's less about our model and more about how unpredictable daily stock prices actually are — they're driven by news and events no model can see coming just from price history alone. So this result is expected, not a shortcoming in what we built."*

---

## Slide 7 — How This Compares to Published Research
**Text:**
- Ran the same kind of backtest Fischer & Krauss (2018, *European Journal of Operational Research*) used on our own model: rank all stocks daily by predicted return, go long the top 10, short the bottom 10, measure the realized portfolio return
- Our model: **+0.03% avg. daily return, Sharpe 0.31** — their published LSTM on S&P 500 stocks (1992–2015): **+0.46%, Sharpe 5.8** (before transaction costs)
- Separately, a systematic survey of deep learning stock prediction research (*Artificial Intelligence Review*) found that of 10,000+ papers reviewed, **only 35 performed proper backtesting** — the survey's own conclusion is that published accuracy claims in this field **"often overstate practical utility"**

**Image:** `fk_comparison.png`

**Sources (for reference / if asked):**
- Fischer, T. & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.
- "Deep learning in the stock market—a systematic survey of practice, backtesting, and applications." *Artificial Intelligence Review* (Springer). https://pmc.ncbi.nlm.nih.gov/articles/PMC9245389/

**Speaker (2) — this directly answers "how does this compare to real-world models," addresses it head-on:**
*"We wanted a real comparison, not just a citation, so we ran the same kind of backtest a well-known 2018 study used: each day, rank stocks by predicted return, go long the top 10, short the bottom 10, and measure what actually happened. Our model's edge came out to +0.03% average daily return with a Sharpe ratio of 0.31 — their published result was +0.46% and a Sharpe of 5.8. So our edge is smaller, which is worth being upfront about. But it's the same direction of finding — even their well-regarded published result only found a modest edge, on the same kind of large-cap US stocks we used. And separately, a systematic survey of this entire research area found that out of over 10,000 papers, only 35 actually did proper backtesting, concluding that published accuracy claims in this field often overstate practical utility. Our methodology — chronological splitting, train-only scaling, honest baseline comparisons — is exactly the rigor that survey identifies as rare."*

---

## Slide 8 — Error Analysis
**Text:**
- Table: the 15 largest prediction misses on the validation set (ticker, date, predicted vs. actual)
- Worst misses cluster around large real-world moves the model had no way to anticipate (e.g., earnings surprises)
- Confirms the model isn't "broken" on specific stocks — it's structurally blind to event-driven moves, since it only sees price/volume history

**Image:** `large_errors.png`

**Speaker (1):** Pick 1-2 rows to narrate live, e.g. "ORCL on 2025-09-10 — model predicted almost no change, actual return was enormous, almost certainly a specific news event the model has no visibility into."

---

## Slide 9 — Improvement Attempt: Capacity Ablation
**Text:**
- Question tested: is the performance ceiling caused by this specific model being too big, too small, or badly tuned?
- Trained 4 model sizes spanning a **25x parameter range** (3,017 to 75,521 params) on identical data
- All four converged to nearly the same validation loss (0.3635–0.3653) — larger models just overfit faster, smaller ones took longer to hit the same wall

**Image:** `capacity_ablation.png`

**Speaker (1):** *"This confirms the ceiling belongs to the data's limited signal content, not our architecture. A 25x larger model should have clearly outperformed a 25x smaller one if capacity were the bottleneck — it didn't."* This is your strongest evidence-backed moment — say it plainly, don't hedge.

---

## Slide 10 — Limitations & Next Steps
**Text:**
- Limitations: only 5 features, no cross-stock correlation modeling, single global scaler, daily horizon is close to the noise floor
- Proposed extensions: GNN + LSTM hybrid for cross-stock relationships, longer forecast horizons, richer feature set (order flow, fundamentals)
- Despite the modest predictive accuracy, the full pipeline was shipped as a real product

**Image:** `system_diagram.png`

**Speaker (2):** Use the diagram to pivot from "the model's accuracy is limited" to "and yet we still built and deployed a complete, working system end-to-end" — sets up the live demo.

---

## Slide 11 — Questions / Live Demo
**Text:**
- Thank you — happy to answer questions
- [Switch to live browser demo of the Flask app here]

**Image:** None (live demo covers this instead of a screenshot)

**Speaker (1 or 2):** Open the actual running web app in a browser and show real predictions — no need for a static image since you're demonstrating it live.
