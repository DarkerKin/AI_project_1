# Limitations and Proposed Extensions

## Results

Trained on 98 large-cap US equities (2018–present, chronological 80/20 split
per stock, no lookahead leakage in feature engineering or scaling). Evaluated
on the full validation set:

| Model                          | MSE (scaled)  | Directional accuracy |
|---------------------------------|--------------:|----------------------:|
| Pooled LSTM + stock embedding   | 1.0769        | 50.91%                |
| Baseline: predict zero return   | 1.0826        | n/a                   |
| Baseline: persistence (repeat yesterday's sign) | n/a | 49.07% |
| Baseline: gradient-boosted trees, no sequence/embedding | 1.1163 | 51.03% |

## Key finding: no demonstrated directional edge

The pooled LSTM's directional accuracy (50.91%) is statistically indistinguishable
from a coin flip, and from both a naive persistence baseline (49.07%) and a
much simpler tabular gradient-boosting model with no temporal window and no
learned stock identity (51.03%). Its MSE is only marginally better than
always predicting zero return.

This is consistent with the weak-form efficient market hypothesis: next-day
returns derived from price/volume-based technical indicators (returns, RSI,
MACD, rolling volatility, market return) carry very little information about
the following day's direction for liquid, heavily-traded large-cap equities.
The model is not underfit or undertrained in a way more epochs or a larger
network would fix — train and validation loss converge to essentially the
same value almost immediately, indicating the input features do not contain
much learnable next-day signal in the first place, not that the model failed
to extract signal that was there.

This project's value is in the engineering — a correctly-built, leakage-free,
multi-stock pooled training pipeline with weight sharing via embeddings — not
in a claim that the resulting model is a profitable trading signal. It is not.

## Architectural limitations

1. **No explicit cross-stock correlation modeling.** Each stock's embedding is
   learned independently; the model has no mechanism to use sector co-movement,
   pairs relationships, or supply-chain/competitor linkages between stocks
   when making a prediction for any single stock.
2. **Point-in-time feature set is thin.** Only 5 engineered features per
   timestep (return, RSI, MACD, volatility, market return). No order-flow,
   options-derived, fundamental, or alternative data — the categories of data
   that actually carry incremental signal in modern quantitative equity
   research.
3. **Single global scaler across all stocks.** `StandardScaler` is fit once
   on pooled training rows across all 98 tickers, so a high-volatility stock
   (e.g. semiconductor names) and a low-volatility stock (e.g. a utility) are
   normalized against the same distribution, which can distort what a given
   scaled deviation actually represents for either one.
4. **Daily horizon is close to the noise floor.** Next-day return is one of
   the hardest possible prediction targets in finance — signal-to-noise ratio
   is far better at longer horizons (weekly/monthly) or for less efficiently
   priced instruments.
5. **No transaction costs, slippage, or position sizing modeled.** Even a
   model with genuine small directional edge is not automatically a viable
   trading strategy without accounting for these.

## Proposed extensions

- **GNN + LSTM hybrid**: model stocks as nodes in a graph (sector membership,
  historical correlation, or supply-chain edges), use a graph neural network
  layer to produce a context vector per stock per day that incorporates
  neighboring stocks' recent behavior, then feed that alongside the existing
  per-stock LSTM output.
- **Multivariate cross-stock input**: instead of one stock per training
  sequence, feed the model a snapshot of all (or a correlated subset of)
  stocks' features at each timestep, letting attention or pooling layers
  learn cross-stock interactions directly.
- **Per-sector or per-stock scaling** instead of one global scaler.
- **Longer forecast horizons** (5-day, 20-day forward return) where technical
  signals have historically shown a somewhat better signal-to-noise ratio.
- **Richer feature set**: order-flow/volume-imbalance features, options
  implied volatility, earnings-calendar proximity, and cross-sectional rank
  features (e.g. this stock's return rank among its sector today).
