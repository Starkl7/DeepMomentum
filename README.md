# DeepMomentum

![Python](https://img.shields.io/badge/python-3.14-blue)
![Data](https://img.shields.io/badge/data-QuantConnect--ohlcv-orange)
![Status](https://img.shields.io/badge/status-research-lightgrey)

Quantitative long-short equity strategy on the Russell 2000, developed
progressively from a classical momentum baseline through ML-enhanced
reclassification.

Inspired by Han & Qin [2023].

**Universe:** Russell 2000 via IWM ETF constituents  
**Capital:** $10 M starting equity  
**Rebalance:** Monthly, 30 min after open on first trading day of month  
**Walk-forward validation:** Oct 2012 – Dec 2025 (159 live rebalances, LEAN v2.5.0.0.17868)

Features were selected a priori from theory and domain knowledge — no
parameters were optimized via backtesting. The full trading window is
therefore a genuine walk-forward validation of each strategy.

---

## Baseline — Traditional Momentum (`phase1_baseline.py`)

Pure Jegadeesh-Titman 12-1 month cross-sectional momentum [Jegadeesh & Titman, 1993] with no ML layer.
Serves as the benchmark against which all model improvements are measured.

### Signal construction

The momentum signal is the cumulative stock return from 12 months ago to 1
month ago — the 1-month skip prevents microstructure reversal from
contaminating the signal. Formally: `ret_12_1 = P[t-21] / P[t-273] - 1`.

Each month, all eligible stocks are ranked by this signal. The top decile
goes long, the bottom decile goes short. Equal-weighted, dollar-neutral
(50% long / 50% short of NAV).

### Universe filters

| Filter | Value | Rationale |
|---|---|---|
| Price | ≥ $5 | Exclude penny stocks with wide spreads |
| ADV | ≥ $1 M | Ensure sufficient liquidity to execute |
| Sector | None | Full Russell 2000 exposure |
| Bar window | 273 bars | 13 months of daily data required for signal |
| Warmup | 24 rebalances | Ensures identical live trading period with XGBoost model |

### Walk-forward validation results (Oct 2012 – Dec 2025, 159 rebalances)

| Metric | Value |
|---|---|
| Compounding Annual Return | 2.780% |
| Total Return | 56.91% |
| Sharpe Ratio | 0.046 |
| Sortino Ratio | 0.043 |
| Max Drawdown | 25.400% |
| Annual Std Dev | 0.092 |
| Beta (vs IWM) | −0.059 |
| Alpha | 0.009 |
| Win Rate / Loss Rate | 49% / 51% |
| Profit-Loss Ratio | 1.11 |
| Portfolio Turnover | 1.62% |
| Total Fees | $372,771 |
| Total Orders | 25,713 |

### Bimodality diagnostics

We tracked the Bimodality Coefficient (BC) [Han & Qin, 2023] of the
forward-return distribution across all 159 live rebalances (Oct 2012–Dec 2025):

| Metric | Value |
|---|---|
| Mean fwd_BC | 0.3235 (threshold 0.555) |
| Pct months fwd_BC > 0.555 | 3.8% |
| Min / Max fwd_BC | 0.1517 / 0.8769 |
| Bimodality present | NO — unimodal on average |
| Mean signal_BC (12-1 return dist) | 0.4272 |

Bimodality is episodic rather than persistent: unimodal on average across the
full walk-forward period, but appearing in ~4% of months (peak BC = 0.8769,
Sep 2020 COVID shock).

---

## v1 Model — XGBoost Reclassifier (pure, reference)

Extends the baseline with a 10-member XGBoost ensemble that reclassifies
momentum stocks into four quadrants. The core idea: momentum correctly
identifies *which* stocks are in extreme return regimes (winners and losers),
but within each group, not all stocks will continue in the same direction.
XGBoost predicts which stocks will outperform the cross-sectional median over
the next month, allowing us to reverse losing positions within each momentum class.

### Design choices

**Why four quadrants instead of a single signal?**  
Pure momentum long/short conflates two separate bets: (1) that winners keep
winning and losers keep losing, and (2) that *every* winner and loser will
continue on trend. The quadrant structure separates these. A "Good Loser"
(bottom-decile momentum stock predicted to outperform) is a contrarian reversal
trade — the mirror image of momentum. Including it in the long book diversifies
alpha sources and reduces drawdown.

**Why binary classification over ranking?**  
The label is `1` if a stock's forward 1-month return exceeds the cross-sectional
median that month — a relative outperformance signal, not absolute return.
This makes the training target regime-invariant: a bear-market winner is
labelled the same as a bull-market winner, which matters for a rolling
36-month training window that spans multiple regimes.

**Why cross-sectional ranking of features?**  
All 15 per-stock features are ranked to percentile [0, 1] each month before
model input. This neutralises cross-sectional distribution shifts (e.g.
P/E ratios mean-reverting over time) and ensures the model learns
relative ordering rather than absolute levels. The 4 market-regime features
(CS means of momentum returns) are appended *unranked* so the model retains
access to absolute market-level magnitude.

**Why 10-run ensemble?**  
A single XGBoost fit is sensitive to the random column/row subsampling drawn
at initialization. Averaging `predict_proba` over 10 independent fits (seeds
17–26) produces smoother probability estimates and reduces month-to-month
noise in quadrant assignments, lowering unnecessary turnover.

### Feature set

| Group | Features | Count |
|---|---|---|
| Return (ranked) | ret_1m, ret_3m, ret_6m, ret_12_1 | 4 |
| Volatility (ranked) | vol_12m | 1 |
| Valuation (ranked) | pe, pb, ev_ebitda, ps | 4 |
| Profitability (ranked) | roe, roa, gross_margin | 3 |
| Growth (ranked) | rev_growth, net_income_growth | 2 |
| Size (ranked) | log_mcap | 1 |
| Market regime (raw) | cs_mean_ret_1m, cs_mean_ret_3m, cs_mean_ret_6m, cs_mean_ret_12_1 | 4 |
| **Total** | | **19** |

All per-stock features are cross-sectionally ranked to percentile [0, 1]
each month. The 4 CS mean features are appended unranked after ranking so
they preserve market-level magnitude.

### Four quadrants

| Quadrant | Momentum | XGB signal | Position |
|---|---|---|---|
| Good Winners (GW) | Top decile | p ≥ 0.50 | **Long** |
| Bad Winners (BW) | Top decile | p < 0.50 | **Short** |
| Good Losers (GL) | Bottom decile | p ≥ 0.50 | **Long** |
| Bad Losers (BL) | Bottom decile | p < 0.50 | **Short** |

Long book = GW ∪ GL. Short book = BW ∪ BL. In the pure v1 model all four
quadrants are always traded. The v2 hybrid adds a persistence gate that can
drop GL or BW when their realized returns indicate the cross-signal is failing
(see v2 section below).

### Training setup

| Parameter | Value |
|---|---|
| Model | XGBoost binary classifier (`binary:logistic`) |
| Ensemble | 10 runs, seeds 17–26 |
| n_estimators | 100 |
| max_depth | 3 |
| learning_rate | 0.05 |
| subsample / colsample_bytree | 0.8 / 0.8 |
| min_child_weight | 10 |
| Training window | Rolling 36-month buffer |
| Warmup | 24 labeled months before first trade |
| Retrain frequency | Monthly |

### Walk-forward validation results (Oct 2012 – Dec 2024, 147 rebalances, LEAN v2.5.0.0.17779)

Historical reference run. See Performance Comparison for current-engine equivalent.

| Metric | Value |
|---|---|
| Compounding Annual Return | 4.082% |
| Total Return | 85.39% |
| Sharpe Ratio | 0.235 |
| Sortino Ratio | 0.234 |
| Max Drawdown | 16.800% |
| Annual Std Dev | 0.054 |
| Beta (vs IWM) | −0.035 |
| Alpha | 0.015 |
| Win Rate / Loss Rate | 52% / 48% |
| Profit-Loss Ratio | 1.02 |
| Portfolio Turnover | 2.57% |
| Total Fees | $590,379 |
| Total Orders | 30,851 |

### Walk-forward validation diagnostics (Oct 2012 – Dec 2024, original run)

| Metric | Value | Notes |
|---|---|---|
| Prob spread (GW−GL) | +0.0004 | Near-zero — XGB acts as regime filter, not stock picker |
| Pct months positive spread | 43.5% | Near chance |
| Mean P(Good Winner) | 0.5257 | |
| Mean P(Good Loser) | 0.5253 | |
| GW avg monthly return | +1.66% | Longed |
| GL avg monthly return | +1.90% | Longed — reversal trade; largest alpha source |
| BW avg monthly return | +1.20% | Shorted |
| BL avg monthly return | +0.81% | Shorted |
| Avg long book (GW+GL)/2 | +1.78%/mo | |
| Avg short book (BW+BL)/2 | +1.00%/mo | |
| L/S spread | +0.78%/mo | |
| Mean net beta | −0.161 | Near-neutral |
| Net beta std dev | 0.308 | Moderate variance |
| Net beta range | −1.084 / +0.525 | |

**Top feature importances (full period):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | vol_12m | 0.1132 |
| 2 | roa | 0.0804 |
| 3 | cs_mean_ret_3m | 0.0799 |
| 4 | cs_mean_ret_12_1 | 0.0732 |
| 5 | cs_mean_ret_6m | 0.0604 |

The XGB signal is near-random across the full period (prob spread +0.0004),
functioning as a regime-sensitive filter rather than a precise stock picker.
The GL reversal trade (+1.90%/mo) is the largest single alpha source. Beta
leakage is moderate (std 0.308, range −1.084/+0.525), with `vol_12m` the
dominant feature — suggesting the model primarily learns to adjust exposure
based on cross-sectional volatility conditions.

---

## v2 Model — XGBoost + Hybrid Enhancements (`phase2_xgb.py`, current)

Extends v1 with three risk-management overlays that improve Sharpe and reduce
drawdown without modifying the core XGBoost signal. These are applied as
post-signal portfolio construction rules — the XGBoost model is unchanged.

### Enhancements

**1. GL / BW leg persistence gate**  
At each rebalance, the GL leg (long losers expected to reverse) and BW leg (short
winners expected to fade) are individually gated on a rolling 2-month realized
return filter. If the mean realized return of the GL book over the past 2 months
is negative, the GL leg is dropped and capital concentrates in GW. Similarly, if
the BW realized return is negative (our shorts are going up — a signal the
contrarian bet is failing), BW is dropped and capital concentrates in BL.

This concentrates risk in the two high-conviction momentum legs (GW, BL) during
regimes where the XGB cross-signal is not delivering.

**2. Volatility scaling (Barroso & Santa-Clara 2015)**  
Position sizes are scaled monthly by `target_vol / realized_vol_21d`, capped at
1.2× (or 1.5× in strong bull markets). This targets a constant 10% annualized
portfolio volatility, scaling down exposure in high-vol regimes and moderately
levering up in quiet periods. Prevents momentum crashes from catching the
portfolio fully deployed.

**3. Market state gate (Cooper et al. 2004)**  
Uses the IWM 12-month return as a market regime signal. When IWM 12m return < 0
(DOWN regime): vol scaling is capped at 1.0× (no leverage) and all position
weights are halved (0.5× exposure). When IWM 12m > +20% (strong bull): the vol
scaling cap is raised to 1.5×. Addresses the finding that momentum strategies
generate nearly all their alpha in UP-market states.

### Walk-forward validation results (Oct 2012 – Dec 2025, 159 rebalances)

| Metric | Value |
|---|---|
| Compounding Annual Return | 3.859% |
| Total Return | 86.27% |
| Sharpe Ratio | 0.143 |
| Sortino Ratio | 0.144 |
| Max Drawdown | 19.900% |
| Annual Std Dev | 0.067 |
| Beta (vs IWM) | −0.053 |
| Alpha | 0.014 |
| Win Rate / Loss Rate | 51% / 49% |
| Profit-Loss Ratio | 1.06 |
| Portfolio Turnover | 3.05% |
| Total Fees | $734,770 |
| Total Orders | 33,585 |

---

## Performance Comparison

### Walk-forward validation comparison

All results on LEAN v2.5.0.0.17868, Oct 2012 – Dec 2025 (159 live rebalances).
v1 XGBoost pure results are also provided on this engine for reference.

| Metric | Baseline | v1 XGBoost (pure) | v2 Hybrid (final) | Δ (Baseline→v2) |
|---|---|---|---|---|
| Compounding Annual Return | 2.780% | 3.049% | **3.859%** | +1.079 pp |
| Sharpe Ratio | 0.046 | 0.104 | **0.143** | +0.097 |
| Sortino Ratio | 0.043 | 0.106 | **0.144** | +0.101 |
| Max Drawdown | 25.400% | 24.200% | **19.900%** | −5.5 pp |
| Annual Std Dev | 0.092 | 0.053 | **0.067** | −0.025 |
| Beta (vs IWM) | −0.059 | −0.031 | **−0.053** | +0.006 |
| Alpha | 0.009 | 0.008 | **0.014** | +0.005 |
| Win Rate | 49% | 52% | **51%** | +2 pp |
| Total Fees | $372,771 | $558,575 | $734,770 | — |

v2 Hybrid outperforms both the baseline and pure XGBoost on all risk-adjusted metrics.
The XGBoost reclassification (v1) alone improves CAR and Sharpe modestly; the hybrid
enhancements (v2) further improve Sharpe (+0.039 over pure XGB) while cutting drawdown
by 4.3 pp. The Sharpe/DD ratio: Baseline 0.18, v1 XGB 0.43, v2 Hybrid **0.72**.

---

## Project Structure

```
DeepMomentum/
├── phase1_baseline.py    # Phase 1 — traditional momentum (baseline)
├── phase2_xgb.py         # Phase 2 — XGBoost + hybrid enhancements (current, end date 2025)
├── config.json           # QuantConnect project config (gitignored)
└── README.md             # This file
```

---

## Running on QuantConnect

QC requires `main.py` as the algorithm entry point. Generate it from the source
file before each push, then launch the backtest:

```bash
# Phase 2 v2 (current strategy)
cp DeepMomentum/phase2_xgb.py DeepMomentum/main.py
.venv/bin/lean cloud backtest <project-id> --push --name "Phase2-run"

# Phase 1 baseline
cp DeepMomentum/phase1_baseline.py DeepMomentum/main.py
.venv/bin/lean cloud backtest <project-id> --push --name "Phase1-run"
```

`main.py` is gitignored — only the named source files are tracked.

---

## Citations

**[Jegadeesh & Titman, 1993]** Jegadeesh, N., & Titman, S. (1993). Returns to
Buying Winners and Selling Losers: Implications for Stock Market Efficiency.
*Journal of Finance*, 48(1), 65–91.  
*Basis for the 12-1 month cross-sectional momentum signal.*

**[Han & Qin, 2023]** Han, Y., & Qin, J. (2023). *Bimodality Everywhere:
International Evidence of Deep Momentum*. SSRN Working Paper.  
*Bimodality Coefficient diagnostic used to characterise the momentum return distribution.*

**[Barroso & Santa-Clara, 2015]** Barroso, P., & Santa-Clara, P. (2015). Momentum
Has Its Moments. *Journal of Financial Economics*, 116(1), 111–120.  
*Volatility scaling methodology: target constant realized vol, scale positions by
`target_vol / realized_vol_21d`. Directly implemented in v2 hybrid.*

**[Cooper, Gutierrez & Hameed, 2004]** Cooper, M. J., Gutierrez, R. C., & Hameed, A.
(2004). Market States and Momentum. *Journal of Finance*, 59(3), 1345–1365.  
*Market state gate: IWM 12-month return as UP/DOWN regime signal. Momentum profits
concentrate in UP-market states; strategy halves exposure in DOWN regimes.*

**[Daniel & Moskowitz, 2016]** Daniel, K., & Moskowitz, T. J. (2016). Momentum
Crashes. *Journal of Financial Economics*, 122(2), 221–247.  
*Momentum crash risk: crashes cluster in high-vol, post-bear-market periods.
Motivates the asymmetric lever cap (no leverage in DOWN markets) in v2.*
