# DeepMomentum

Quantitative long-short equity strategy on the Russell 2000, developed
progressively from a classical momentum baseline through ML-enhanced
reclassification.

Inspired by Han & Qin [2023].

**Universe:** Russell 2000 via IWM ETF constituents  
**Capital:** $10 M starting equity  
**Rebalance:** Monthly, 30 min after open on first trading day of month  
**Walk-forward validation:** Oct 2012 – Dec 2024 (147 live rebalances)

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
| Warmup | 24 rebalances | Ensures identical live trading period with v1 |

### Walk-forward validation results (Oct 2012 – Dec 2024, 147 rebalances)

| Metric | Value |
|---|---|
| Compounding Annual Return | 3.498% |
| Total Return | 69.99% |
| Sharpe Ratio | 0.126 |
| Sortino Ratio | 0.114 |
| Max Drawdown | 24.900% |
| Annual Std Dev | 0.091 |
| Beta (vs IWM) | −0.066 |
| Alpha | 0.017 |
| Win Rate / Loss Rate | 49% / 51% |
| Profit-Loss Ratio | 1.12 |
| Portfolio Turnover | 1.59% |
| Total Fees | $345,583 |
| Total Orders | 23,924 |

### Bimodality diagnostics

We tracked the Bimodality Coefficient (BC) [Han & Qin, 2023] of the
forward-return distribution across all 147 live rebalances:

| Metric | Value |
|---|---|
| Mean fwd_BC | 0.3183 (threshold 0.555) |
| Pct months fwd_BC > 0.555 | 3.4% |
| Min / Max fwd_BC | 0.1517 / 0.8769 |
| Bimodality present | NO — unimodal on average |
| Mean signal_BC (12-1 return dist) | 0.4140 |

Bimodality is episodic rather than persistent: unimodal on average across the
full walk-forward period, but appearing in ~3% of months (peak BC = 0.8769,
Sep 2020 COVID shock).

---

## v1 Model — XGBoost Reclassifier (`phase2_xgb.py`)

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

Long book = GW ∪ GL. Short book = BW ∪ BL. All four quadrants are always
traded — no quadrant is skipped.

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

### Walk-forward validation results (Oct 2012 – Dec 2024, 147 rebalances)

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

### Walk-forward validation diagnostics (Oct 2012 – Dec 2024, 147 rebalances)

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

## Performance Comparison

### Full walk-forward validation (Oct 2012 – Dec 2024, 147 rebalances)

| Metric | Baseline | v1 (XGBoost) | Δ |
|---|---|---|---|
| Compounding Annual Return | 3.498% | 4.082% | +0.584 pp |
| Sharpe Ratio | 0.126 | 0.235 | +0.109 |
| Sortino Ratio | 0.114 | 0.234 | +0.120 |
| Max Drawdown | 24.900% | 16.800% | −8.1 pp |
| Annual Std Dev | 0.091 | 0.054 | −0.037 |
| Beta (vs IWM) | −0.066 | −0.035 | +0.031 |
| Alpha | 0.017 | 0.015 | −0.002 |
| Portfolio Turnover | 1.59% | 2.57% | +0.98 pp |
| Win Rate | 49% | 52% | +3 pp |

v1 improves on all risk-adjusted metrics over the full walk-forward period. Sharpe
nearly doubles (+0.109), max drawdown falls by 8.1 pp, and annual volatility drops
by 40%. The CAGR improvement is modest (+0.58 pp) — the XGBoost layer adds more
value by reducing risk than by increasing raw returns.

### Sub-period breakdown (analytical)

| Metric | Baseline 2012–2020 | Baseline 2021–2024 | v1 2012–2020 | v1 2021–2024 |
|---|---|---|---|---|
| CAR | 1.563% | 5.503% | 4.272% | 3.339% |
| Sharpe | 0.033 | 0.179 | 0.418 | −0.042 |
| Max Drawdown | 23.10% | 15.200% | 10.90% | 13.600% |
| Annual Std Dev | 0.082 | 0.082 | 0.045 | 0.052 |
| Beta | −0.047 | −0.058 | −0.014 | −0.041 |
| Alpha | 0.007 | 0.017 | 0.020 | 0.000 |

The sub-period data explains the mechanism behind v1's full-period risk reduction.
The 2012–2020 period showed strong outperformance (Sharpe 0.418 vs 0.033), driven
largely by the GL reversal trade. The 2021–2024 period showed v1 underperforming
on Sharpe (−0.042 vs 0.179) despite maintaining an L/S spread of +0.85%/mo —
the underperformance came from net beta instability, not alpha collapse. Assessed
over the full walk-forward window, v1's structural volatility reduction (StdDev
0.054 vs 0.091) and drawdown reduction (16.8% vs 24.9%) dominate, producing a
nearly doubled Sharpe ratio.

---

## Project Structure

```
DeepMomentum/
├── main.py               # Baseline — traditional momentum (Phase 1)
├── phase2_benchmark.py   # v1 — XGBoost 4-quadrant reclassifier (Phase 2)
├── phase3_1.py           # v2 (in progress) — MDN direct
├── phase3_2.py           # v2 (in progress) — MDN 4-quadrant
├── config.json           # QuantConnect project config
├── research.ipynb        # Analysis notebook
└── README.md             # This file
```

---

## Running on QuantConnect

```bash
# Push and run on QC cloud
lean cloud push --project "DeepMomentum"
lean cloud backtest "DeepMomentum" --name "baseline-wfv"

# Each phase file is self-contained. To run a specific version, set it
# as the entry point in config.json (or copy/rename to main.py).
```

---

## Citations

**[Han & Qin, 2023]** Han, Y., & Qin, J. (2023). *Bimodality Everywhere:
International Evidence of Deep Momentum*. SSRN Working Paper.

**[Jegadeesh & Titman, 1993]** Jegadeesh, N., & Titman, S. (1993). Returns to
Buying Winners and Selling Losers: Implications for Stock Market Efficiency.
*Journal of Finance*, 48(1), 65–91.
