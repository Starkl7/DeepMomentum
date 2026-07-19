# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class Phase1TraditionalMomentum(QCAlgorithm):
    """
    Phase 1: Traditional Cross-Sectional Momentum (12-1 month)
    Universe  : Russell 2000 via IWM ETF constituents
    Signal    : Cumulative return from T-13mo to T-1mo (skip most-recent month)
    Classes   : Top 10% = Class H (long), Bottom 10% = Class L (short)
    Portfolio : Equal-weighted, dollar-neutral (50% long / 50% short of NAV)
    Rebalance : Monthly, first trading day of month, fills at next open
    Diagnostics: Bimodality Coefficient on 12-1 return cross-section each month
    """

    # ── momentum parameters ──────────────────────────────────────────────────
    _MOM_PERIOD  = 252   # ~12 months of trading days
    _SKIP_PERIOD = 21    # ~1 month reversal-skip
    _LOOKBACK    = 273   # _MOM_PERIOD + _SKIP_PERIOD
    _QUANTILE              = 0.10  # top/bottom decile
    # 13 bars-warmup months (window fills) + 24 training months = 37 total,
    # matching Phase 2's combined window warmup + model training delay.
    _WARMUP_REBALANCES     = 24

    # ── universe filters ─────────────────────────────────────────────────────
    _MIN_ADV      = 1_000_000   # 30-day avg daily $ volume (lowered for Russell 2000)
    _MIN_PRICE    = 5.0
    _EXCL_SECTORS = frozenset({})   # Financials, Utilities

    # ─────────────────────────────────────────────────────────────────────────

    def Initialize(self):
        self.SetStartDate(2009, 8, 1)   # IWM data available from ~Nov 2010; 24-month warmup → live from Oct 2012
        self.SetEndDate(2024, 12, 31)
        self.SetCash(10_000_000)

        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Adjusted

        # Anchor equities: SPY for scheduling, IWM as universe source + benchmark
        # IWM = iShares Russell 2000 ETF (inception May 2000)
        # ETFConstituentsUniverse for IWM requires QC cloud — not in local LEAN sample data
        spy = self.AddEquity("SPY", Resolution.Daily)
        iwm = self.AddEquity("IWM", Resolution.Daily)
        self.SetBenchmark("IWM")
        self._anchor_syms = {spy.Symbol, iwm.Symbol}

        self.AddUniverse(self.Universe.ETF("IWM", self.UniverseSettings, self._ETFFilter))

        # Per-symbol rolling bar windows
        self._bars = {}      # Symbol -> RollingWindow[TradeBar]

        # Diagnostic state
        self._prev_h  = []   # Class H symbols from prior rebalance (for forward BC)
        self._bc_log      = []   # BC of the 12-1 signal distribution (diagnostic only)
        self._fwd_bc_log  = []   # BC of Class H 1-month forward returns (the meaningful metric)

        # Warmup gate: mirror Phase 2's 24 valid-rebalance training barrier so
        # both phases trade over exactly the same calendar period.
        self._warmup_count = 0

        # DateRules.MonthStart("SPY") fires on the first TRADING day of each month for
        # SPY's calendar — it automatically skips weekends and all market holidays
        # (including the New Year's Day observance that causes MonthStart() without a
        # symbol to silently drop every January and ~30% of other months).
        # Source: https://www.quantconnect.com/docs/v2/writing-algorithms/scheduled-events
        self.Schedule.On(
            self.DateRules.MonthStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self._Rebalance
        )

    # ── universe ──────────────────────────────────────────────────────────────

    def _ETFFilter(self, constituents):
        """Accept all IWM constituents; liquidity filters applied at rebalance."""
        return [c.Symbol for c in constituents]

    def OnSecuritiesChanged(self, changes):
        for s in changes.AddedSecurities:
            sym = s.Symbol
            if sym not in self._anchor_syms and sym not in self._bars:
                self._bars[sym] = RollingWindow[TradeBar](self._LOOKBACK + 10)
        for s in changes.RemovedSecurities:
            self._bars.pop(s.Symbol, None)

    # ── data ──────────────────────────────────────────────────────────────────

    def OnData(self, data: Slice):
        for sym, window in self._bars.items():
            if data.Bars.ContainsKey(sym):
                window.Add(data.Bars[sym])

    # ── rebalance ─────────────────────────────────────────────────────────────

    def _Rebalance(self):
        eligible = self._ComputeMomentum()
        n = len(eligible)

        if n < 20:
            self.Log(f"{self.Time:%Y-%m}: {n} eligible stocks — skipping rebalance")
            return

        # Count only valid rebalances (n >= 20).  Skip the first _WARMUP_REBALANCES
        # so Phase 1 starts trading at the same calendar point as Phase 2 (which
        # must accumulate 24 labeled training months before its first trade).
        self._warmup_count += 1
        if self._warmup_count <= self._WARMUP_REBALANCES:
            self.Log(
                f"{self.Time:%Y-%m}: warming up "
                f"({self._warmup_count}/{self._WARMUP_REBALANCES} valid rebalances) — "
                f"n={n}"
            )
            return

        cutoff = max(1, int(n * self._QUANTILE))
        eligible.sort(key=lambda x: x[1])   # ascending by 12-1 return

        class_l = eligible[:cutoff]          # bottom decile → short
        class_h = eligible[-cutoff:]         # top decile    → long

        targets = self._TargetWeights(class_h, class_l, n)

        # Exit positions no longer in the portfolio
        for sym in list(self.Portfolio.Keys):
            if self.Portfolio[sym].Invested and sym not in targets:
                self.Liquidate(sym)

        # Rebalance to rank-weighted, dollar-neutral targets
        for sym, weight in targets.items():
            self.SetHoldings(sym, weight)

        # ── diagnostics ──────────────────────────────────────────────────────
        all_mom  = [x[1] for x in eligible]
        bc       = self._BimodalityCoeff(all_mom)
        fwd_bc   = self._ForwardBC()

        self._prev_h = [sym for sym, _ in class_h]
        self._bc_log.append((self.Time, bc))
        self._fwd_bc_log.append((self.Time, fwd_bc))

        h_avg   = float(np.mean([x[1] for x in class_h]))
        l_avg   = float(np.mean([x[1] for x in class_l]))
        spread  = h_avg - l_avg

        long_ws  = [w for w in targets.values() if w > 0]
        short_ws = [abs(w) for w in targets.values() if w < 0]
        w_min    = min(long_ws)  if long_ws  else 0.0
        w_max    = max(long_ws)  if long_ws  else 0.0
        w_ratio  = w_max / w_min if w_min > 1e-8 else 0.0   # max/min tilt from vol+rank

        self.Log(
            f"{self.Time:%Y-%m}  n={n:4d}  H={len(class_h)}  L={len(class_l)}"
            f"  spread={spread:+.3f}"
            f"  w_min={w_min:.4f}  w_max={w_max:.4f}  w_ratio={w_ratio:.2f}x"
            f"  signal_bc={bc:.4f}"
            f"  fwd_bc={fwd_bc:.4f}"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _TargetWeights(self, class_h, class_l, n_total):
        """
        Equal-weighted, dollar-neutral.
        Each long = +0.5 / len(class_h), each short = -0.5 / len(class_l).
        """
        cutoff = len(class_h)
        w_long  =  0.5 / cutoff
        w_short = -0.5 / cutoff
        weights = {sym: w_long  for sym, _ in class_h}
        weights.update({sym: w_short for sym, _ in class_l})
        return weights

    def _ComputeMomentum(self):
        """
        Return list of (symbol, 12-1_return) for all eligible stocks.
        Eligibility: >= _LOOKBACK bars, price >= _MIN_PRICE, 30-day ADV >= _MIN_ADV.
        """
        results = []

        for sym, window in self._bars.items():
            if not window.IsReady:
                continue
            bars = list(window)
            if len(bars) < self._LOOKBACK:
                continue

            cur_price = bars[0].Close
            if cur_price < self._MIN_PRICE:
                continue

            avg_dv = float(np.mean([b.Close * b.Volume for b in bars[:30]]))
            if avg_dv < self._MIN_ADV:
                continue

            # Sector filter
            if sym in self.Securities:
                fund = self.Securities[sym].Fundamentals
                if fund is not None:
                    if fund.AssetClassification.MorningstarSectorCode in self._EXCL_SECTORS:
                        continue

            # 12-1 momentum: return from ~13 months ago to ~1 month ago
            p_skip = bars[self._SKIP_PERIOD].Close    # close ~1 month ago
            p_base = bars[self._LOOKBACK - 1].Close   # close ~13 months ago
            if p_base <= 0:
                continue
            mom = p_skip / p_base - 1.0

            results.append((sym, mom))

        return results

    def _ForwardBC(self):
        """
        Bimodality Coefficient of realized 1-month returns for the prior rebalance's
        Class H — forward-looking diagnostic for whether winners exhibit bimodal outcomes.
        """
        if not self._prev_h:
            return 0.0
        fwd_rets = []
        for sym in self._prev_h:
            if sym in self._bars and self._bars[sym].IsReady:
                bars = list(self._bars[sym])
                if len(bars) > self._SKIP_PERIOD:
                    fwd_rets.append(bars[0].Close / bars[self._SKIP_PERIOD].Close - 1.0)
        return self._BimodalityCoeff(fwd_rets) if len(fwd_rets) >= 4 else 0.0

    def _BimodalityCoeff(self, returns):
        """
        Sarle's Bimodality Coefficient (SAS Institute, 1990; Pfister et al., 2013).
        BC = (skew² + 1) / (excess_kurtosis + 3(n-1)²/((n-2)(n-3)))
        BC > 0.555 indicates bimodality (uniform distribution threshold).
        """
        n = len(returns)
        if n < 4:
            return 0.0
        arr  = np.array(returns, dtype=float)
        std  = float(arr.std())
        if std < 1e-10:
            return 0.0
        z    = (arr - arr.mean()) / std
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4)) - 3.0      # excess kurtosis
        denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        if abs(denom) < 1e-10:
            return 0.0
        return (skew ** 2 + 1.0) / denom

    # ── end of algorithm ──────────────────────────────────────────────────────

    def OnEndOfAlgorithm(self):
        if not self._fwd_bc_log:
            return
        self.Log("=" * 50)
        self.Log("PHASE 1 — BIMODALITY DIAGNOSTICS")

        # ── Forward BC (meaningful) ───────────────────────────────────────────
        # BC of Class H 1-month forward returns: tests whether momentum winners
        # exhibit bimodal outcomes (continue winning OR crash). BC > 0.555 = bimodal.
        fwd_bcs = [bc for _, bc in self._fwd_bc_log if bc > 0]
        live_rebalances = max(0, self._warmup_count - self._WARMUP_REBALANCES)
        self.Log(f"  Warmup rebalances skipped: {self._WARMUP_REBALANCES}")
        self.Log(f"  Live rebalances          : {live_rebalances}")
        self.Log(f"  [FORWARD RETURNS — Class H, meaningful bimodality test]")
        self.Log(f"  Mean fwd_BC              : {np.mean(fwd_bcs):.4f}  (threshold 0.555)")
        self.Log(f"  Pct months fwd_BC>0.555  : {np.mean([b > 0.555 for b in fwd_bcs]):.1%}")
        self.Log(f"  Min / Max fwd_BC         : {min(fwd_bcs):.4f} / {max(fwd_bcs):.4f}")
        self.Log(f"  Bimodality present       : {'YES — proceed to Phase 2' if np.mean(fwd_bcs) > 0.555 else 'NO — distribution is unimodal'}")

        # ── Signal BC (reference only) ────────────────────────────────────────
        # BC of the 12-1 momentum signal cross-section. This is ~normal across
        # 500+ stocks, so BC ≈ 0.30 always. Not a bimodality test — for reference only.
        sig_bcs = [bc for _, bc in self._bc_log]
        self.Log(f"  [SIGNAL DIST — 12-1 return cross-section, reference only]")
        self.Log(f"  Mean signal_BC           : {np.mean(sig_bcs):.4f}  (expect ~0.30 for normal dist)")

        self.Log("=" * 50)
