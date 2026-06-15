# region imports
from AlgorithmImports import *
from scipy.stats import rankdata
import numpy as np
import xgboost as xgb
from collections import deque
# endregion

class Phase2XGBoostMomentum(QCAlgorithm):
    """
    Phase 2: XGBoost Momentum Reclassifier

    Extends Phase 1 (12-1 month momentum) with 19 cross-sectional features to
    reclassify momentum stocks into four quadrants based on XGB signal direction
    (p >= 0.50 = bullish / above-median return predicted):

      - Good Winners  (Class H + XGB bullish, p >= 0.50) -> Long
      - Bad  Winners  (Class H + XGB bearish, p <  0.50) -> Short
      - Good Losers   (Class L + XGB bullish, p >= 0.50) -> Long   # reversal
      - Bad  Losers   (Class L + XGB bearish, p <  0.50) -> Short

    Long book  = Good Winners u Good Losers  (XGB says above-median return)
    Short book = Bad Winners  u Bad Losers   (XGB says below-median return)
    All four quadrants are always traded -- no quadrant is skipped.

    Universe  : Russell 2000 via IWM ETF constituents (same as Phase 1)
    Features  : 22 cross-sectionally ranked factors (18 per-stock + 4 CS regime means)
    Training  : Rolling 36-month buffer, retrained monthly, min 24 months required
    Portfolio : Equal-weighted, dollar-neutral (50% long / 50% short of NAV)
    Rebalance : Monthly, 30 min after open on first trading day of month
    """

    # ── eligibility (same as Phase 1) ────────────────────────────────────────
    _MOM_PERIOD   = 252
    _SKIP_PERIOD  = 21
    _LOOKBACK     = 273     # _MOM_PERIOD + _SKIP_PERIOD
    _MAX_BARS     = _LOOKBACK + 10  # 283 bars — 10-bar buffer matches Phase 1; signal uses only _LOOKBACK
    _QUANTILE     = 0.10
    _MIN_ADV      = 1_000_000
    _EXCL_SECTORS = frozenset({})   # Financials, Utilities
    _MIN_PRICE    = 5.0


    # ── XGBoost / training ────────────────────────────────────────────────────
    _MIN_TRAIN_MONTHS  = 24   # min labeled months before first trade
    _TRAIN_BUF_MONTHS  = 36   # rolling training window size
    _PROB_THRESHOLD    = 0.50  # Good Winner: p >= 0.5 | Good Loser: p < 0.5
    _N_XGB_RUNS        = 10   # ensemble: average predict_proba over N independent fits
    _SEED              = 17   # master seed — ensemble uses _SEED … _SEED + _N_XGB_RUNS - 1
    _USE_HIGH_FEATURES = False # set False to drop p_52wh / p_12wh / h50_h200 everywhere

    # ── safe attribute accessor ───────────────────────────────────────────────
    @staticmethod
    def _F(obj, *attrs):
        """
        Chain getattr calls without raising; return np.nan on any miss or bad value.
        Usage: _F(fund.ValuationRatios, 'PERatio')
               _F(fund.OperationRatios, 'ROE', 'OneYear')
        """
        try:
            for a in attrs:
                obj = getattr(obj, a, None)
                if obj is None:
                    return np.nan
            v = float(obj)
            return v if np.isfinite(v) and v != 0 else np.nan
        except Exception:
            return np.nan

    # ── feature names (order must match _ExtractRaw) ──────────────────────────
    FEATURES = [
        # Per-stock cross-sectionally ranked features (indices 0–17)
        # ret_24m dropped: _MAX_BARS = 283 bars, which is < 504
        # bars needed for a 24m return — permanently NaN, so excluded.
        'ret_1m', 'ret_3m', 'ret_6m', 'ret_12_1', 'vol_12m',
        'pe',     'pb',     'ev_ebitda',
        'roe',    'roa',    'gross_margin', 'rev_growth', 'net_income_growth',
        'ps',     'log_mcap',
        # Distance-from-high features (George-Hwang 52wh factor + variants)
        'p_52wh',   # current close / 52-week rolling-high close  (bounded (0, 1])
        'p_12wh',   # current close / 12-week rolling-high close  (bounded (0, 1])
        'h50_h200', # 50-day high   / 200-day high                (bounded (0, 1])
        # Cross-sectional means of momentum features (indices 18–21)
        'cs_mean_ret_1m', 'cs_mean_ret_3m', 'cs_mean_ret_6m', 'cs_mean_ret_12_1',
    ]
    N_FEAT     = len(FEATURES)   # 22
    N_FEAT_RAW = 18              # columns in X_raw (before means are appended)

    # ─────────────────────────────────────────────────────────────────────────

    def Initialize(self):
        np.random.seed(self._SEED)   # global numpy seed for reproducibility

        # Trim instance-level feature metadata when high features are disabled.
        # _CSRank and _BuildSnapshot use X_raw.shape[1] dynamically, so no
        # changes needed there — only FEATURES / N_FEAT / N_FEAT_RAW need adjusting.
        if not self._USE_HIGH_FEATURES:
            _HIGH = {'p_52wh', 'p_12wh', 'h50_h200'}
            self.FEATURES   = [f for f in self.FEATURES if f not in _HIGH]
            self.N_FEAT     = len(self.FEATURES)   # 19
            self.N_FEAT_RAW = 15

        self.SetStartDate(2009, 8, 1)   # IWM data available from ~Nov 2010; 24-month warmup → live from Oct 2012
        self.SetEndDate(2024, 12, 31)
        self.SetCash(10_000_000)

        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Adjusted

        spy = self.AddEquity("SPY", Resolution.Daily)
        iwm = self.AddEquity("IWM", Resolution.Daily)
        self.SetBenchmark("IWM")
        self._anchor_syms = {spy.Symbol, iwm.Symbol}

        self.AddUniverse(self.Universe.ETF("IWM", self.UniverseSettings, self._ETFFilter))

        # Per-symbol rolling bar windows (larger than Phase 1 for 24m feature)
        self._bars = {}

        # Training pipeline state
        self._snap_buf  = deque(maxlen=self._TRAIN_BUF_MONTHS)  # X arrays
        self._label_buf = deque(maxlen=self._TRAIN_BUF_MONTHS)  # y arrays
        self._prev_snap = None   # snapshot from previous rebalance
        self._models    = []     # list of trained XGBClassifiers (ensemble)

        # Diagnostics
        self._rebalance_log = []   # (date, n_gw, n_gl)
        self._diag_log      = []   # monthly diagnostic dicts
        self._prev_quads    = None  # {sym: (quad_label, prev_price)} for realized-return tracking

        # SPY rolling window for beta computation
        self._spy_bars = RollingWindow[TradeBar](63)

        self.Schedule.On(
            self.DateRules.MonthStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self._Rebalance
        )

    # ── universe ──────────────────────────────────────────────────────────────

    def _ETFFilter(self, constituents):
        return [c.Symbol for c in constituents]

    def OnSecuritiesChanged(self, changes):
        for s in changes.AddedSecurities:
            sym = s.Symbol
            if sym not in self._anchor_syms and sym not in self._bars:
                self._bars[sym] = RollingWindow[TradeBar](self._MAX_BARS)
        for s in changes.RemovedSecurities:
            self._bars.pop(s.Symbol, None)

    def OnData(self, data: Slice):
        for sym, window in self._bars.items():
            if data.Bars.ContainsKey(sym):
                window.Add(data.Bars[sym])
        # SPY for beta computation
        for sym in self._anchor_syms:
            if sym.Value == "SPY" and data.Bars.ContainsKey(sym):
                self._spy_bars.Add(data.Bars[sym])

    # ── monthly rebalance ─────────────────────────────────────────────────────

    def _Rebalance(self):
        # 1. Build feature snapshot for all eligible stocks
        snap = self._BuildSnapshot()
        if snap is None or len(snap['syms']) < 20:
            self.Log(f"{self.Time:%Y-%m}: <20 eligible — skip")
            self._prev_snap = snap
            return

        # 2. Use prev snapshot + current prices to create labeled training rows
        if self._prev_snap is not None:
            self._AddTrainingRows(self._prev_snap, snap)

        # 3. Retrain if enough history
        if len(self._label_buf) >= self._MIN_TRAIN_MONTHS:
            self._TrainXGB()

        # Store for next month's label computation
        self._prev_snap = snap

        # 4. Skip trading during warm-up
        if not self._models:
            self.Log(f"{self.Time:%Y-%m}: warming up ({len(self._label_buf)}/{self._MIN_TRAIN_MONTHS} mo)")
            return

        # 5. Momentum classes → XGB reclassification → portfolio
        h_idx, l_idx = self._MomentumClasses(snap)
        long_idx, short_idx, probs, quads = self._XGBClassify(snap, h_idx, l_idx)

        if not long_idx or not short_idx:
            self.Log(f"{self.Time:%Y-%m}: empty long or short book — skip")
            self._prev_quads = None
            return

        targets = self._TargetWeights(snap, long_idx, short_idx)

        # Exit stale positions
        for sym in list(self.Portfolio.Keys):
            if self.Portfolio[sym].Invested and sym not in targets:
                self.Liquidate(sym)

        # Enter / resize
        for sym, w in targets.items():
            self.SetHoldings(sym, w)

        # ── diagnostics ──────────────────────────────────────────────────────
        self._LogDiagnostics(snap, h_idx, l_idx, quads['gw'], quads['gl'], probs, targets)

        n_gw, n_gl = len(quads['gw']), len(quads['gl'])
        n_bw, n_bl = len(quads['bw']), len(quads['bl'])
        self._rebalance_log.append((self.Time, n_gw + n_gl, n_bw + n_bl))
        self.Log(
            f"{self.Time:%Y-%m}  "
            f"LONG={len(long_idx)} (GW={n_gw} GL={n_gl})  "
            f"SHORT={len(short_idx)} (BW={n_bw} BL={n_bl})  "
            f"train={len(self._label_buf)}mo"
        )

        # Store all four quadrant assignments for next month's realized-return check
        self._prev_quads = {}
        syms, prices = snap['syms'], snap['prices']
        for i in quads['gw']: self._prev_quads[syms[i]] = ('GW', float(prices[i]))
        for i in quads['gl']: self._prev_quads[syms[i]] = ('GL', float(prices[i]))
        for i in quads['bw']: self._prev_quads[syms[i]] = ('BW', float(prices[i]))
        for i in quads['bl']: self._prev_quads[syms[i]] = ('BL', float(prices[i]))

    # ── snapshot (feature extraction + cross-sectional ranking) ───────────────

    def _BuildSnapshot(self):
        """Return dict: syms, X (ranked), ret_12_1, mcap, prices."""
        raw_rows, syms = [], []

        for sym, window in self._bars.items():
            if not window.IsReady:
                continue
            bars = list(window)
            if len(bars) < self._LOOKBACK:
                continue

            cur = bars[0].Close
            if cur < self._MIN_PRICE:
                continue
            adv = float(np.mean([b.Close * b.Volume for b in bars[:30]]))
            if adv < self._MIN_ADV:
                continue

            # Sector filter (graceful if fundamentals unavailable)
            if sym in self.Securities:
                fund = self.Securities[sym].Fundamentals
                if fund is not None:
                    if fund.AssetClassification.MorningstarSectorCode in self._EXCL_SECTORS:
                        continue

            row = self._ExtractRaw(sym, bars)
            if row is None:
                continue
            syms.append(sym)
            raw_rows.append(row)

        if len(syms) < 20:
            return None

        X_raw = np.array([r['feat'] for r in raw_rows], dtype=float)
        X     = self._CSRank(X_raw)   # (n, 18) per-stock percentile ranks

        # Cross-sectional means of the 4 momentum features (ret_1m … ret_12_1).
        # ret_24m excluded: window too small (_MAX_BARS = 283 < 504 bars required).
        # These are identical for every stock in a given month — they give the
        # model a market-level context signal (bull/bear regime awareness).
        # Appended AFTER _CSRank so they are not collapsed to rank 0.5.
        cs_means   = np.nanmean(X_raw[:, :4], axis=0)       # (4,)
        mean_block = np.tile(cs_means, (len(raw_rows), 1))  # (n, 4)
        X          = np.hstack([X, mean_block])              # (n, 22)

        return {
            'syms':    syms,
            'X':       X,
            'X_raw':   X_raw,   # unranked 15-col — used for NaN-rate diagnostics
            'mom':     np.array([r['mom'] for r in raw_rows]),
            'mcap':    np.array([r['mcap'] for r in raw_rows]),
            'prices':  np.array([r['price'] for r in raw_rows]),
        }

    def _ExtractRaw(self, sym, bars):
        """
        Extract the 15 raw features for one stock.
        Returns None if the momentum signal itself is invalid.
        eps_growth replaced with net_income_growth (NetIncomeGrowth.OneYear) —
        EarningsGrowth attribute does not exist in QC's OperationRatios API.
        """
        n = len(bars)

        def ret(i, j):
            pi, pj = bars[i].Close, bars[j].Close
            return pi / pj - 1.0 if pj > 0 else np.nan

        ret_1m   = ret(0, 21)
        ret_3m   = ret(0, 63)   if n > 63  else np.nan
        ret_6m   = ret(0, 126)  if n > 126 else np.nan
        ret_12_1 = ret(self._SKIP_PERIOD, self._LOOKBACK - 1)
        # ret_24m removed: _MAX_BARS = 283 < 504 bars required.

        if np.isnan(ret_12_1):
            return None

        # 12m realized vol from daily price-pct-returns
        end = min(253, n)
        prices_arr = np.array([bars[i].Close for i in range(end - 1, -1, -1)])
        pct_rets   = np.diff(prices_arr) / np.maximum(prices_arr[:-1], 1e-8)
        vol_12m    = float(np.std(pct_rets) * np.sqrt(252)) if len(pct_rets) > 10 else np.nan

        # Fundamental features — _F() silently returns np.nan for missing/zero attrs
        pe = pb = ev_ebitda = roe = roa = gross_margin = np.nan
        rev_growth = net_income_growth = ps = np.nan
        mcap = bars[0].Close   # price fallback if shares unavailable

        if sym in self.Securities:
            fund = self.Securities[sym].Fundamentals
            if fund is not None:
                vr  = fund.ValuationRatios
                or_ = fund.OperationRatios

                pe        = self._F(vr,  'PERatio')
                pb        = self._F(vr,  'PBRatio')
                ev_ebitda = self._F(vr,  'EVToEBITDA')
                ps        = self._F(vr,  'PSRatio')

                roe              = self._F(or_, 'ROE',             'OneYear')
                roa              = self._F(or_, 'ROA',             'OneYear')
                gross_margin     = self._F(or_, 'GrossMargin',     'OneYear')
                rev_growth       = self._F(or_, 'RevenueGrowth',   'OneYear')
                net_income_growth = self._F(or_, 'NetIncomeGrowth', 'OneYear')

                shares = self._F(fund, 'CompanyProfile', 'SharesOutstanding')
                if np.isfinite(shares) and shares > 0:
                    mcap = bars[0].Close * shares

        log_mcap = float(np.log(mcap)) if mcap > 0 else np.nan

        feat = [
            ret_1m, ret_3m, ret_6m, ret_12_1, vol_12m,
            pe, pb, ev_ebitda,
            roe, roa, gross_margin, rev_growth, net_income_growth,
            ps, log_mcap,
        ]
        if self._USE_HIGH_FEATURES:
            # n >= _LOOKBACK = 273 is guaranteed by the caller, so 252/200/63/50 are safe.
            cur_close = bars[0].Close
            high_52w  = max(b.Close for b in bars[:252])
            high_12w  = max(b.Close for b in bars[:63])
            high_50d  = max(b.Close for b in bars[:50])
            high_200d = max(b.Close for b in bars[:200])
            feat += [
                cur_close / high_52w  if high_52w  > 0 else np.nan,  # p_52wh
                cur_close / high_12w  if high_12w  > 0 else np.nan,  # p_12wh
                high_50d  / high_200d if high_200d > 0 else np.nan,  # h50_h200
            ]

        return {'feat': feat, 'mom': ret_12_1, 'mcap': mcap, 'price': bars[0].Close}

    def _CSRank(self, X):
        """
        Cross-sectional percentile rank each feature column.
        NaN values (missing data) map to the neutral rank 0.5.
        """
        ranked = np.full_like(X, 0.5)
        for j in range(X.shape[1]):
            col   = X[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                continue
            r     = rankdata(col[valid], method='average')
            ranked[valid, j] = r / (valid.sum() + 1)
        return ranked

    # ── training ──────────────────────────────────────────────────────────────

    def _AddTrainingRows(self, prev, curr):
        """
        Forward returns = curr prices / prev prices - 1.
        Label = 1 if return > cross-sectional median (relative outperformance).
        """
        curr_map = dict(zip(curr['syms'], curr['prices']))
        valid_i, fwd = [], []

        for i, sym in enumerate(prev['syms']):
            if sym in curr_map and prev['prices'][i] > 0:
                valid_i.append(i)
                fwd.append(curr_map[sym] / prev['prices'][i] - 1.0)

        if len(valid_i) < 10:
            return

        fwd = np.array(fwd)
        y   = (fwd > np.median(fwd)).astype(int)
        X   = prev['X'][valid_i]

        self._snap_buf.append(X)
        self._label_buf.append(y)

    def _TrainXGB(self):
        """Retrain XGBoost ensemble (_N_XGB_RUNS independent fits, averaged at predict time).

        Each run uses a different random seed, so subsample / colsample_bytree
        draw different feature/row subsets.  Averaging their predict_proba outputs
        reduces variance and produces more stable probability estimates than a
        single fit — especially important given our relatively small training
        window (~36 months × ~500 stocks).
        """
        X_all = np.vstack(self._snap_buf)
        y_all = np.concatenate(self._label_buf)

        # Drop rows with any NaN (cross-sectional ranking minimises these)
        ok = ~np.any(np.isnan(X_all), axis=1)
        X_all, y_all = X_all[ok], y_all[ok]
        if len(y_all) < 50:
            return

        models = []
        for run in range(self._N_XGB_RUNS):
            clf = xgb.XGBClassifier(
                n_estimators      = 100,
                max_depth         = 3,
                learning_rate     = 0.05,
                subsample         = 0.8,
                colsample_bytree  = 0.8,
                min_child_weight  = 10,
                objective         = 'binary:logistic',
                eval_metric       = 'logloss',
                random_state      = self._SEED + run,   # seeds: _SEED … _SEED + _N_XGB_RUNS - 1
                verbosity         = 0,
            )
            clf.fit(X_all, y_all)
            models.append(clf)
        self._models = models

    # ── classification ────────────────────────────────────────────────────────

    def _MomentumClasses(self, snap):
        """Top/bottom decile by 12-1 momentum → Class H / Class L indices."""
        n      = len(snap['syms'])
        cutoff = max(1, int(n * self._QUANTILE))
        order  = np.argsort(snap['mom'])
        return set(order[-cutoff:].tolist()), set(order[:cutoff].tolist())

    def _XGBClassify(self, snap, h_idx, l_idx):
        """
        Predict P(above-median return) for all stocks.

        "Good" = predicted above-median return (high-return mode) → LONG
        "Bad"  = predicted below-median return (low-return mode)  → SHORT

        Four quadrants:
          Good Winners (H + high p)  → LONG   Bad Winners (H + low p)  → SHORT
          Good Losers  (L + high p)  → LONG   Bad Losers  (L + low p)  → SHORT

        Long book  = Good Winners ∪ Good Losers
        Short book = Bad Winners  ∪ Bad Losers

        Returns (long_idx, short_idx, probs_array, quadrant_dict) for portfolio
        construction and diagnostics.
        """
        X     = snap['X']
        ok    = ~np.any(np.isnan(X), axis=1)
        probs = np.full(len(snap['syms']), 0.5)
        if self._models and ok.sum() > 0:
            # Stack each model's P(above-median) → shape (N_runs, n_ok), then mean
            prob_stack = np.stack(
                [m.predict_proba(X[ok])[:, 1] for m in self._models]
            )
            probs[ok] = prob_stack.mean(axis=0)

        gw_idx = {i for i in h_idx if probs[i] >= self._PROB_THRESHOLD}  # Good Winners → LONG
        gl_idx = {i for i in l_idx if probs[i] >= self._PROB_THRESHOLD}  # Good Losers  → LONG
        bw_idx = {i for i in h_idx if probs[i] <  self._PROB_THRESHOLD}  # Bad  Winners → SHORT
        bl_idx = {i for i in l_idx if probs[i] <  self._PROB_THRESHOLD}  # Bad  Losers  → SHORT

        long_idx  = gw_idx | gl_idx
        short_idx = bw_idx | bl_idx

        quads = {'gw': gw_idx, 'gl': gl_idx, 'bw': bw_idx, 'bl': bl_idx}
        return long_idx, short_idx, probs, quads

    # ── portfolio construction ─────────────────────────────────────────────────

    def _TargetWeights(self, snap, long_idx, short_idx):
        """
        Equal-weighted, dollar-neutral: 50% long / 50% short of NAV.
        long_idx  = Good Winners ∪ Good Losers
        short_idx = Bad Winners  ∪ Bad Losers
        """
        syms = snap['syms']
        w_long  =  0.5 / len(long_idx)  if long_idx  else 0.0
        w_short = -0.5 / len(short_idx) if short_idx else 0.0
        weights = {syms[i]: w_long  for i in long_idx}
        weights.update({syms[i]: w_short for i in short_idx})
        return weights

    # ── diagnostic helpers ────────────────────────────────────────────────────

    def _LogDiagnostics(self, snap, h_idx, l_idx, gw_idx, gl_idx, probs, targets):
        syms, prices = snap['syms'], snap['prices']

        # 1. Probability spread — key indicator of model signal quality
        p_gw = [probs[i] for i in gw_idx] or [0.5]
        p_gl = [probs[i] for i in gl_idx] or [0.5]
        p_bw = [probs[i] for i in h_idx - gw_idx] or [0.5]
        p_bl = [probs[i] for i in l_idx - gl_idx] or [0.5]
        spread = float(np.mean(p_gw) - np.mean(p_gl))

        # 2. Realized quadrant returns from last rebalance
        q_rets = self._QuadrantReturns(snap)

        # 3. Feature NaN rates (fundamental data quality check)
        nan_rates = self._FeatureNanRates(snap)
        top_nan = sorted(zip(self.FEATURES, nan_rates), key=lambda x: -x[1])[:4]
        nan_str  = "  ".join(f"{n}={v:.0%}" for n, v in top_nan if v > 0.1)

        # 4. Long/short book beta vs SPY
        long_beta, short_beta = self._BookBetas(targets)

        rec = {
            'date':       self.Time,
            'prob_gw':    float(np.mean(p_gw)),
            'prob_gl':    float(np.mean(p_gl)),
            'prob_bw':    float(np.mean(p_bw)),
            'prob_bl':    float(np.mean(p_bl)),
            'prob_spread': spread,
            'ret_gw':     q_rets['GW'],
            'ret_gl':     q_rets['GL'],
            'ret_bw':     q_rets['BW'],
            'ret_bl':     q_rets['BL'],
            'long_beta':  long_beta,
            'short_beta': short_beta,
            'net_beta':   long_beta + short_beta if not (np.isnan(long_beta) or np.isnan(short_beta)) else np.nan,
            'n_gw':       len(gw_idx),
            'n_gl':       len(gl_idx),
        }
        self._diag_log.append(rec)

        self.Log(
            f"  DIAG {self.Time:%Y-%m} | "
            f"prob_spread={spread:+.3f} (GW={np.mean(p_gw):.3f} GL={np.mean(p_gl):.3f}) | "
            f"quad_ret GW={q_rets['GW']:+.3f} BW={q_rets['BW']:+.3f} GL={q_rets['GL']:+.3f} BL={q_rets['BL']:+.3f} | "
            f"beta L={long_beta:+.2f} S={short_beta:+.2f} net={rec['net_beta']:+.2f} | "
            f"nan: {nan_str}"
        )

    def _QuadrantReturns(self, curr_snap):
        """Realized 1-month returns for each quadrant assigned last rebalance."""
        empty = {'GW': np.nan, 'GL': np.nan, 'BW': np.nan, 'BL': np.nan}
        if self._prev_quads is None:
            return empty
        curr_map = dict(zip(curr_snap['syms'], curr_snap['prices']))
        buckets  = {'GW': [], 'GL': [], 'BW': [], 'BL': []}
        for sym, (quad, prev_p) in self._prev_quads.items():
            if sym in curr_map and prev_p > 0:
                buckets[quad].append(curr_map[sym] / prev_p - 1.0)
        return {q: float(np.mean(v)) if v else np.nan for q, v in buckets.items()}

    def _FeatureNanRates(self, snap):
        """Per-feature NaN rate across current eligible stocks."""
        X_raw = snap.get('X_raw')
        if X_raw is None or X_raw.size == 0:
            return [np.nan] * self.N_FEAT
        return [float(np.mean(np.isnan(X_raw[:, j]))) for j in range(X_raw.shape[1])]

    def _BookBetas(self, targets):
        """
        Weighted average SPY-beta for the long and short books.
        Uses 30-day rolling OLS. Returns (long_beta, short_beta).
        """
        if not self._spy_bars.IsReady:
            return np.nan, np.nan

        spy_prices = np.array([self._spy_bars[i].Close for i in range(min(31, self._spy_bars.Count))][::-1])
        spy_rets   = np.diff(spy_prices) / np.maximum(spy_prices[:-1], 1e-8)   # pct returns, oldest→newest
        var_spy    = float(np.var(spy_rets))
        if var_spy < 1e-12:
            return np.nan, np.nan

        long_beta_acc, long_w = 0.0, 0.0
        short_beta_acc, short_w = 0.0, 0.0

        for sym, w in targets.items():
            if sym not in self._bars or not self._bars[sym].IsReady:
                continue
            bars     = list(self._bars[sym])
            s_prices = np.array([bars[i].Close for i in range(min(31, len(bars)))][::-1])
            s_rets   = np.diff(s_prices) / np.maximum(s_prices[:-1], 1e-8)   # pct returns
            n = min(len(s_rets), len(spy_rets))
            if n < 10:
                continue
            cov = float(np.cov(s_rets[:n], spy_rets[:n])[0, 1])
            beta = cov / var_spy
            if w > 0:
                long_beta_acc  += w * beta
                long_w         += w
            else:
                short_beta_acc += w * beta   # w is negative
                short_w        += abs(w)

        lb = long_beta_acc  / long_w   if long_w  > 0 else np.nan
        sb = short_beta_acc / short_w  if short_w > 0 else np.nan
        return lb, sb

    # ── end of algorithm ──────────────────────────────────────────────────────

    def OnEndOfAlgorithm(self):
        self.Log("=" * 65)
        self.Log("PHASE 2 — DIAGNOSTIC SUMMARY")
        self.Log(f"  Live rebalances : {len(self._diag_log)}")

        if not self._diag_log:
            self.Log("  No diagnostic data collected.")
            self.Log("=" * 65)
            return

        # ── 1. Model signal quality ──────────────────────────────────────────
        spreads = [d['prob_spread'] for d in self._diag_log if not np.isnan(d['prob_spread'])]
        self.Log(f"\n  [SIGNAL QUALITY — probability spread GW-GL]")
        self.Log(f"  Mean spread  : {np.mean(spreads):+.4f}  (0 = random, >0.10 = useful)")
        self.Log(f"  Pct positive : {np.mean([s > 0 for s in spreads]):.1%}")
        self.Log(f"  Mean P(GW)   : {np.mean([d['prob_gw'] for d in self._diag_log]):.4f}")
        self.Log(f"  Mean P(GL)   : {np.mean([d['prob_gl'] for d in self._diag_log]):.4f}")

        # ── 2. Realized quadrant returns ─────────────────────────────────────
        def avg_ret(key):
            v = [d[key] for d in self._diag_log if not np.isnan(d.get(key, np.nan))]
            return float(np.mean(v)) if v else np.nan
        r_gw = avg_ret('ret_gw')
        r_gl = avg_ret('ret_gl')
        r_bw = avg_ret('ret_bw')
        r_bl = avg_ret('ret_bl')
        avg_long  = (r_gw + r_gl) / 2   # equal-weight GW and GL (both longed)
        avg_short = (r_bw + r_bl) / 2   # equal-weight BW and BL (both shorted)
        self.Log(f"\n  [REALIZED QUADRANT RETURNS — monthly avg]")
        self.Log(f"  Good Winners : {r_gw:+.4f}  (Longed)")
        self.Log(f"  Good Losers  : {r_gl:+.4f}  (Longed — reversal trade)")
        self.Log(f"  Bad  Winners : {r_bw:+.4f}  (Shorted — drag when positive)")
        self.Log(f"  Bad  Losers  : {r_bl:+.4f}  (Shorted — drag when positive)")
        self.Log(f"  Avg long  (GW+GL)/2 : {avg_long:+.4f}")
        self.Log(f"  Avg short (BW+BL)/2 : {avg_short:+.4f}")
        self.Log(f"  L/S spread          : {avg_long - avg_short:+.4f}  (long - short; positive = strategy adds value)")

        # ── 3. Beta leakage ──────────────────────────────────────────────────
        net_betas = [d['net_beta'] for d in self._diag_log if not np.isnan(d.get('net_beta', np.nan))]
        if net_betas:
            self.Log(f"\n  [BETA LEAKAGE]")
            self.Log(f"  Mean net beta   : {np.mean(net_betas):+.3f}  (target ~0)")
            self.Log(f"  Std  net beta   : {np.std(net_betas):.3f}  (low = stable neutrality)")
            self.Log(f"  Min / Max beta  : {min(net_betas):+.3f} / {max(net_betas):+.3f}")

        # ── 4. Feature NaN rates ──────────────────────────────────────────────
        # Per-feature NaN rates are logged each month in DIAG lines.
        # No aggregate needed here — grep "nan:" in the log for per-feature detail.
        self.Log(f"\n  [FEATURE DATA QUALITY]")
        self.Log(f"  (Per-feature NaN rates logged monthly — grep 'nan:' in backtest log)")

        # ── 5. Feature importances (ensemble average, final trained models) ──
        if self._models:
            fi   = np.mean([m.feature_importances_ for m in self._models], axis=0)
            top5 = sorted(zip(self.FEATURES, fi), key=lambda x: -x[1])[:5]
            self.Log(f"\n  [FEATURE IMPORTANCES — ensemble avg ({len(self._models)} runs)]")
            for name, imp in top5:
                self.Log(f"    {name:<22} {imp:.4f}")

        self.Log("=" * 65)
