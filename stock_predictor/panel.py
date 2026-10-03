"""Cross-sectional panel model over the whole S&P 500 (the "real-world" quant approach).

Why this instead of one model per ticker?
  * One stock has ~3.5k daily rows → far too little data for the weak signal in prices.
    The panel has ~1.7M rows, so the model learns patterns shared across 500 stocks.
  * The target is *relative*: "will this stock beat the median S&P 500 stock over the next
    H days?". This strips out market direction (the noisiest, least predictable part),
    gives perfectly balanced classes, and is what long/short equity funds actually trade.
  * Evaluated with the industry-standard Information Coefficient (IC), decile spreads and
    a long/short portfolio backtest with transaction costs.

Protocol (no leakage):
  1. Hyper-parameters are tuned with Optuna on yearly walk-forward folds in the VALIDATION
     period (default 2016-2019) only.
  2. With the tuned parameters the model is re-trained every year (expanding window, purged
     by H days) and scored on the untouched HOLDOUT period (default 2020 → end of data).
  3. A final model is trained on all data and used for live predictions.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from .features import build_features
from .kaggle_data import DEFAULT_VERSION, fetch_recent_yahoo, load_panel

log = logging.getLogger(__name__)

RANK_COLS = [
    "ret_1", "ret_5", "ret_20", "ret_60", "ret_120", "vol_20", "vol_60", "rsi_14", "rsi_5",
    "sma_dist_20", "sma_dist_50", "sma_dist_200", "dist_52w_high", "dist_52w_low",
    "volume_z_20", "atr_14", "beta_60", "bb_pctb", "macd_hist", "obv_slope_20", "skew_20",
    "rel_strength_20", "rel_strength_60", "dollar_vol_20", "amihud_20", "mfi_14",
]
DROP_COLS = ["dow", "month"]

DEFAULT_PARAMS = {
    "target_type": "regression",
    "num_leaves": 63,
    "max_depth": 8,
    "learning_rate": 0.03,
    "min_child_samples": 1000,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_lambda": 10.0,
    "reg_alpha": 1e-3,
}
DEFAULT_ROUNDS = 400


@dataclass
class PanelConfig:
    horizon: int = 5
    tune_start: str = "2016-01-01"
    test_start: str = "2020-01-01"
    top_q: float = 0.1                # long top 10%, short bottom 10%
    cost_bps: float = 5.0             # per unit of turnover
    n_trials: int = 25
    tune_stride: int = 2              # use every 2nd date when tuning (speed)
    kaggle_version: int = DEFAULT_VERSION
    data_dir: str = "data"
    model_dir: str = "models"
    report_dir: str = "reports"
    seed: int = 42
    n_jobs: int = -1

    @property
    def tag(self) -> str:
        return f"panel_h{self.horizon}"


# ============================================================================
# Features
# ============================================================================
def equal_weight_market(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["Close"].unstack("Symbol").astype(float)
    rets = close.pct_change(fill_method=None).clip(-0.5, 0.5)
    mret = rets.mean(axis=1).fillna(0.0)
    return pd.DataFrame({"Close": 100 * (1 + mret).cumprod()})


def _symbol_features(sym: str, g: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    g = g.droplevel("Symbol").astype(float)
    f = build_features(g, market).drop(columns=DROP_COLS, errors="ignore")
    c, v = g["Close"], g["Volume"]
    dv = (c * v).rolling(20).mean()
    f["dollar_vol_20"] = np.log1p(dv)
    f["amihud_20"] = np.log1p((c.pct_change().abs() / (c * v).replace(0, np.nan)).rolling(20).mean() * 1e9)
    f["Symbol"] = sym
    return f


def build_panel_features(
    prices: pd.DataFrame, companies: pd.DataFrame, sectors: List[str], n_jobs: int = -1
) -> pd.DataFrame:
    t0 = time.time()
    market = equal_weight_market(prices)
    groups = list(prices.groupby(level="Symbol"))
    parts = Parallel(n_jobs=n_jobs)(delayed(_symbol_features)(s, g, market) for s, g in groups)
    feats = pd.concat(parts).reset_index().set_index(["Date", "Symbol"]).sort_index()
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats[feats["sma_dist_200"].notna() & feats["ret_120"].notna()]

    dates = feats.index.get_level_values("Date")
    grp = feats.groupby(level="Date")
    for col in RANK_COLS:
        feats[f"cs_{col}"] = grp[col].rank(pct=True)

    sector = companies["Sector"].reindex(feats.index.get_level_values("Symbol")).fillna("Unknown").values
    feats["sector"] = pd.Categorical(sector, categories=sectors).codes
    for col in ("ret_5", "ret_20", "ret_60"):
        feats[f"sec_rel_{col}"] = feats[col] - feats.groupby([dates, sector])[col].transform("mean")
    feats["universe_size"] = grp["ret_1"].transform("size")

    log.info("Built %d features for %d rows in %.0fs", feats.shape[1], len(feats), time.time() - t0)
    return feats.astype("float32")


def make_targets(prices: pd.DataFrame, index: pd.MultiIndex, horizon: int):
    close = prices["Close"].astype(float)
    fwd = (close.groupby(level="Symbol").shift(-horizon) / close - 1).reindex(index)
    d = index.get_level_values("Date")
    med = fwd.groupby(d).transform("median")
    y = (fwd > med).astype(float).where(fwd.notna())
    rank = fwd.groupby(d).rank(pct=True) - 0.5
    return fwd, y, rank


# ============================================================================
# Metrics
# ============================================================================
def daily_ic(score, fwd, dates) -> pd.Series:
    """Per-date Spearman rank correlation between prediction and realized forward return."""
    df = pd.DataFrame({"d": dates, "s": score, "f": fwd})
    g = df.groupby("d")
    rs = g["s"].rank()
    rf = g["f"].rank()
    rs = rs - rs.groupby(df["d"]).transform("mean")
    rf = rf - rf.groupby(df["d"]).transform("mean")
    num = (rs * rf).groupby(df["d"]).sum()
    den = np.sqrt((rs**2).groupby(df["d"]).sum() * (rf**2).groupby(df["d"]).sum())
    return (num / den).dropna()


def _perf(rets: pd.Series, periods_per_year: float) -> Dict[str, float]:
    r = rets.fillna(0.0)
    eq = (1 + r).cumprod()
    years = len(r) / periods_per_year
    sd = r.std()
    return {
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else float("nan"),
        "ann_vol": float(sd * np.sqrt(periods_per_year)),
        "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "total_return": float(eq.iloc[-1] - 1),
        "win_rate": float((r > 0).mean()),
    }


def portfolio_backtest(oos: pd.DataFrame, horizon: int, top_q: float, cost_bps: float):
    """Rebalance every `horizon` days (non-overlapping). Long top-q, short bottom-q, equal weight."""
    dates = np.sort(oos["date"].unique())[::horizon]
    sub = oos[oos["date"].isin(dates)]
    rows, prev_long, prev_short = [], {}, {}

    def turnover(new: set, old: dict) -> float:
        w_new = {s: 1 / len(new) for s in new} if new else {}
        keys = set(w_new) | set(old)
        return sum(abs(w_new.get(k, 0) - old.get(k, 0)) for k in keys), w_new

    for d, g in sub.groupby("date"):
        n = max(1, int(len(g) * top_q))
        g = g.sort_values("score")
        short, long_ = g.head(n), g.tail(n)
        to_l, prev_long = turnover(set(long_["symbol"]), prev_long)
        to_s, prev_short = turnover(set(short["symbol"]), prev_short)
        c = cost_bps / 1e4
        lr, sr, ur = long_["fwd"].mean(), short["fwd"].mean(), g["fwd"].mean()
        rows.append({
            "date": d, "long": lr, "short": sr, "universe": ur,
            "long_short": (lr - sr) / 2 - c * (to_l + to_s) / 2,   # 50% long / 50% short, dollar-neutral
            "long_only": lr - c * to_l,
            "turnover_long": to_l / 2,
        })
    bt = pd.DataFrame(rows).set_index("date")
    ppy = 252 / horizon
    perf = {
        "long_short_net": _perf(bt["long_short"], ppy),
        "long_only_top_net": _perf(bt["long_only"], ppy),
        "universe_equal_weight": _perf(bt["universe"], ppy),
        "avg_turnover_long_per_rebalance": float(bt["turnover_long"].mean()),
    }
    return bt, perf


def evaluate_oos(oos: pd.DataFrame, cfg: PanelConfig, decile_map: Optional[pd.DataFrame] = None) -> dict:
    H = cfg.horizon
    ic = daily_ic(oos["score"].values, oos["fwd"].values, oos["date"].values)
    med = oos.groupby("date")["score"].transform("median")
    acc = float(((oos["score"] > med).astype(int) == oos["y"]).mean())
    dec = decile_table(oos)
    _, perf = portfolio_backtest(oos, H, cfg.top_q, cfg.cost_bps)
    out = {
        "period": f"{oos['date'].min().date()} → {oos['date'].max().date()}",
        "n_rows": int(len(oos)),
        "n_dates": int(oos["date"].nunique()),
        "auc": float(roc_auc_score(oos["y"], oos["score"])),
        "cs_accuracy": acc,
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_ir_annual": float(ic.mean() / ic.std() * np.sqrt(252 / H)),
        "ic_tstat": float(ic.mean() / ic.std() * np.sqrt(len(ic) / H)),  # overlap-adjusted
        "ic_hit_rate": float((ic > 0).mean()),
        "top_minus_bottom_decile_fwd": float(dec["mean_fwd"].iloc[-1] - dec["mean_fwd"].iloc[0]),
        "deciles_mean_fwd": [float(x) for x in dec["mean_fwd"]],
        "portfolio": perf,
    }
    if decile_map is not None:
        out["price_forecast"] = price_forecast_eval(oos, decile_map)
    return out


def _add_decile(oos: pd.DataFrame) -> pd.Series:
    pct = oos.groupby("date")["score"].rank(pct=True)
    return np.ceil(pct * 10).clip(1, 10).astype(int)


def decile_table(oos: pd.DataFrame) -> pd.DataFrame:
    """Expected return & spread per score decile (used to turn rankings into price targets)."""
    o = oos.assign(dec=_add_decile(oos))
    by_date = o.groupby(["date", "dec"])["fwd"].mean().groupby("dec").mean()
    q = o.groupby("dec")["fwd"].quantile([0.1, 0.5, 0.9]).unstack()
    t = pd.DataFrame({"mean_fwd": by_date, "q10": q[0.1], "median": q[0.5], "q90": q[0.9],
                      "hit_rate_up": o.groupby("dec")["fwd"].apply(lambda x: (x > 0).mean())})
    return t


def price_forecast_eval(oos: pd.DataFrame, decile_map: pd.DataFrame) -> dict:
    """Price target = close × (1 + expected return of the stock's decile). Compared to naive
    'price unchanged' and 'universe average drift' forecasts."""
    o = oos.assign(dec=_add_decile(oos))
    mu = o["dec"].map(decile_map["mean_fwd"])
    actual = o["close"] * (1 + o["fwd"])
    pred = o["close"] * (1 + mu)
    drift = o["close"] * (1 + decile_map["mean_fwd"].mean())
    lo = o["close"] * (1 + o["dec"].map(decile_map["q10"]))
    hi = o["close"] * (1 + o["dec"].map(decile_map["q90"]))

    def mape(p):
        return float((np.abs(p - actual) / actual).mean())

    return {
        "mape_model": mape(pred),
        "mape_naive_last_price": mape(o["close"]),
        "mape_drift": mape(drift),
        "interval_10_90_coverage": float(((actual >= lo) & (actual <= hi)).mean()),
        "abs_direction_accuracy": float(((mu > 0) == (o["fwd"] > 0)).mean()),
        "abs_direction_base_rate_up": float((o["fwd"] > 0).mean()),
    }


# ============================================================================
# Training helpers
# ============================================================================
def lgb_params(p: dict, seed: int, n_jobs: int) -> dict:
    binary = p.get("target_type", "regression") == "binary"
    return {
        "objective": "binary" if binary else "regression",
        "metric": "binary_logloss" if binary else "l2",
        "num_leaves": int(p["num_leaves"]),
        "max_depth": int(p["max_depth"]),
        "learning_rate": float(p["learning_rate"]),
        "min_data_in_leaf": int(p["min_child_samples"]),
        "bagging_fraction": float(p["subsample"]),
        "bagging_freq": 1,
        "feature_fraction": float(p["colsample_bytree"]),
        "lambda_l2": float(p["reg_lambda"]),
        "lambda_l1": float(p.get("reg_alpha", 0.0)),
        "max_bin": 63,
        "verbosity": -1,
        "seed": seed,
        "num_threads": 0 if n_jobs in (-1, None) else n_jobs,
        "feature_pre_filter": False,
    }


def _label(target_type: str, y: np.ndarray, rank: np.ndarray) -> np.ndarray:
    return y if target_type == "binary" else rank


def yearly_folds(dates: np.ndarray, start: str, end: Optional[str], horizon: int):
    """Yield (year, train_last_date, test_first_date, test_last_date) with an H-day purge."""
    ud = np.sort(np.unique(dates))
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) if end else pd.Timestamp(ud[-1]) + pd.Timedelta(days=1)
    for yr in range(s.year, e.year + 1):
        t0, t1 = max(pd.Timestamp(yr, 1, 1), s), min(pd.Timestamp(yr + 1, 1, 1), e)
        td = ud[(ud >= np.datetime64(t0)) & (ud < np.datetime64(t1))]
        if len(td) == 0:
            continue
        pos = np.searchsorted(ud, td[0])
        if pos - horizon - 1 < 252:
            continue
        yield yr, ud[pos - horizon - 1], td[0], td[-1]


class PanelData:
    """Holds the numpy arrays for fast repeated training."""

    def __init__(self, feats: pd.DataFrame, prices: pd.DataFrame, horizon: int):
        fwd, y, rank = make_targets(prices, feats.index, horizon)
        lab = fwd.notna().values
        self.feature_cols = [c for c in feats.columns]
        self.X = feats.to_numpy(np.float32)[lab]
        self.dates = feats.index.get_level_values("Date").values[lab]
        self.symbols = feats.index.get_level_values("Symbol").values[lab]
        self.fwd = fwd.values[lab].astype(np.float64)
        self.y = y.values[lab].astype(np.float32)
        self.rank = rank.values[lab].astype(np.float32)
        self.close = prices["Close"].reindex(feats.index).values[lab].astype(np.float64)
        self.cat_idx = [self.feature_cols.index("sector")]
        ud = np.unique(self.dates)
        self.date_pos = np.searchsorted(ud, self.dates)

    def dataset(self, mask, target_type, reference=None):
        return lgb.Dataset(self.X[mask], label=_label(target_type, self.y[mask], self.rank[mask]),
                           feature_name=self.feature_cols, categorical_feature=self.cat_idx,
                           reference=reference, free_raw_data=False,
                           params={"max_bin": 63, "feature_pre_filter": False, "verbosity": -1})

    def oos_frame(self, mask, score) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.to_datetime(self.dates[mask]), "symbol": self.symbols[mask], "score": score,
            "y": self.y[mask].astype(int), "fwd": self.fwd[mask], "close": self.close[mask],
        })


# ============================================================================
# Tuning
# ============================================================================
def tune(pdata: PanelData, cfg: PanelConfig) -> dict:
    import optuna

    folds = list(yearly_folds(pdata.dates, cfg.tune_start, cfg.test_start, cfg.horizon))
    cache: Dict[tuple, tuple] = {}

    def get(k, tt):
        if (k, tt) not in cache:
            _, tr_last, te0, te1 = folds[k]
            tr = (pdata.dates <= tr_last) & (pdata.date_pos % cfg.tune_stride == 0)
            te = (pdata.dates >= te0) & (pdata.dates <= te1)
            dtr = pdata.dataset(tr, tt)
            dva = pdata.dataset(te, tt, reference=dtr)
            cache[(k, tt)] = (dtr, dva, te)
        return cache[(k, tt)]

    def objective(trial):
        p = {
            "target_type": trial.suggest_categorical("target_type", ["regression", "binary"]),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 200, 8000, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        }
        params = lgb_params(p, cfg.seed, cfg.n_jobs)
        ics, iters = [], []
        for k in range(len(folds)):
            dtr, dva, te = get(k, p["target_type"])
            booster = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            score = booster.predict(pdata.X[te], num_iteration=booster.best_iteration)
            ics.append(float(daily_ic(score, pdata.fwd[te], pdata.dates[te]).mean()))
            iters.append(int(booster.best_iteration or 1500))
            trial.report(float(np.mean(ics)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("iters", iters)
        trial.set_user_attr("fold_ics", ics)
        log.info("Trial %d: mean IC=%.4f folds=%s iters=%s", trial.number, np.mean(ics),
                 np.round(ics, 4).tolist(), iters)
        return float(np.mean(ics))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1),
    )
    study.enqueue_trial(DEFAULT_PARAMS)  # always evaluate the hand-picked baseline too
    study.optimize(objective, n_trials=cfg.n_trials)
    best = study.best_trial
    rounds = int(np.median(best.user_attrs["iters"]) * 1.15) + 10  # more data in final fits
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    return {"params": dict(best.params), "rounds": rounds, "val_ic": best.value,
            "fold_ics": best.user_attrs["fold_ics"], "trials": trials}


# ============================================================================
# Full pipeline
# ============================================================================
def _sectors(companies: pd.DataFrame) -> List[str]:
    return sorted(companies["Sector"].dropna().unique().tolist()) + ["Unknown"]


def load_features(cfg: PanelConfig):
    prices, companies = load_panel(cfg.data_dir, cfg.kaggle_version)
    sectors = _sectors(companies)
    cache = os.path.join(cfg.data_dir, f"sp500_features_v{cfg.kaggle_version}.parquet")
    if os.path.exists(cache):
        feats = pd.read_parquet(cache)
    else:
        feats = build_panel_features(prices, companies, sectors, cfg.n_jobs)
        feats.to_parquet(cache)
    return prices, companies, sectors, feats


def run_panel(cfg: PanelConfig, make_plots: bool = True) -> dict:
    t_start = time.time()
    prices, companies, sectors, feats = load_features(cfg)
    pdata = PanelData(feats, prices, cfg.horizon)
    log.info("Panel dataset: %d labeled rows × %d features", *pdata.X.shape)
    del feats

    # 1) Tune on the validation period only
    if cfg.n_trials > 0:
        t = time.time()
        tuned = tune(pdata, cfg)
        params, rounds = tuned["params"], tuned["rounds"]
        log.info("Tuning done in %.1f min: val IC=%.4f params=%s rounds=%d",
                 (time.time() - t) / 60, tuned["val_ic"], params, rounds)
    else:
        tuned, params, rounds = None, dict(DEFAULT_PARAMS), DEFAULT_ROUNDS
    lp = lgb_params(params, cfg.seed, cfg.n_jobs)
    tt = params.get("target_type", "regression")

    # 2) Walk-forward re-training every year over validation + holdout
    frames = []
    for yr, tr_last, te0, te1 in yearly_folds(pdata.dates, cfg.tune_start, None, cfg.horizon):
        tr = pdata.dates <= tr_last
        te = (pdata.dates >= te0) & (pdata.dates <= te1)
        booster = lgb.train(lp, pdata.dataset(tr, tt), num_boost_round=rounds)
        frames.append(pdata.oos_frame(te, booster.predict(pdata.X[te])).assign(fold=yr))
        log.info("Walk-forward %d: train ≤ %s (%d rows) → test %d rows", yr,
                 pd.Timestamp(tr_last).date(), tr.sum(), te.sum())
    oos = pd.concat(frames, ignore_index=True)

    val = oos[oos["date"] < pd.Timestamp(cfg.test_start)]
    hold = oos[oos["date"] >= pd.Timestamp(cfg.test_start)]
    val_deciles = decile_table(val)
    metrics = {
        "validation": evaluate_oos(val, cfg),
        "holdout": evaluate_oos(hold, cfg, decile_map=val_deciles),
    }
    per_year = []
    for yr, g in oos.groupby("fold"):
        ic = daily_ic(g["score"].values, g["fwd"].values, g["date"].values)
        _, perf = portfolio_backtest(g, cfg.horizon, cfg.top_q, cfg.cost_bps)
        per_year.append({"year": yr, "set": "holdout" if g["date"].min() >= pd.Timestamp(cfg.test_start) else "validation",
                         "ic_mean": ic.mean(), "auc": roc_auc_score(g["y"], g["score"]),
                         "ls_return": perf["long_short_net"]["total_return"],
                         "top_decile_return": perf["long_only_top_net"]["total_return"],
                         "universe_return": perf["universe_equal_weight"]["total_return"]})
    per_year = pd.DataFrame(per_year)

    # 3) Final model on everything
    booster = lgb.train(lp, pdata.dataset(np.ones(len(pdata.y), bool), tt), num_boost_round=rounds)
    importance = pd.Series(booster.feature_importance("gain"), index=pdata.feature_cols)
    importance = (importance / importance.sum()).sort_values(ascending=False)
    full_deciles = decile_table(oos)

    os.makedirs(cfg.model_dir, exist_ok=True)
    booster.save_model(os.path.join(cfg.model_dir, f"{cfg.tag}.txt"))
    meta = {
        "config": asdict(cfg), "params": params, "rounds": rounds, "feature_cols": pdata.feature_cols,
        "sectors": sectors, "decile_map": full_deciles.reset_index().to_dict(orient="list"),
        "train_end": str(pd.Timestamp(pdata.dates.max()).date()), "n_rows": int(len(pdata.y)),
        "metrics": metrics,
    }
    with open(os.path.join(cfg.model_dir, f"{cfg.tag}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    out_dir = os.path.join(cfg.report_dir, cfg.tag)
    os.makedirs(out_dir, exist_ok=True)
    oos.to_parquet(os.path.join(out_dir, "oos_predictions.parquet"))
    per_year.to_csv(os.path.join(out_dir, "per_year.csv"), index=False)
    importance.rename("gain_share").to_csv(os.path.join(out_dir, "feature_importance.csv"))
    full_deciles.to_csv(os.path.join(out_dir, "deciles.csv"))
    if tuned:
        tuned["trials"].to_csv(os.path.join(out_dir, "optuna_trials.csv"), index=False)
    bt_hold, _ = portfolio_backtest(hold, cfg.horizon, cfg.top_q, cfg.cost_bps)
    bt_all, _ = portfolio_backtest(oos, cfg.horizon, cfg.top_q, cfg.cost_bps)
    bt_all.to_csv(os.path.join(out_dir, "portfolio_backtest.csv"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"metrics": metrics, "params": params, "rounds": rounds,
                   "tuning": {k: v for k, v in (tuned or {}).items() if k != "trials"},
                   "runtime_min": (time.time() - t_start) / 60}, fh, indent=2, default=str)
    if make_plots:
        try:
            _plots(bt_all, oos, importance, full_deciles, cfg, out_dir)
        except Exception as exc:
            log.warning("Plotting failed: %s", exc)

    return {"metrics": metrics, "per_year": per_year, "importance": importance, "deciles": full_deciles,
            "params": params, "rounds": rounds, "tuned": tuned, "report_dir": out_dir,
            "runtime_min": (time.time() - t_start) / 60}


# ============================================================================
# Live prediction
# ============================================================================
def load_panel_model(cfg: PanelConfig):
    path = os.path.join(cfg.model_dir, f"{cfg.tag}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No panel model at {path}. Run `python main.py panel-train` first.")
    with open(os.path.join(cfg.model_dir, f"{cfg.tag}_meta.json")) as fh:
        meta = json.load(fh)
    return lgb.Booster(model_file=path), meta


def predict_panel(cfg: PanelConfig, refresh: bool = True) -> pd.DataFrame:
    """Score every S&P 500 stock on the latest date. Returns a ranked table with price targets."""
    booster, meta = load_panel_model(cfg)
    prices, companies = load_panel(cfg.data_dir, cfg.kaggle_version)
    if refresh:
        prices = fetch_recent_yahoo(prices.index.get_level_values("Symbol").unique().tolist())
    else:
        last_dates = np.sort(prices.index.get_level_values("Date").unique())[-400:]
        prices = prices[prices.index.get_level_values("Date").isin(last_dates)]
    feats = build_panel_features(prices, companies, meta["sectors"], cfg.n_jobs)
    last = feats.index.get_level_values("Date").max()
    x = feats.xs(last, level="Date")
    score = booster.predict(x[meta["feature_cols"]].to_numpy(np.float32))
    dmap = pd.DataFrame(meta["decile_map"]).set_index("dec")
    out = pd.DataFrame({"score": score}, index=x.index)
    out["pct_rank"] = out["score"].rank(pct=True)
    out["decile"] = np.ceil(out["pct_rank"] * 10).clip(1, 10).astype(int)
    out["prob_beat_median"] = out["decile"].map(dmap["hit_rate_up"])  # P(up) of decile, OOS
    out["exp_return"] = out["decile"].map(dmap["mean_fwd"])
    close = prices["Close"].xs(last, level="Date").reindex(out.index).astype(float)
    out["close"] = close
    out["target_price"] = close * (1 + out["exp_return"])
    out["low_10"] = close * (1 + out["decile"].map(dmap["q10"]))
    out["high_90"] = close * (1 + out["decile"].map(dmap["q90"]))
    out["signal"] = np.select([out["decile"] == 10, out["decile"] == 1], ["STRONG BUY (top 10%)", "STRONG SELL (bottom 10%)"],
                              np.where(out["decile"] >= 8, "BUY", np.where(out["decile"] <= 3, "SELL", "NEUTRAL")))
    out = out.join(companies[["Shortname", "Sector"]], how="left")
    out.insert(0, "as_of", str(last.date()))
    out = out.sort_values("score", ascending=False)
    os.makedirs(os.path.join(cfg.report_dir, cfg.tag), exist_ok=True)
    out.to_csv(os.path.join(cfg.report_dir, cfg.tag, f"live_predictions_{last.date()}.csv"))
    return out


# ============================================================================
# Plots
# ============================================================================
def _plots(bt, oos, importance, deciles, cfg, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for col, lab in [("long_short", "Long/short top-bottom decile (net)"),
                     ("long_only", "Long top decile (net)"), ("universe", "S&P 500 equal-weight")]:
        ax.plot(bt.index, (1 + bt[col]).cumprod(), label=lab)
    ax.axvline(pd.Timestamp(cfg.test_start), color="k", ls="--", lw=1)
    ax.text(pd.Timestamp(cfg.test_start), ax.get_ylim()[1] * 0.95, "  holdout →", va="top")
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Panel model (H={cfg.horizon}d) — walk-forward out-of-sample portfolios")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "portfolio_equity.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(deciles.index, deciles["mean_fwd"] * 1e4, color=["tab:red"] * 3 + ["tab:gray"] * 4 + ["tab:green"] * 3)
    ax.set_xlabel("Predicted score decile (1 = worst, 10 = best)")
    ax.set_ylabel(f"Avg realized {cfg.horizon}-day return (bps)")
    ax.set_title("Out-of-sample decile returns"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "decile_returns.png"), dpi=120); plt.close(fig)

    ic = daily_ic(oos["score"].values, oos["fwd"].values, oos["date"].values)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ic.index, ic.rolling(63).mean(), label="IC (63-day rolling mean)")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(pd.Timestamp(cfg.test_start), color="k", ls="--", lw=1)
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Information Coefficient over time")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "rolling_ic.png"), dpi=120); plt.close(fig)

    top = importance.head(25)[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top.index, top.values); ax.set_title("Top 25 features (gain share)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=120); plt.close(fig)
