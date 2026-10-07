"""Deterministic time-series analytics.

All numbers shown to the user come from here. Incomplete trailing periods are
detected and excluded so that a half-finished month is never reported as a
collapse; forecasts are only produced when enough history exists and are
back-tested so their reliability can be stated honestly.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .dataset import Dataset
from .fmt import month_name, period_label, unit_word, weekday_name

_SEASON_LEN = {"M": 12, "Q": 4, "D": 7}
_MIN_HISTORY = {"M": 18, "Q": 12, "W": 26, "D": 42, "Y": 8}
_HORIZON = {"M": 6, "Q": 4, "W": 8, "D": 14, "Y": 2}


@dataclass
class TS:
    measure: str
    freq: str
    values: pd.Series                      # complete periods only; index = period start
    counts: pd.Series
    partial_last: Optional[dict] = None    # {"start": Timestamp, "value": float, "coverage": float}
    partial_first_dropped: bool = False
    gap_share: float = 0.0
    kind: str = "intensity"

    def label(self, ts: pd.Timestamp) -> str:
        return period_label(ts, self.freq)


def choose_freq(ds: Dataset) -> str:
    spec = ds.time_spec
    if spec is None:
        raise ValueError("no time axis")
    if spec["kind"] == "year":
        return "Y"
    t = ds.times().dropna()
    span = (t.max() - t.min()).days
    if spec["kind"] == "year_month":
        return "M" if span <= 365 * 8 else ("Q" if span <= 365 * 30 else "Y")
    if span <= 60:
        return "D"
    if span < 150:
        return "W"
    if span <= 365 * 8:
        return "M"
    return "Q" if span <= 365 * 30 else "Y"


def build_ts(ds: Dataset, measure: str, freq: str | None = None) -> Optional[TS]:
    if not ds.has_time or measure not in ds.cols:
        return None
    freq = freq or choose_freq(ds)
    kind = ds.kind(measure)
    frame = pd.DataFrame({"t": ds.times(), "v": ds.values(measure)}).dropna(subset=["t"])
    if frame["v"].notna().sum() < 3:
        return None
    per = frame["t"].dt.to_period(freq)
    grp = frame.groupby(per)["v"]
    agg = grp.sum(min_count=1) if kind == "additive" else grp.mean()
    cnt = grp.count()
    full = pd.period_range(per.min(), per.max(), freq=freq)
    agg, cnt = agg.reindex(full), cnt.reindex(full).fillna(0)
    gap_share = float((cnt == 0).mean())
    if kind == "additive":
        agg = agg.fillna(0)

    partial_last, dropped_first = None, False
    coarse = ds.time_spec["kind"] != "datetime" or (
        freq in ("M", "Q", "Y") and (frame["t"].dt.day == 1).mean() > 0.95
    )
    if not coarse and len(full) >= 3:
        last = full[-1]
        span = (last.end_time - last.start_time).total_seconds()
        coverage = max(0.0, min(1.0, (frame["t"].max() - last.start_time).total_seconds() / span))
        if coverage < 0.85:
            partial_last = {"start": last.start_time, "value": float(agg.iloc[-1]), "coverage": coverage}
            agg, cnt = agg.iloc[:-1], cnt.iloc[:-1]
        first = full[0]
        fcov = (first.end_time - frame["t"].min()).total_seconds() / (first.end_time - first.start_time).total_seconds()
        if fcov < 0.85 and kind == "additive" and len(agg) > 3:
            agg, cnt, dropped_first = agg.iloc[1:], cnt.iloc[1:], True

    agg.index = agg.index.to_timestamp()
    cnt.index = cnt.index.to_timestamp()
    return TS(measure, freq, agg.astype(float), cnt, partial_last, dropped_first, gap_share, kind)


# --------------------------------------------------------------------------------------
def period_over_period(ts: TS) -> Optional[dict[str, Any]]:
    v = ts.values.dropna()
    if len(v) < 2:
        return None
    latest, prev = float(v.iloc[-1]), float(v.iloc[-2])
    pct = None if prev == 0 else (latest - prev) / abs(prev) * 100
    z, unusual = None, False
    chg = v.pct_change().replace([np.inf, -np.inf], np.nan).dropna() if (v > 0).all() else v.diff().dropna()
    if len(chg) >= 7:
        hist, last_chg = chg.iloc[:-1], chg.iloc[-1]
        if hist.std() > 0:
            z = float((last_chg - hist.mean()) / hist.std())
            unusual = abs(z) >= 2
    lag = {"M": 12, "Q": 4, "W": 52, "D": 364}.get(ts.freq)
    yoy = yoy_val = yoy_label = None
    if lag and len(v) > lag and v.iloc[-lag - 1] != 0:
        yoy_val = float(v.iloc[-lag - 1])
        yoy = (latest - yoy_val) / abs(yoy_val) * 100
        yoy_label = ts.label(v.index[-lag - 1])
    return {
        "freq": ts.freq, "latest_label": ts.label(v.index[-1]), "prev_label": ts.label(v.index[-2]),
        "latest_start": v.index[-1], "prev_start": v.index[-2],
        "latest": latest, "prev": prev, "change": latest - prev, "pct": pct, "z": z, "unusual": unusual,
        "yoy_pct": yoy, "yoy_value": yoy_val, "yoy_label": yoy_label, "periods": int(len(v)),
    }


def trend(ts: TS) -> Optional[dict[str, Any]]:
    v = ts.values.dropna()
    if len(v) < 6 or v.nunique() < 3:
        return None
    x = np.arange(len(v), dtype=float)
    res = stats.linregress(x, v.to_numpy())
    mean = float(v.mean())
    per_period_pct = None if mean == 0 else float(res.slope / abs(mean) * 100)
    significant = bool(res.pvalue < 0.05 and res.rvalue ** 2 >= 0.2)
    cv = float(v.std() / abs(mean)) if mean else None
    return {
        "slope": float(res.slope), "p_value": float(res.pvalue), "r2": float(res.rvalue ** 2),
        "pct_per_period": per_period_pct, "significant": significant, "periods": int(len(v)),
        "direction": "up" if res.slope > 0 else "down", "volatility_cv": cv,
        "first": float(v.iloc[0]), "last": float(v.iloc[-1]),
        "total_change_pct": None if v.iloc[0] == 0 else float((v.iloc[-1] - v.iloc[0]) / abs(v.iloc[0]) * 100),
    }


def seasonality(ts: TS) -> Optional[dict[str, Any]]:
    m = _SEASON_LEN.get(ts.freq)
    y = ts.values.interpolate().dropna()
    if m is None or ts.gap_share > 0.3:
        return None
    need = {"M": 24, "Q": 12, "D": 56}[ts.freq]
    if len(y) < need:
        return None
    ma = y.rolling(m).mean()
    trend_c = ma.rolling(2).mean().shift(-(m // 2)) if m % 2 == 0 else ma.shift(-((m - 1) // 2))
    detr = (y - trend_c).dropna()
    if len(detr) < m * 2 or detr.var() == 0:
        return None
    if ts.freq == "M":
        pos = detr.index.month
    elif ts.freq == "Q":
        pos = detr.index.quarter
    else:
        pos = detr.index.dayofweek
    idx = detr.groupby(pos).mean()
    idx = idx - idx.mean()
    resid = detr - idx.reindex(pos).to_numpy()
    strength = float(max(0.0, 1 - resid.var() / detr.var()))
    base = float(y.mean()) or 1.0
    names = (lambda k: month_name(k)) if ts.freq == "M" else ((lambda k: f"Q{k}") if ts.freq == "Q" else (lambda k: weekday_name(k)))
    peak, trough = idx.idxmax(), idx.idxmin()
    return {
        "freq": ts.freq, "strength": strength, "meaningful": bool(strength >= 0.3),
        "peak": names(peak), "trough": names(trough),
        "peak_pct": float(idx.max() / abs(base) * 100), "trough_pct": float(idx.min() / abs(base) * 100),
        "profile": pd.DataFrame({"position": [names(k) for k in idx.index], "effect_pct": (idx / abs(base) * 100).to_numpy()}),
        "season_length": m, "effect_by_pos": idx,
    }


def period_anomalies(ts: TS, season: Optional[dict] = None, z_cut: float = 3.5) -> list[dict[str, Any]]:
    v = ts.values.dropna()
    if len(v) < 8:
        return []
    adj = v.copy()
    if season and season.get("meaningful"):  # judge each period against its own seasonal expectation
        pos = v.index.month if ts.freq == "M" else (v.index.quarter if ts.freq == "Q" else v.index.dayofweek)
        adj = v - season["effect_by_pos"].reindex(pos).to_numpy()
    base = adj.rolling(5, center=True, min_periods=3).median()
    resid = adj - base
    mad = float(np.median(np.abs(resid - resid.median())))
    if mad == 0:
        return []
    z = 0.6745 * (resid - resid.median()) / mad
    out = []
    for ts_, zz in z.items():
        if abs(zz) >= z_cut:
            out.append({"period": ts.label(ts_), "start": ts_, "value": float(v.loc[ts_]),
                        "expected": float(v.loc[ts_] - resid.loc[ts_]), "z": float(zz),
                        "direction": "high" if zz > 0 else "low"})
    return sorted(out, key=lambda d: -abs(d["z"]))[:5]


# --------------------------------------------------------------------------------------
def _fit_predict(name: str, train: pd.Series, steps: int, m: int | None) -> np.ndarray:
    y = train.to_numpy(dtype=float)
    if name == "naive":
        return np.repeat(y[-1], steps)
    if name == "linear":
        x = np.arange(len(y))
        b, a = np.polyfit(x, y, 1)
        return a + b * np.arange(len(y), len(y) + steps)
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "holt":
            fit = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit()
        else:
            fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=m,
                                       initialization_method="estimated").fit()
        return np.asarray(fit.forecast(steps))


def forecast(ts: TS, season: Optional[dict] = None, horizon: Optional[int] = None) -> dict[str, Any]:
    unit = unit_word(ts.freq)
    y = ts.values.interpolate().dropna()
    n = len(y)
    need = _MIN_HISTORY.get(ts.freq, 12)
    if ts.gap_share > 0.3:
        return {"available": False, "reason": "Forecasting is unavailable because more than 30% of periods contain no data."}
    if n < need:
        return {"available": False,
                "reason": f"Forecasting is unavailable: it needs at least {need} complete {unit}s of history, and only {n} are available."}
    if y.nunique() < 4:
        return {"available": False, "reason": "Forecasting is unavailable because the series has almost no variation."}

    h = int(min(max(horizon, 1), 24)) if horizon else _HORIZON[ts.freq]
    m = season["season_length"] if season and season.get("meaningful") else None
    hold = int(min(max(3, n // 5), 12))
    train, test = y.iloc[:-hold], y.iloc[-hold:]
    candidates = ["naive", "linear", "holt"] + (["holt_winters"] if m and len(train) >= 2 * m + 2 else [])
    errors: dict[str, float] = {}
    preds: dict[str, np.ndarray] = {}
    for name in candidates:
        try:
            p = _fit_predict(name, train, hold, m)
            preds[name] = p
            errors[name] = float(np.mean(np.abs(test.to_numpy() - p)))
        except Exception:
            continue
    if not errors:
        return {"available": False, "reason": "Forecasting could not be validated on this series."}
    best = min(errors, key=errors.get)
    err = test.to_numpy() - preds[best]
    denom = np.abs(test.to_numpy())
    mape = float(np.mean(np.abs(err) / denom)) if (denom > 0).all() else None
    rmse = float(np.sqrt(np.mean(err ** 2)))
    skill = None if errors["naive"] == 0 else float(1 - errors[best] / errors["naive"])
    if mape is None:
        reliability = "Medium" if skill and skill > 0.1 else "Low"
    else:
        reliability = "High" if mape <= 0.15 else ("Medium" if mape <= 0.30 else "Low")

    point = _fit_predict(best, y, h, m)
    k = np.arange(1, h + 1)
    band = 1.28 * rmse * np.sqrt(1 + 0.25 * (k - 1))
    lo, hi = point - band, point + band
    if (y >= 0).all():
        point, lo, hi = np.maximum(point, 0), np.maximum(lo, 0), np.maximum(hi, 0)
    idx = pd.date_range(y.index[-1], periods=h + 1, freq=_pd_freq(ts.freq))[1:]
    fdf = pd.DataFrame({"period": idx, "forecast": point, "lower": lo, "upper": hi})
    recent = float(y.iloc[-min(h, n):].mean())
    fut = float(point.mean())
    names = {"naive": "naive baseline", "linear": "linear trend", "holt": "damped-trend exponential smoothing",
             "holt_winters": "seasonal exponential smoothing"}
    return {
        "available": True, "freq": ts.freq, "horizon": h, "frame": fdf, "model": names[best],
        "backtest_mape": mape, "backtest_periods": hold, "skill_vs_naive": skill, "reliability": reliability,
        "future_mean": fut, "recent_mean": recent,
        "change_pct": None if recent == 0 else (fut - recent) / abs(recent) * 100,
        "history_periods": n,
    }


def _pd_freq(freq: str) -> str:
    return {"M": "MS", "Q": "QS", "Y": "YS", "W": "W-MON", "D": "D"}.get(freq, "MS")
