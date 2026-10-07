"""Record-level and period-level anomaly detection with human-readable reasons.

Records are described as *unusual*, never as errors: whether they are genuine,
mistaken or fraudulent is a judgement for the data owner.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .dataset import Dataset
from .fmt import num


def _robust_z(x: pd.Series) -> tuple[pd.Series, bool]:
    xs = x.dropna()
    logged = bool((xs > 0).all() and xs.skew() > 1.0)
    base = np.log(x.where(x > 0)) if logged else x
    med = base.median()
    mad = float(np.median(np.abs(base.dropna() - med)))
    if mad == 0:
        sd = float(base.std())
        return (pd.Series(0.0, index=x.index), logged) if sd == 0 else ((base - med) / sd, logged)
    return 0.6745 * (base - med) / mad, logged


def detect_anomalies(ds: Dataset, max_rows: int = 25) -> dict[str, Any]:
    df = ds.df
    n = len(df)
    feats = [m for m in ds.measures if ds.cols[m].nunique >= 8][:12]
    if n < 30 or not feats:
        return {"available": False, "reason": "Anomaly detection needs at least 30 records and a continuous numeric field."}

    zs: dict[str, pd.Series] = {}
    reasons: dict[Any, list[str]] = {}
    by_field: dict[str, int] = {}
    severity = pd.Series(0.0, index=df.index)

    def add(idx, text):
        reasons.setdefault(idx, []).append(text)

    for c in feats:
        x = ds.values(c)
        if x.notna().sum() < 20:
            continue
        z, logged = _robust_z(x)
        zs[c] = z
        med = float(x.median())
        flagged = z.abs() >= 3.5
        by_field[ds.label(c)] = int(flagged.sum())
        severity = severity.where(~flagged, np.maximum(severity, z.abs()))
        for idx in z.index[flagged]:
            val = float(x.loc[idx])
            if z.loc[idx] > 0:
                add(idx, f"{ds.label(c)} of {num(val)} is {num(val / med, 1)}x the typical value ({num(med)})" if med > 0 and val > 0
                    else f"{ds.label(c)} of {num(val)} is far above typical ({num(med)})")
            else:
                add(idx, f"{ds.label(c)} of {num(val)} is far below typical ({num(med)})")
        clean = x.dropna()
        if (clean >= 0).mean() >= 0.99 and (clean < 0).any():
            for idx in x.index[x < 0]:
                add(idx, f"negative {ds.label(c)} ({num(float(x.loc[idx]))}) in an otherwise non-negative field")
                severity.loc[idx] = max(severity.loc[idx], 4.0)

    # multivariate: unusual combinations
    if len(zs) >= 2 and n >= 50:
        from sklearn.ensemble import IsolationForest

        M = pd.DataFrame(zs).clip(-8, 8).fillna(0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iso = IsolationForest(n_estimators=150, contamination=min(0.03, max(0.005, 5 / n)), random_state=0)
            fit_idx = M.sample(min(len(M), 50000), random_state=0).index
            iso.fit(M.loc[fit_idx])
            lab = iso.predict(M)
        for idx in M.index[lab == -1]:
            if idx not in reasons:
                top = M.loc[idx].abs().sort_values(ascending=False).index[:2]
                add(idx, "unusual combination of " + " and ".join(ds.label(t) for t in top))
                severity.loc[idx] = max(severity.loc[idx], 2.5)

    flagged_idx = list(reasons.keys())
    out: dict[str, Any] = {"available": True, "total": len(flagged_idx), "share_pct": len(flagged_idx) / n * 100,
                           "by_field": {k: v for k, v in by_field.items() if v}, "period": []}
    if not flagged_idx:
        out["table"] = pd.DataFrame()
        out["high"] = 0
        return out
    order = severity.loc[flagged_idx].sort_values(ascending=False).index[:max_rows]
    show_cols = []
    for c in ds.id_columns[:1] + ([ds.time_spec["col"]] if ds.time_spec and ds.time_spec["kind"] == "datetime" else []) \
            + ds.categorical_dimensions[:2] + list(zs)[:3]:
        if c not in show_cols:
            show_cols.append(c)
    tbl = df.loc[order, show_cols].copy()
    tbl.columns = [ds.label(c) for c in show_cols]
    tbl.insert(0, "Record", [int(i) + 1 for i in order])
    tbl["Why it stands out"] = ["; ".join(reasons[i][:2]) for i in order]
    tbl["Priority"] = ["High" if severity.loc[i] >= 6 else "Medium" for i in order]
    out["table"] = tbl.reset_index(drop=True)
    out["high"] = int((severity.loc[flagged_idx] >= 6).sum())
    out["priority_records"] = order.tolist()
    return out
