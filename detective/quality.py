"""Dataset health assessment: completeness, uniqueness and validity."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dataset import Dataset
from .fmt import num, pct


def _band(score: float) -> str:
    return "Excellent" if score >= 90 else "Good" if score >= 75 else "Fair" if score >= 60 else "Needs attention"


def assess_quality(ds: Dataset) -> dict[str, Any]:
    df = ds.df
    n, m = len(df), df.shape[1]
    total = max(n * m, 1)
    missing = int(df.isna().sum().sum())
    missing_pct = missing / total * 100
    dup_rows = int(df.duplicated().sum())
    dup_pct = dup_rows / max(n, 1) * 100

    issues: list[dict[str, str]] = []
    id_dups = 0
    for c in ds.id_columns:
        d = int(df[c].dropna().duplicated().sum())
        if d:
            id_dups += d
            issues.append({"severity": "high", "title": f"Repeated identifiers in {ds.label(c)}",
                           "detail": f"{d:,} records reuse an existing {ds.label(c)} value.",
                           "action": f"Confirm whether {ds.label(c)} is meant to be unique; repeats can double-count activity."})
    if dup_rows:
        issues.append({"severity": "high" if dup_pct >= 1 else "medium", "title": "Fully duplicated records",
                       "detail": f"{dup_rows:,} rows ({pct(dup_pct)}) are exact copies of another row.",
                       "action": "Remove or reconcile duplicates before reporting totals; they inflate counts and sums."})

    col_rows = []
    validity_flags = 0
    for c in df.columns:
        s = df[c]
        info = ds.cols[c]
        miss = int(s.isna().sum())
        mp = miss / max(n, 1) * 100
        flags: list[str] = []
        if mp >= 30:
            flags.append("high missingness")
            issues.append({"severity": "high", "title": f"{ds.label(c)} is largely empty",
                           "detail": f"{pct(mp)} of values are missing.",
                           "action": f"Decide whether {ds.label(c)} is reliable enough to analyse or should be excluded."})
        elif mp >= 5:
            flags.append("notable missingness")
            issues.append({"severity": "medium", "title": f"Missing values in {ds.label(c)}",
                           "detail": f"{miss:,} values ({pct(mp)}) are missing.",
                           "action": f"Check whether the gaps in {ds.label(c)} are random or tied to a specific source or period."})
        if info.role == "constant":
            flags.append("constant")
            validity_flags += 1
            issues.append({"severity": "low", "title": f"{ds.label(c)} has a single value",
                           "detail": "A column with no variation carries no analytical signal.",
                           "action": "Consider excluding it from analysis."})
        if info.role == "measure":
            x = pd.to_numeric(s, errors="coerce").dropna()
            if len(x) > 20 and (x >= 0).mean() >= 0.99 and (x < 0).any():
                k = int((x < 0).sum())
                flags.append("unexpected negatives")
                validity_flags += 1
                issues.append({"severity": "medium", "title": f"Unexpected negative values in {ds.label(c)}",
                               "detail": f"{k:,} negative value(s) in a field that is otherwise non-negative.",
                               "action": "Check whether these are refunds, reversals or entry errors."})
        if info.role == "dimension":
            raw = s.dropna().astype(str)
            norm = raw.str.strip().str.lower()
            if raw.nunique() > norm.nunique():
                flags.append("inconsistent labels")
                validity_flags += 1
                issues.append({"severity": "low", "title": f"Inconsistent labels in {ds.label(c)}",
                               "detail": "Some categories differ only by capitalisation or spacing.",
                               "action": "Standardise labels so groups are not split."})
        col_rows.append({
            "Field": ds.label(c), "Role": info.role.replace("_", " ").title(), "Missing": miss,
            "Missing %": round(mp, 1), "Distinct values": int(info.nunique), "Flags": ", ".join(flags) or "—",
        })

    t = ds.times()
    if t is not None and t.notna().any() and t.max() > pd.Timestamp.now() + pd.Timedelta(days=1):
        validity_flags += 1
        issues.append({"severity": "medium", "title": "Future-dated records",
                       "detail": f"The latest date ({t.max():%d %b %Y}) is after today.",
                       "action": "Confirm whether these are forecasts, scheduled items or date-entry errors."})

    completeness = max(0.0, 100 - 2 * missing_pct)
    uniqueness = max(0.0, 100 - 4 * dup_pct - 3 * (id_dups / max(n, 1) * 100))
    validity = max(0.0, 100 - 12 * validity_flags)
    score = round(0.4 * completeness + 0.3 * uniqueness + 0.3 * validity, 1)

    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: order[i["severity"]])
    return {
        "score": score, "band": _band(score), "rows": n, "columns": m,
        "completeness": round(completeness, 1), "uniqueness": round(uniqueness, 1), "validity": round(validity, 1),
        "missing_cells": missing, "missing_pct": round(missing_pct, 2),
        "duplicate_rows": dup_rows, "duplicate_pct": round(dup_pct, 2), "id_duplicates": id_dups,
        "issues": issues, "columns_table": pd.DataFrame(col_rows),
    }
