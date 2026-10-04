from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis import profile_dataset, probable_target_columns
from anomaly import detect_anomalies
from segmentation import segment_dataset


def build_evidence(df: pd.DataFrame) -> dict[str, Any]:
    profile = profile_dataset(df)
    anomalies = detect_anomalies(df)
    evidence: dict[str, Any] = {
        "profile": profile,
        "anomalies": {
            col: {"method": item["method"], "count": item["count"]}
            for col, item in anomalies.items()
        },
        "metrics": [],
        "trends": [],
        "drivers": [],
        "segments": {},
        "concentration": [],
    }

    for col in profile["numeric_columns"][:12]:
        s = df[col].dropna()
        if s.empty:
            continue
        evidence["metrics"].append({
            "column": col,
            "mean": float(s.mean()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
        })

    # Time-series evidence: only when a usable date and measure exist.
    if profile["datetime_columns"] and profile["numeric_columns"]:
        date_col = profile["datetime_columns"][0]
        target = probable_target_columns(df)[0]
        x = df[[date_col, target]].dropna().sort_values(date_col)
        if len(x) >= 6:
            monthly = x.set_index(date_col)[target].resample("ME").mean().dropna()
            if len(monthly) >= 4:
                first = float(monthly.iloc[0])
                last = float(monthly.iloc[-1])
                change = None if first == 0 else (last - first) / abs(first) * 100
                evidence["trends"].append({
                    "date_column": date_col,
                    "measure": target,
                    "first_period_mean": first,
                    "last_period_mean": last,
                    "period_change_pct": None if change is None else round(float(change), 2),
                    "periods": int(len(monthly)),
                })

    # Driver evidence: correlations to the strongest probable target.
    if len(profile["numeric_columns"]) >= 2:
        target = probable_target_columns(df)[0]
        for col in profile["numeric_columns"]:
            if col == target:
                continue
            r = df[target].corr(df[col])
            if not pd.isna(r):
                evidence["drivers"].append({
                    "target": target,
                    "feature": col,
                    "correlation": round(float(r), 3),
                    "absolute_strength": round(abs(float(r)), 3),
                })
        evidence["drivers"] = sorted(evidence["drivers"], key=lambda x: x["absolute_strength"], reverse=True)[:8]

    # Statistical segmentation evidence.
    segmentation = segment_dataset(df)
    if segmentation.get("available"):
        evidence["segments"] = segmentation

    # Lightweight concentration evidence for categorical dimensions.
    for col in profile["categorical_columns"][:8]:
        counts = df[col].value_counts(dropna=False)
        if len(counts) >= 2:
            top = int(counts.iloc[0])
            share = top / max(len(df), 1) * 100
            evidence["concentration"].append({
                "dimension": col,
                "top_value": str(counts.index[0]),
                "top_count": top,
                "top_share_pct": round(share, 2),
            })

    return evidence
