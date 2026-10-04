from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def segment_dataset(df: pd.DataFrame, max_rows: int = 5000) -> dict[str, Any]:
    numeric = df.select_dtypes(include="number").columns.tolist()
    if len(numeric) < 2 or len(df) < 12:
        return {"available": False, "reason": "Segmentation requires at least 12 rows and 2 numeric features."}

    usable = [c for c in numeric if df[c].nunique(dropna=True) >= 3]
    if len(usable) < 2:
        return {"available": False, "reason": "Not enough variable numeric features for meaningful segmentation."}

    sample = df[usable].copy()
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=42)

    imputed = SimpleImputer(strategy="median").fit_transform(sample)
    scaled = StandardScaler().fit_transform(imputed)

    k = min(5, max(3, len(sample) // 500))
    k = min(k, len(sample) - 1)
    if k < 2:
        return {"available": False, "reason": "Insufficient observations for clustering."}

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)
    segmented = sample.copy()
    segmented["segment"] = labels

    counts = segmented["segment"].value_counts().sort_index()
    profiles = []
    for segment_id in sorted(counts.index):
        subset = segmented[segmented["segment"] == segment_id]
        profile = {
            "segment": int(segment_id) + 1,
            "records": int(len(subset)),
            "share_pct": round(float(len(subset) / len(segmented) * 100), 2),
        }
        for col in usable[:8]:
            profile[col] = round(float(subset[col].mean()), 3)
        profiles.append(profile)

    return {
        "available": True,
        "features": usable,
        "clusters": k,
        "sampled_rows": len(sample),
        "profiles": profiles,
        "note": "Segments are statistical clusters, not pre-defined business personas.",
    }
