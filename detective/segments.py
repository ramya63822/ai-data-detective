"""Statistical segmentation, offered only when the data shows genuinely separable groups."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .dataset import Dataset


def segment_dataset(ds: Dataset, max_rows: int = 5000) -> dict[str, Any]:
    feats = [m for m in ds.all_measures if ds.cols[m].nunique >= 5 and ds.values(m).notna().mean() >= 0.8][:10]
    n = len(ds.df)
    if len(feats) < 2 or n < 60:
        return {"available": False,
                "reason": "Segmentation needs at least 60 records and two varied numeric fields."}
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X = pd.DataFrame({f: ds.values(f) for f in feats})
    for f in feats:
        s = X[f].dropna()
        if len(s) and (s > 0).all() and abs(s.skew()) > 1.5:
            X[f] = np.log1p(X[f])
    X = X.fillna(X.median())
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=0)
    Z = np.clip(StandardScaler().fit_transform(X), -3, 3)   # outliers must not form their own 'segment'
    best = (-1.0, None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(2, 7):
            km = KMeans(n_clusters=k, random_state=0, n_init=5).fit(Z)
            if np.bincount(km.labels_).min() < max(10, 0.03 * len(Z)):
                continue                                  # tiny clusters are not actionable segments
            sil = float(silhouette_score(Z, km.labels_, sample_size=min(2000, len(Z)), random_state=0))
            if sil > best[0]:
                best = (sil, k, km.labels_)
    sil, k, labels = best
    if k is None:
        return {"available": False, "reason": "No stable segments of meaningful size were found."}
    if sil < 0.25:
        return {"available": False, "silhouette": sil,
                "reason": f"No well-separated segments were found (best cluster separation {sil:.2f}, below the 0.25 threshold), so none are reported."}

    raw = pd.DataFrame({f: ds.values(f).loc[X.index] for f in feats})
    primary = ds.primary
    if primary and primary not in raw.columns:
        raw[primary] = ds.values(primary).loc[X.index]     # outcome flags are profiled but not clustered on
    raw["segment"] = labels
    z = pd.DataFrame(Z, index=X.index, columns=feats)
    z["segment"] = labels
    profiles = []
    for sid in sorted(set(labels)):
        sub_z = z[z["segment"] == sid].drop(columns="segment").mean()
        dist = sub_z.reindex(sub_z.abs().sort_values(ascending=False).index)
        traits = [(f, float(v)) for f, v in dist.items() if abs(v) >= 0.5][:3]
        desc = ", ".join(f"{'higher' if v > 0 else 'lower'} {ds.label(f)}" for f, v in traits) or "close to the overall average"
        size = int((labels == sid).sum())
        row = {"id": int(sid) + 1, "records": size, "share_pct": size / len(labels) * 100, "description": desc,
               "traits": traits}
        if primary:
            overall = float(raw[primary].mean())
            seg_mean = float(raw.loc[raw["segment"] == sid, primary].mean())
            row.update({"primary_mean": seg_mean, "primary_index": seg_mean / overall if overall else np.nan})
        for f in feats[:6]:
            row[f] = float(raw.loc[raw["segment"] == sid, f].mean())
        profiles.append(row)
    profiles.sort(key=lambda r: -(r.get("primary_mean") or 0))
    for i, r in enumerate(profiles):
        r["name"] = f"Segment {chr(65 + i)}"
    pca = PCA(n_components=2, random_state=0).fit_transform(Z)
    sample = np.random.default_rng(0).choice(len(Z), size=min(2000, len(Z)), replace=False)
    name_by_id = {r["id"]: r["name"] for r in profiles}
    scatter = pd.DataFrame({"x": pca[sample, 0], "y": pca[sample, 1],
                            "segment": [name_by_id[int(labels[i]) + 1] for i in sample]})
    return {"available": True, "k": int(k), "silhouette": sil, "features": feats, "profiles": profiles,
            "scatter": scatter, "rows_used": int(len(Z))}
