"""Driver screening (what is associated with the target) and correlation analysis.

Drivers are ranked by out-of-sample permutation importance of a gradient-boosted
model, then each top driver is described with a plain association statistic.
These are associations, never causal claims, and the module says so when the
model has little predictive structure.
"""
from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .dataset import Dataset


def _feature_frame(ds: Dataset, target: str, y: pd.Series):
    X, cat_mask, kinds, redundant = {}, [], {}, []
    for c in ds.cols.values():
        if c.name == target or c.role not in ("measure", "binary", "dimension"):
            continue
        if c.role == "dimension":
            X[c.name] = ds.dim(c.name).astype("category").cat.codes.astype(int)
            cat_mask.append(True)
            kinds[c.name] = "category"
        else:
            x = ds.values(c.name)
            if x.nunique() < 2:
                continue
            both = pd.concat([y, x], axis=1).dropna()
            if len(both) > 10 and abs(both.iloc[:, 0].corr(both.iloc[:, 1])) >= 0.98:
                redundant.append(c.name)
                continue
            X[c.name] = x
            cat_mask.append(False)
            kinds[c.name] = "numeric"
    return pd.DataFrame(X), cat_mask, kinds, redundant


def _assoc_pvalue(ds: Dataset, feat: str, kind: str, y_all: pd.Series) -> float:
    try:
        if kind == "numeric":
            both = pd.concat([y_all, ds.values(feat)], axis=1).dropna()
            if both.iloc[:, 1].nunique() < 2:
                return 1.0
            return float(stats.spearmanr(both.iloc[:, 1], both.iloc[:, 0]).pvalue)
        frame = pd.DataFrame({"g": ds.dim(feat), "y": y_all}).dropna()
        frame = frame[frame["g"] != "(missing)"]
        groups = [g["y"].to_numpy() for _, g in frame.groupby("g") if len(g) >= 5]
        return float(stats.kruskal(*groups).pvalue) if len(groups) >= 2 else 1.0
    except Exception:
        return 1.0


def analyze_drivers(ds: Dataset, target: str, max_rows: int = 20000, top_n: int = 6) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    y_all = ds.values(target)
    is_binary = ds.cols[target].role == "binary"
    X, cat_mask, kinds, redundant = _feature_frame(ds, target, y_all)
    valid = y_all.notna()
    if X.shape[1] < 1 or valid.sum() < 60:
        return {"available": False, "reason": "Driver analysis needs at least 60 records and one other usable field."}
    X, y = X.loc[valid], y_all.loc[valid]
    if len(X) > max_rows:
        idx = X.sample(max_rows, random_state=0).index
        X, y = X.loc[idx], y.loc[idx]
    y_fit = y if is_binary else y.clip(y.quantile(0.005), y.quantile(0.995))   # extreme records must not drive the model
    Xtr, Xte, ytr, yte = train_test_split(X, y_fit, test_size=0.25, random_state=0)
    params = dict(max_iter=120, learning_rate=0.06, max_depth=6, min_samples_leaf=25, l2_regularization=1.0,
                  random_state=0, categorical_features=cat_mask if any(cat_mask) else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_binary:
            model = HistGradientBoostingClassifier(**params).fit(Xtr, ytr.astype(int))
            from sklearn.metrics import roc_auc_score
            score = float(roc_auc_score(yte.astype(int), model.predict_proba(Xte)[:, 1])) if yte.nunique() > 1 else 0.5
            predictive, metric = score >= 0.60, "AUC"
            pi = permutation_importance(model, Xte, yte.astype(int), n_repeats=5, random_state=0, scoring="roc_auc")
        else:
            model = HistGradientBoostingRegressor(**params).fit(Xtr, ytr)
            score = float(model.score(Xte, yte))
            predictive, metric = score >= 0.10, "R2"
            pi = permutation_importance(model, Xte, yte, n_repeats=5, random_state=0, scoring="r2")
    imp = pd.Series(np.clip(pi.importances_mean, 0, None), index=X.columns)
    total = float(imp.sum())
    result: dict[str, Any] = {"available": True, "target": target, "score": score, "metric": metric, "n": int(len(X)),
                              "predictive": bool(predictive and total > 0), "redundant": redundant, "drivers": []}
    if total == 0:
        return result
    for feat in imp.sort_values(ascending=False).index[:top_n + 3]:
        share = float(imp[feat] / total * 100)
        p = _assoc_pvalue(ds, feat, kinds[feat], y_all)
        if share < 3 or (not result["predictive"] and p >= 0.01) or p >= 0.05:
            continue                      # a driver must be statistically visible on its own, too
        item: dict[str, Any] = {"feature": feat, "label": ds.label(feat), "importance_pct": share,
                                "type": kinds[feat], "p_value": p}
        if kinds[feat] == "numeric":
            x = ds.values(feat)
            both = pd.concat([y_all, x], axis=1).dropna()
            both.columns = ["y", "x"]
            rho = float(stats.spearmanr(both["x"], both["y"]).statistic) if both["x"].nunique() > 1 else 0.0
            item["rho"] = rho
            item["direction"] = "positive" if rho > 0 else "negative"
            if both["x"].nunique() >= 4:
                q = pd.qcut(both["x"].rank(method="first"), 4, labels=False)
                item.update({"low_quartile_mean": float(both.loc[q == 0, "y"].mean()),
                             "high_quartile_mean": float(both.loc[q == 3, "y"].mean())})
        else:
            frame = pd.DataFrame({"g": ds.dim(feat), "y": y_all}).dropna()
            frame = frame[frame["g"] != "(missing)"]
            grp = frame.groupby("g")["y"].agg(["mean", "size"])
            grp = grp[grp["size"] >= max(5, int(0.005 * len(frame)))]
            if len(grp) >= 2:
                overall = frame["y"].mean()
                ss_between = float((grp["size"] * (grp["mean"] - overall) ** 2).sum())
                ss_total = float(((frame["y"] - overall) ** 2).sum())
                item["eta2"] = ss_between / ss_total if ss_total else 0.0
                item["best"], item["best_mean"] = str(grp["mean"].idxmax()), float(grp["mean"].max())
                item["worst"], item["worst_mean"] = str(grp["mean"].idxmin()), float(grp["mean"].min())
        result["drivers"].append(item)
        if len(result["drivers"]) >= top_n:
            break
    return result


def correlations(ds: Dataset, max_cols: int = 14, max_rows: int = 50000) -> Optional[dict[str, Any]]:
    cols = [m for m in ds.all_measures if ds.values(m).nunique() > 2 or ds.cols[m].role == "binary"][:max_cols]
    if len(cols) < 2:
        return None
    frame = pd.DataFrame({c: ds.values(c) for c in cols})
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=0)
    pear = frame.corr(method="pearson")
    spear = frame.corr(method="spearman")
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r, rs = pear.loc[a, b], spear.loc[a, b]
            if pd.isna(r):
                continue
            pairs.append({"a": a, "b": b, "label_a": ds.label(a), "label_b": ds.label(b), "r": float(r),
                          "spearman": float(rs), "nonlinear": bool(abs(rs) - abs(r) > 0.15)})
    pairs.sort(key=lambda d: -abs(d["r"]))
    matrix = pear.rename(index=ds.label, columns=ds.label)
    return {"matrix": matrix, "pairs": pairs, "strong": [p for p in pairs if abs(p["r"]) >= 0.5],
            "near_duplicates": [p for p in pairs if abs(p["r"]) >= 0.95]}
