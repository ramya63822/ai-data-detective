"""Segment performance, contribution and change-decomposition analytics.

The change decomposition is exact (components sum to the observed change), which
is what lets the copilot answer "why did X change?" with verified evidence
instead of speculation.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .dataset import Dataset
from .timeseries import TS, period_over_period


# --------------------------------------------------------------------------------------
# Group tables, rankings, contribution
# --------------------------------------------------------------------------------------
def min_group_size(n: int) -> int:
    return max(5, int(0.005 * n))


def group_table(ds: Dataset, dim: str, measure: str) -> pd.DataFrame:
    kind = ds.kind(measure)
    frame = pd.DataFrame({"g": ds.dim(dim), "v": ds.values(measure)})
    frame = frame[frame["v"].notna()]
    if frame.empty:
        return pd.DataFrame()
    grp = frame.groupby("g")["v"]
    out = pd.DataFrame({"records": grp.size(), "total": grp.sum(), "avg": grp.mean()})
    overall_avg = float(frame["v"].mean())
    out["value"] = out["total"] if kind == "additive" else out["avg"]
    denom = out["total"].sum() if kind == "additive" else out["records"].sum()
    basis = out["total"] if kind == "additive" else out["records"]
    out["share"] = basis / denom * 100 if denom else np.nan
    out["record_share"] = out["records"] / out["records"].sum() * 100
    out["index"] = out["avg"] / overall_avg if overall_avg else np.nan
    out["eligible"] = out["records"] >= min_group_size(len(frame))
    out.index.name = "group"
    return out.reset_index().sort_values("value", ascending=False).reset_index(drop=True)


def rank_groups(ds: Dataset, dim: str, measure: str, lowest: bool = False, top_n: int = 10) -> pd.DataFrame:
    t = group_table(ds, dim, measure)
    if t.empty:
        return t
    pool = t[t["eligible"]] if t["eligible"].sum() >= 2 else t
    pool = pool[pool["group"] != "(missing)"] if (pool["group"] != "(missing)").sum() >= 2 else pool
    return pool.sort_values("value", ascending=lowest).head(top_n).reset_index(drop=True)


def contribution(ds: Dataset, dim: str, measure: str) -> Optional[dict[str, Any]]:
    t = group_table(ds, dim, measure)
    t = t[t["group"] != "(missing)"] if (t["group"] != "(missing)").sum() >= 2 else t
    if len(t) < 2:
        return None
    basis = t.sort_values("value", ascending=False)
    shares = basis["share"].to_numpy() / 100
    shares = shares / shares.sum() if shares.sum() else shares
    cum = np.cumsum(shares)
    hhi = float((shares ** 2).sum())
    k = len(shares)
    hhi_norm = (hhi - 1 / k) / (1 - 1 / k) if k > 1 else 1.0
    return {
        "table": basis.reset_index(drop=True), "groups": k,
        "top1_share": float(shares[0] * 100), "top1": str(basis.iloc[0]["group"]),
        "top3_share": float(shares[:3].sum() * 100), "pareto_groups": int(np.searchsorted(cum, 0.8) + 1),
        "hhi_norm": float(hhi_norm), "basis": "total" if ds.kind(measure) == "additive" else "records",
    }


def segment_gaps(ds: Dataset, dim: str, measure: str) -> pd.DataFrame:
    """Groups whose average per-record performance differs materially from the overall average."""
    t = group_table(ds, dim, measure)
    if t.empty:
        return t
    t = t[t["eligible"] & (t["group"] != "(missing)")].copy()
    if len(t) < 2:
        return t.iloc[0:0]
    overall = float(ds.values(measure).mean())
    t["gap_value"] = (overall - t["avg"]) * t["records"]  # shortfall (+) or surplus (-) vs. average performance
    v = pd.DataFrame({"g": ds.dim(dim), "v": ds.values(measure)}).dropna()
    pvals = []
    for g in t["group"]:
        a, b = v.loc[v["g"] == g, "v"], v.loc[v["g"] != g, "v"]
        pvals.append(float(stats.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 2 and len(b) > 2 else 1.0)
    t["p_value"] = pvals
    return t


# --------------------------------------------------------------------------------------
# Change decomposition
# --------------------------------------------------------------------------------------
def period_masks(ds: Dataset, ts: TS, pop: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    per = ds.times().dt.to_period(ts.freq)
    p1 = pd.Period(pop["latest_start"], freq=ts.freq)
    p0 = pd.Period(pop["prev_start"], freq=ts.freq)
    return per == p0, per == p1


def decompose_change(ds: Dataset, measure: str, dim: str, m0: pd.Series, m1: pd.Series,
                     key: pd.Series | None = None) -> Optional[pd.DataFrame]:
    """Exact attribution of the change between two record sets to groups of `dim` (or a custom key)."""
    kind = ds.kind(measure)
    g = key if key is not None else ds.dim(dim)
    v = ds.values(measure)
    f0 = pd.DataFrame({"g": g[m0], "v": v[m0]}).dropna(subset=["v"])
    f1 = pd.DataFrame({"g": g[m1], "v": v[m1]}).dropna(subset=["v"])
    if f0.empty or f1.empty:
        return None
    a0, a1 = f0.groupby("g")["v"], f1.groupby("g")["v"]
    groups = sorted(set(f0["g"]) | set(f1["g"]))
    t = pd.DataFrame(index=pd.Index(groups, name="group"))
    t["n0"], t["n1"] = a0.size().reindex(groups).fillna(0), a1.size().reindex(groups).fillna(0)
    if kind == "additive":
        t["v0"], t["v1"] = a0.sum().reindex(groups).fillna(0.0), a1.sum().reindex(groups).fillna(0.0)
        t["delta"] = t["v1"] - t["v0"]
    else:
        t["v0"], t["v1"] = a0.mean().reindex(groups), a1.mean().reindex(groups)
        w0, w1 = t["n0"] / t["n0"].sum(), t["n1"] / t["n1"].sum()
        M0 = float(f0["v"].mean())
        m0_eff = t["v0"].where(t["v0"].notna(), t["v1"])  # new groups: no rate effect
        t["mix"] = (w1 - w0) * (m0_eff - M0)
        t["rate"] = (w1 * (t["v1"] - m0_eff)).where(t["v1"].notna(), 0.0).fillna(0.0)
        t["delta"] = t["mix"] + t["rate"]
    total = float(t["delta"].sum())
    t["share_of_change"] = t["delta"] / total * 100 if total else np.nan
    t["pct_change"] = np.where(t["v0"].fillna(0) != 0, (t["v1"] - t["v0"]) / t["v0"].abs() * 100, np.nan)
    if kind == "additive":
        v0_total = float(t["v0"].sum())
        t["base_share"] = t["v0"] / v0_total * 100 if v0_total else np.nan
        t["excess"] = t["delta"] - total * (t["v0"] / v0_total if v0_total else 0)  # movement beyond a proportional share
    else:
        t["base_share"] = t["n0"] / t["n0"].sum() * 100
        t["excess"] = t["delta"]
    t = t.reset_index()
    return t.reindex(t["delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def volume_value_effect(ds: Dataset, measure: str, m0: pd.Series, m1: pd.Series) -> Optional[dict[str, Any]]:
    if ds.kind(measure) != "additive":
        return None
    v = ds.values(measure)
    x0, x1 = v[m0].dropna(), v[m1].dropna()
    if len(x0) == 0 or len(x1) == 0:
        return None
    n0, n1, a0, a1 = len(x0), len(x1), float(x0.mean()), float(x1.mean())
    volume, value = (n1 - n0) * a0, n1 * (a1 - a0)
    return {"n0": n0, "n1": n1, "avg0": a0, "avg1": a1, "volume_effect": volume, "value_effect": value,
            "total": volume + value}


def explain_change(ds: Dataset, measure: str, ts: TS, max_dims: int = 6) -> Optional[dict[str, Any]]:
    """Multi-step root-cause evidence: size the change, test it, then attribute it across dimensions."""
    pop = period_over_period(ts)
    if pop is None:
        return None
    m0, m1 = period_masks(ds, ts, pop)
    kind = ds.kind(measure)
    out: dict[str, Any] = {"pop": pop, "measure": measure, "kind": kind, "dimensions": [], "pairs": []}
    out["volume_value"] = volume_value_effect(ds, measure, m0, m1)

    v = ds.values(measure)
    a, b = v[m0].dropna(), v[m1].dropna()
    if kind != "additive" and len(a) > 2 and len(b) > 2:
        out["p_value"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)

    net = pop["change"]
    scored = []
    core = [d for d in ds.dimensions if ds.cols[d].role != "measure"]          # categorical + flag dimensions
    filler = [d for d in ds.dimensions if ds.cols[d].role == "measure"]        # discrete numeric codes, used sparingly
    candidates = (core + filler[: max(0, 3 - len(core))])[: max_dims + 2]
    for dim in candidates:
        if dim == measure:
            continue
        t = decompose_change(ds, measure, dim, m0, m1)
        if t is None or len(t) < 2:
            continue
        known = t[t["group"] != "(missing)"]
        if len(known) < 2:
            continue
        abs_sum = float(known["delta"].abs().sum())
        if abs_sum == 0:
            continue
        if kind == "additive":      # how far does the most out-of-line group sit from a proportional share?
            focus = float(known["excess"].abs().max() / max(abs(net), 1e-12))
        else:                       # normalised top-1 concentration (0 = evenly spread)
            k = len(known)
            s1 = float(known["delta"].abs().max() / abs_sum)
            focus = (s1 - 1 / k) / (1 - 1 / k)
        unattributed = t.loc[t["group"] == "(missing)", "delta"].sum()
        aligned = known[np.sign(known["delta"]) == np.sign(net)] if net else known
        scored.append((focus, dim, t, known, aligned, float(unattributed)))
    scored.sort(key=lambda s_: -s_[0])
    for focus, dim, t, known, aligned, unattr in scored[:max_dims]:
        lead = known.reindex(known["excess"].abs().sort_values(ascending=False).index) if kind == "additive" else known
        out["dimensions"].append({"dim": dim, "table": t, "known": known, "focus": focus, "lead": lead.iloc[0].to_dict(),
                                  "aligned": aligned, "unattributed": unattr})

    # influence of single extreme records on each period (additive measures)
    if kind == "additive":
        out["influence"] = {
            "prev_top_share": float(a.max() / a.sum() * 100) if len(a) and a.sum() else None,
            "latest_top_share": float(b.max() / b.sum() * 100) if len(b) and b.sum() else None,
            "prev_top": float(a.max()) if len(a) else None, "latest_top": float(b.max()) if len(b) else None,
        }

    # pairwise drill-down on the two most concentrated dimensions
    if len(out["dimensions"]) >= 2:
        d1, d2 = out["dimensions"][0]["dim"], out["dimensions"][1]["dim"]
        key = ds.dim(d1) + " / " + ds.dim(d2)
        if key.nunique() <= 400:
            t = decompose_change(ds, measure, d1, m0, m1, key=key)
            if t is not None and len(t) >= 2:
                t = t[~t["group"].str.contains(r"\(missing\)", regex=True)]
                out["pairs"].append({"dims": (d1, d2), "table": t.reset_index(drop=True)})
    return out


# --------------------------------------------------------------------------------------
# Valuable / unusual entities
# --------------------------------------------------------------------------------------
def valuable_entities(ds: Dataset, entity: str, measure: str, top_n: int = 10) -> Optional[dict[str, Any]]:
    kind = ds.kind(measure)
    frame = pd.DataFrame({"e": ds.dim(entity), "v": ds.values(measure)}).dropna()
    frame = frame[frame["e"] != "(missing)"]
    if frame["e"].nunique() < 5:
        return None
    grp = frame.groupby("e")["v"]
    t = pd.DataFrame({"records": grp.size(), "total": grp.sum(), "avg": grp.mean()})
    t["value"] = t["total"] if kind == "additive" else t["avg"]
    if kind != "additive":
        t = t[t["records"] >= 3]
        if len(t) < 5:
            return None
    q1, q3 = t["value"].quantile([0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    med = float(t["value"].median())
    t["x_median"] = t["value"] / med if med else np.nan
    t["unusual"] = t["value"] > fence
    # robust z on log scale when strictly positive
    pos = (t["value"] > 0).all()
    base = np.log(t["value"]) if pos else t["value"]
    mad = float(np.median(np.abs(base - base.median())))
    t["robust_z"] = 0.6745 * (base - base.median()) / mad if mad else 0.0
    t["share"] = t["total"] / t["total"].sum() * 100 if kind == "additive" else np.nan
    t = t.sort_values("value", ascending=False)
    flagged = t[t["unusual"] & (t["robust_z"] >= 3.5)]
    top = t.head(top_n).reset_index().rename(columns={"e": "entity"})
    return {"table": top, "entities": int(len(t)), "flagged": flagged.reset_index().rename(columns={"e": "entity"}),
            "fence": float(fence), "median": med, "kind": kind,
            "top_share": float(t["total"].head(top_n).sum() / t["total"].sum() * 100) if kind == "additive" else None,
            "top1_share": float(t["share"].iloc[0]) if kind == "additive" else None}
