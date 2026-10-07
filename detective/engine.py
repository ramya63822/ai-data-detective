"""Analysis orchestration and evidence-grounded finding generation.

Every finding is produced by a rule that reads verified numbers computed in the
analysis modules. Recommendations ("what to investigate next") are attached to
those findings by the same rules, never invented by a language model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import anomalies as an
from . import drivers as dr
from . import performance as pf
from . import segments as sg
from . import timeseries as tsm
from .dataset import Dataset
from .fmt import join_and, num, pct, unit_word, value
from .quality import assess_quality


@dataclass
class Finding:
    key: str
    category: str            # change | trend | seasonality | forecast | concentration | performance | driver | anomaly | quality | segment | entity | distribution
    kind: str                # risk | positive | opportunity | info
    title: str
    fact: str
    importance: float
    investigate: Optional[str] = None
    measure: Optional[str] = None


@dataclass
class Analysis:
    ds: Dataset
    quality: dict[str, Any]
    kpis: list[dict[str, Any]] = field(default_factory=list)
    primary: Optional[str] = None
    ts: Optional[tsm.TS] = None
    pop: Optional[dict] = None
    trend: Optional[dict] = None
    season: Optional[dict] = None
    forecast: Optional[dict] = None
    period_anoms: list = field(default_factory=list)
    change: Optional[dict] = None
    rankings: dict[str, dict] = field(default_factory=dict)
    gaps: dict[str, pd.DataFrame] = field(default_factory=dict)
    drivers: Optional[dict] = None
    corr: Optional[dict] = None
    anomalies: dict[str, Any] = field(default_factory=dict)
    segments: dict[str, Any] = field(default_factory=dict)
    entities: Optional[dict] = None
    distributions: dict[str, dict] = field(default_factory=dict)
    hotspots: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    investigations: list[str] = field(default_factory=list)

    @property
    def risks(self) -> list[Finding]:
        return [f for f in self.findings if f.kind == "risk"]

    @property
    def opportunities(self) -> list[Finding]:
        return [f for f in self.findings if f.kind == "opportunity"]

    @property
    def positives(self) -> list[Finding]:
        return [f for f in self.findings if f.kind == "positive"]


# --------------------------------------------------------------------------------------
def tone(ds: Dataset, measure: str, change: Optional[float]) -> str:
    """Classify a movement as risk / positive / info using the measure's polarity (if known)."""
    pol = ds.polarity(measure)
    if change is None or pol == 0:
        return "info"
    return "positive" if (change > 0) == (pol > 0) else "risk"


def headline_value(ds: Dataset, measure: str) -> tuple[float, str]:
    v = ds.values(measure).dropna()
    kind = ds.kind(measure)
    if kind == "additive":
        return float(v.sum()), "Total"
    if kind == "rate":
        return float(v.mean()), "Rate"
    return float(v.mean()), "Average"


def describe_distribution(ds: Dataset, measure: str) -> Optional[dict[str, Any]]:
    v = ds.values(measure).dropna()
    if len(v) < 5:
        return None
    q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(q[0.5]), "p05": float(q[0.05]),
            "p25": float(q[0.25]), "p75": float(q[0.75]), "p95": float(q[0.95]), "min": float(v.min()),
            "max": float(v.max()), "std": float(v.std()), "skew": float(v.skew()) if len(v) > 2 else 0.0}


def _build_kpis(ds: Dataset, ts: Optional[tsm.TS], pop: Optional[dict], focus: list[str]) -> list[dict[str, Any]]:
    kpis = [{"label": "Records", "value": f"{len(ds.df):,}", "caption": "Total records", "delta": None,
             "favourable": None, "spark": None, "measure": None}]
    for m in focus:
        val, cap = headline_value(ds, m)
        k = {"label": ds.label(m), "value": value(ds.kind(m), val), "caption": cap, "delta": None,
             "favourable": None, "spark": None, "measure": m}
        t = tsm.build_ts(ds, m) if ds.has_time else None
        p = tsm.period_over_period(t) if t is not None else None
        if p and p["pct"] is not None:
            k["delta"] = f"{pct(p['pct'], signed=True)} {p['latest_label']} vs {p['prev_label']}"
            t_ = tone(ds, m, p["pct"])
            k["favourable"] = None if t_ == "info" else (t_ == "positive")
            k["spark"] = [float(x) for x in t.values.tail(14).fillna(0)]
        kpis.append(k)
    return kpis


# --------------------------------------------------------------------------------------
def analyze(ds: Dataset) -> Analysis:
    A = Analysis(ds=ds, quality=assess_quality(ds), primary=ds.primary)
    P = ds.primary
    focus = ([P] if P else []) + [m for m in ds.outcome_flags if m != P][:1] + [m for m in ds.measures if m != P][:2]
    focus = [m for i, m in enumerate(focus) if m and m not in focus[:i]][:4]
    for m in (ds.measures + ds.rate_measures)[:6]:
        d = describe_distribution(ds, m)
        if d:
            A.distributions[m] = d

    if P and ds.has_time:
        A.ts = tsm.build_ts(ds, P)
        if A.ts is not None and len(A.ts.values.dropna()) >= 2:
            A.pop = tsm.period_over_period(A.ts)
            A.trend = tsm.trend(A.ts)
            A.season = tsm.seasonality(A.ts)
            A.forecast = tsm.forecast(A.ts, A.season)
            A.period_anoms = tsm.period_anomalies(A.ts, A.season)
            if A.pop:
                A.change = pf.explain_change(ds, P, A.ts)
    A.kpis = _build_kpis(ds, A.ts, A.pop, focus)

    if P:
        for dim in ds.dimensions[:6]:
            if dim == P or ds.cols[dim].role == "measure" and len(ds.categorical_dimensions) >= 3:
                continue
            top = pf.rank_groups(ds, dim, P, lowest=False, top_n=5)
            bottom = pf.rank_groups(ds, dim, P, lowest=True, top_n=5)
            con = pf.contribution(ds, dim, P)
            if not top.empty:
                A.rankings[dim] = {"top": top, "bottom": bottom, "contribution": con}
                gaps = pf.segment_gaps(ds, dim, P)
                if not gaps.empty:
                    A.gaps[dim] = gaps
        A.drivers = dr.analyze_drivers(ds, P)
        if ds.entities and ds.kind(P) in ("additive", "intensity"):
            A.entities = pf.valuable_entities(ds, ds.entities[0], P)
    A.corr = dr.correlations(ds)
    A.anomalies = an.detect_anomalies(ds)
    A.segments = sg.segment_dataset(ds)
    for rm in ds.outcome_flags[:2]:
        for dim in ds.dimensions[:6]:
            if dim == rm:
                continue
            g = pf.segment_gaps(ds, dim, rm)
            if g.empty:
                continue
            g = g[(g["p_value"] < 0.01)]
            if not g.empty:
                row = g.reindex(g["index"].sub(1).abs().sort_values(ascending=False).index).iloc[0]
                A.hotspots.append({"measure": rm, "dim": dim, **row.to_dict()})

    A.findings = _findings(A)
    A.findings.sort(key=lambda f: -f.importance)
    A.investigations = _investigations(A)
    A.summary = _summary(A)
    return A


# --------------------------------------------------------------------------------------
def _findings(A: Analysis) -> list[Finding]:
    ds, P = A.ds, A.primary
    out: list[Finding] = []
    L = ds.label(P) if P else ""
    kind = ds.kind(P) if P else "intensity"

    # ---- change versus the previous period ------------------------------------------
    if A.pop and P:
        p = A.pop
        t = tone(ds, P, p["pct"])
        pv = A.change.get("p_value") if A.change else None
        noise = kind != "additive" and pv is not None and pv >= 0.05      # averages need a significance check
        if noise and p["pct"] is not None:
            out.append(Finding("pop", "change", "info", f"{L} is stable versus {p['prev_label']}",
                               f"{L} moved {pct(p['pct'], signed=True)} to {value(kind, p['latest'])} in {p['latest_label']} versus "
                               f"{p['prev_label']}, but the difference is within normal sampling variation (p={pv:.2f}).", 22, None, P))
        elif p["pct"] is not None and (abs(p["pct"]) >= 5 or p["unusual"]):
            verb = "rose" if p["pct"] > 0 else "fell"
            extra = " — a movement outside the normal range of past changes" if p["unusual"] else ""
            out.append(Finding(
                "pop", "change", t, f"{L} {verb} {abs(p['pct']):.1f}% in {p['latest_label']}",
                f"{L} {verb} {abs(p['pct']):.1f}% to {value(kind, p['latest'])} in {p['latest_label']}, from "
                f"{value(kind, p['prev'])} in {p['prev_label']}{extra}.",
                48 + min(38, abs(p["pct"]) * 0.8) + (8 if p["unusual"] else 0) + (6 if t == "risk" else 0),
                None, P))
        elif p["pct"] is not None:
            out.append(Finding("pop", "change", "info", f"{L} is stable versus {p['prev_label']}",
                               f"{L} moved {pct(p['pct'], signed=True)} in {p['latest_label']} versus {p['prev_label']}, "
                               f"within the range of ordinary variation.", 22, None, P))
        if p["yoy_pct"] is not None and abs(p["yoy_pct"]) >= 10:
            ty = tone(ds, P, p["yoy_pct"])
            out.append(Finding("yoy", "change", ty, f"{L} is {abs(p['yoy_pct']):.0f}% {'above' if p['yoy_pct'] > 0 else 'below'} last year",
                               f"{L} in {p['latest_label']} is {abs(p['yoy_pct']):.1f}% {'higher' if p['yoy_pct'] > 0 else 'lower'} than "
                               f"the same period a year earlier ({p['yoy_label']}: {value(kind, p['yoy_value'])}).",
                               42 + min(30, abs(p["yoy_pct"]) * 0.4) + (4 if ty == "risk" else 0), None, P))

    if A.ts is not None and A.ts.partial_last:
        pl = A.ts.partial_last
        out.append(Finding("partial", "quality", "info", "Latest period is incomplete",
                           f"{A.ts.label(pl['start'])} covers only {pl['coverage'] * 100:.0f}% of the period and is excluded from "
                           f"comparisons and trend calculations so it is not mistaken for a decline.", 30,
                           "Re-run this analysis once the latest period is complete.", P))

    # ---- why: attribution of the latest change --------------------------------------
    ch = A.change
    if ch and ch["dimensions"] and A.pop and A.pop["pct"] is not None and abs(A.pop["pct"]) >= 5 \
            and (kind == "additive" or (ch.get("p_value") is not None and ch["p_value"] < 0.05)):
        d0 = ch["dimensions"][0]
        lead, dim = d0["lead"], d0["dim"]
        p = A.pop
        t = tone(ds, P, p["pct"])
        move = "decline" if p["change"] < 0 else "increase"
        if kind == "additive" and lead.get("base_share") is not None and not pd.isna(lead.get("pct_change")):
            fact = (f"{ds.seg_text(dim, lead['group'])} changed {pct(lead['pct_change'], signed=True)} versus "
                    f"{pct(p['pct'], signed=True)} overall and accounts for {abs(lead['share_of_change']):.0f}% of the net {move}, "
                    f"from {lead['base_share']:.0f}% of the prior-period base.")
        else:
            fact = (f"{ds.seg_text(dim, lead['group'])} is the largest single contributor to the {move} in {L}, "
                    f"at {abs(lead['share_of_change']):.0f}% of the net change.")
        inv = None
        vv = ch.get("volume_value")
        if vv:
            part = "record volume" if abs(vv["volume_effect"]) > abs(vv["value_effect"]) else "average value per record"
            inv = (f"Investigate {ds.seg_text(dim, lead['group'])} between {p['prev_label']} and {p['latest_label']}; "
                   f"the {move} is driven mainly by {part}.")
        else:
            inv = f"Investigate {ds.seg_text(dim, lead['group'])} between {p['prev_label']} and {p['latest_label']}."
        out.append(Finding("change_driver", "change", t, f"{ds.seg_short(dim, lead['group'])} drives the {L} {move}", fact,
                           46 + min(36, abs(p["pct"]) * 0.7) + (6 if t == "risk" else 0), inv, P))

    if A.trend and A.trend["significant"] and P:
        tr = A.trend
        t = tone(ds, P, tr["slope"])
        u = unit_word(A.ts.freq)
        direction = "upward" if tr["direction"] == "up" else "downward"
        out.append(Finding("trend", "trend", t, f"{L} shows a significant {direction} trend",
                           f"{L} has a statistically significant {direction} trend of about {abs(tr['pct_per_period']):.1f}% per {u} "
                           f"over {tr['periods']} {u}s (trend explains {tr['r2'] * 100:.0f}% of variation, p={tr['p_value']:.3f}).",
                           50 + min(24, abs(tr["total_change_pct"] or 0) * 0.2) + (5 if t == "risk" else 0),
                           f"Confirm whether the sustained {direction} movement in {L} is expected and whether it is shared across segments."
                           if t == "risk" else None, P))

    if A.season and A.season["meaningful"] and P:
        s = A.season
        out.append(Finding("season", "seasonality", "opportunity", f"{L} is strongly seasonal",
                           f"{L} follows a clear seasonal pattern (strength {s['strength'] * 100:.0f}%): highest in {s['peak']} "
                           f"({pct(s['peak_pct'], signed=True, decimals=0)} versus average), lowest in {s['trough']} "
                           f"({pct(s['trough_pct'], signed=True, decimals=0)}).",
                           38 + s["strength"] * 22,
                           f"Check that planning, capacity and targets reflect the {s['peak']} peak and {s['trough']} trough.", P))

    fc = A.forecast
    if fc and fc.get("available") and fc["reliability"] != "Low" and P and fc["change_pct"] is not None:
        t = tone(ds, P, fc["change_pct"]) if abs(fc["change_pct"]) >= 3 else "info"
        u = unit_word(fc["freq"])
        out.append(Finding("forecast", "forecast", t, f"{L} outlook: {pct(fc['change_pct'], signed=True, decimals=0)} versus recent {u}s",
                           f"The {fc['model']} projects {L} to average {value(kind, fc['future_mean'])} per {u} over the next "
                           f"{fc['horizon']} {u}s, {abs(fc['change_pct']):.1f}% {'above' if fc['change_pct'] > 0 else 'below'} the last "
                           f"{fc['horizon']} (reliability {fc['reliability'].lower()}; back-test error "
                           f"{pct((fc['backtest_mape'] or 0) * 100, decimals=0)}).",
                           36 + min(26, abs(fc["change_pct"]) * 0.5) + (4 if t == "risk" else 0),
                           f"Compare the projected {L} with targets or capacity; the forecast is statistical and assumes past patterns continue.",
                           P))

    # ---- concentration & performance -------------------------------------------------
    for dim, r in list(A.rankings.items())[:4]:
        con = r["contribution"]
        if con and con["groups"] >= 3:
            basis = "total" if con["basis"] == "total" else "records"
            if con["top1_share"] >= 40 and con["hhi_norm"] >= 0.15:
                out.append(Finding(f"conc_{dim}", "concentration", "risk", f"{ds.seg_short(dim, con['top1'])} dominates {ds.label(dim)}",
                                   f"{con['top1']} accounts for {con['top1_share']:.0f}% of {L if basis == 'total' else 'records'} "
                                   f"across {ds.label(dim)} ({con['groups']} groups).",
                                   44 + min(30, (con["top1_share"] - 40) * 0.8),
                                   f"Assess dependency on {ds.seg_text(dim, con['top1'])} and how results would change if it underperformed.", P))
            elif con["groups"] >= 6 and con["pareto_groups"] <= max(1, con["groups"] // 4):
                out.append(Finding(f"pareto_{dim}", "concentration", "info",
                                   f"{con['pareto_groups']} of {con['groups']} {ds.label(dim)} values drive 80%",
                                   f"{con['pareto_groups']} of {con['groups']} {ds.label(dim)} values account for 80% of "
                                   f"{L if basis == 'total' else 'records'}.", 34, None, P))
        top, bot = r["top"], r["bottom"]
        if len(top) >= 2 and len(bot) >= 1 and dim in A.rankings and list(A.rankings).index(dim) < 2:
            t0, b0 = top.iloc[0], bot.iloc[0]
            if t0["group"] != b0["group"]:
                out.append(Finding(f"perf_{dim}", "performance", "info", f"{ds.label(dim)}: {ds.seg_short(dim, t0['group'])} leads, {ds.seg_short(dim, b0['group'])} trails",
                                   f"By {ds.label(dim)}, {ds.seg_short(dim, t0['group'])} leads on {L} at {value(kind, t0['value'])} "
                                   f"({t0['share']:.0f}% of {'total' if kind == 'additive' else 'records'}), while {ds.seg_short(dim, b0['group'])} is "
                                   f"lowest at {value(kind, b0['value'])} ({b0['share']:.0f}%).", 36, None, P))

    for dim, g in A.gaps.items():
        low = g[(g["index"] < 0.85) & (g["p_value"] < 0.01) & (g["record_share"] >= 5)]
        if not low.empty and P:
            row = low.sort_values("index").iloc[0]
            out.append(Finding(
                f"gap_{dim}", "performance", "info", f"{ds.seg_short(dim, row['group'])} has the lowest per-record {L}",
                f"{ds.seg_text(dim, row['group'])} averages {abs(1 - row['index']) * 100:.0f}% below the overall per-record {L} "
                f"across {int(row['records']):,} records (statistically significant).",
                36 + min(18, abs(1 - row["index"]) * 25),
                f"Check whether the {ds.seg_short(dim, row['group'])} gap reflects genuinely weaker performance or a structurally "
                f"different segment (for example a different price level or mix) before treating it as an opportunity.", P))

    # ---- rate hotspots (outcome flags such as churn / attrition / defects) ----------
    for h in A.hotspots[:3]:
        rm, dim = h["measure"], h["dim"]
        higher = h["index"] > 1
        t = tone(ds, rm, 1 if higher else -1) if ds.polarity(rm) != 0 else "info"
        out.append(Finding(
            f"hot_{rm}_{dim}", "performance", t if t != "positive" else "opportunity",
            f"{ds.seg_short(dim, h['group'])} has {'higher' if higher else 'lower'} {ds.label(rm)}",
            f"{ds.label(rm)} is {pct(h['avg'] * 100)} for {ds.seg_text(dim, h['group'])} versus "
            f"{pct(float(ds.values(rm).mean()) * 100)} overall, across {int(h['records']):,} records (statistically significant).",
            46 + min(24, abs(h["index"] - 1) * 30),
            f"Examine what distinguishes {ds.seg_text(dim, h['group'])} from other groups in {ds.label(rm)}.", rm))

    # ---- drivers ---------------------------------------------------------------------
    d = A.drivers
    if d and d.get("available") and P:
        if d["predictive"] and d["drivers"]:
            top = d["drivers"][:3]
            names = join_and([f"{x['label']} ({x['importance_pct']:.0f}%)" for x in top])
            qual = (f"explaining about {max(d['score'], 0) * 100:.0f}% of variation out-of-sample" if d["metric"] == "R2"
                    else f"with a predictive AUC of {d['score']:.2f} out-of-sample")
            out.append(Finding("drivers", "driver", "opportunity", f"Strongest drivers of {L}: {top[0]['label']}",
                               f"The fields most associated with {L} are {names} of relative importance, {qual}. "
                               f"These are statistical associations, not proven causes.",
                               52 + min(22, max(d["score"], 0) * 25),
                               f"Use {join_and([x['label'] for x in top[:2]])} to structure the analysis of {L}, and validate with a controlled comparison before treating them as causes.", P))
        elif not d["predictive"]:
            out.append(Finding("drivers_weak", "driver", "info", f"No strong drivers of {L} found",
                               f"The other fields in this dataset have little predictive power for {L} "
                               f"(out-of-sample {'R²' if d['metric'] == 'R2' else 'AUC'} {d['score']:.2f}), so any single-factor explanation would be unreliable.",
                               28, f"Consider whether key explanatory data for {L} is missing from this dataset.", P))
        if d.get("redundant"):
            out.append(Finding("redundant", "driver", "info", "Fields nearly duplicate the primary measure",
                               f"{join_and([ds.label(c) for c in d['redundant'][:3]])} is almost perfectly correlated with {L}, "
                               f"so it is probably derived from it and was excluded from driver analysis.", 24, None, P))
    if A.corr and A.corr["near_duplicates"]:
        pr = A.corr["near_duplicates"][0]
        out.append(Finding("neardup", "driver", "info", f"{pr['label_a']} and {pr['label_b']} move together",
                           f"{pr['label_a']} and {pr['label_b']} are almost perfectly correlated (r = {pr['r']:.2f}); one is likely derived from the other.",
                           26, None, None))

    # ---- anomalies -------------------------------------------------------------------
    an_ = A.anomalies
    if an_.get("available") and an_["total"]:
        first = an_["table"].iloc[0] if not an_["table"].empty else None
        t = "risk" if an_["high"] > 0 or an_["share_pct"] >= 2 else "info"
        txt = f"{an_['total']:,} records ({an_['share_pct']:.1f}%) look unusual, {an_['high']:,} of them high priority."
        if first is not None:
            txt += f" Most extreme: {first['Why it stands out']}."
        out.append(Finding("anomalies", "anomaly", t, f"{an_['total']:,} unusual records detected", txt,
                           40 + min(34, an_["high"] * 2.5 + an_["share_pct"] * 2),
                           f"Review the {max(an_['high'], 1):,} highest-priority records with the data owner to confirm whether they are genuine, entry errors or misuse.",
                           None))
    for pa in A.period_anoms[:2]:
        t = tone(ds, P, 1 if pa["direction"] == "high" else -1) if P else "info"
        out.append(Finding(f"panom_{pa['period']}", "anomaly", t if t != "positive" else "info",
                           f"{pa['period']} stands out for {L}",
                           f"{L} in {pa['period']} was {value(kind, pa['value'])}, {'above' if pa['direction'] == 'high' else 'below'} the "
                           f"{value(kind, pa['expected'])} expected from surrounding periods and seasonality.",
                           34 + min(14, abs(pa["z"])), None, P))

    # ---- data quality ----------------------------------------------------------------
    q = A.quality
    serious = [i for i in q["issues"] if i["severity"] in ("high", "medium")]
    if serious:
        top = serious[0]
        out.append(Finding("quality", "quality", "risk" if (top["severity"] == "high" or q["score"] < 75) else "info",
                           top["title"] if len(serious) == 1 else f"{top['title']} (+{len(serious) - 1} more data issues)",
                           f"{top['detail']} Data health is {q['band'].lower()} ({q['score']:.0f}/100)."
                           + (" Duplicated records inflate counts and totals." if "duplicat" in top["title"].lower() else ""),
                           34 + (12 if top["severity"] == "high" else 4) + (100 - q["score"]) * 0.5, top["action"], None))

    # ---- segments / entities / distribution ------------------------------------------
    sgm = A.segments
    if sgm.get("available") and sgm["profiles"] and P:
        best = sgm["profiles"][0]
        if best.get("primary_index") and abs(best["primary_index"] - 1) >= 0.15:
            out.append(Finding("segments", "segment", "opportunity", f"{sgm['k']} distinct record segments found",
                               f"The data separates into {sgm['k']} statistical segments (separation {sgm['silhouette']:.2f}). "
                               f"{best['name']} ({best['share_pct']:.0f}% of records) has the highest average {L} "
                               f"({best['primary_index']:.1f}x overall) and is characterised by {best['description']}.",
                               40, f"Profile {best['name']} against the rest to see how it can be grown or replicated.", P))
    e = A.entities
    if e and P:
        ent = ds.label(ds.entities[0])
        names = e["flagged"]["entity"].head(3).tolist()
        if names and e["kind"] == "additive":
            out.append(Finding("entities", "entity", "opportunity", f"A few {ent} values are unusually valuable",
                               f"{join_and(names)} stand out among {e['entities']:,} {ent} values, each far above the typical "
                               f"{ent} (median {num(e['median'])}); the top {len(e['table'])} contribute {e['top_share']:.0f}% of {L}.",
                               42, f"Review the relationship and dependency on {join_and(names)}.", P))
    if P and P in A.distributions:
        dd = A.distributions[P]
        if dd["median"] and abs(dd["mean"] - dd["median"]) / abs(dd["median"]) >= 0.15 and kind != "rate":
            out.append(Finding("skew", "distribution", "info", f"{L} is skewed: median differs from average",
                               f"The average {L} ({num(dd['mean'])}) is {abs(dd['mean'] / dd['median'] - 1) * 100:.0f}% "
                               f"{'above' if dd['mean'] > dd['median'] else 'below'} the median ({num(dd['median'])}), so a minority of "
                               f"{'high' if dd['mean'] > dd['median'] else 'low'} values pulls the average; the median is the more typical value.",
                               27, None, P))
    return out


def _investigations(A: Analysis) -> list[str]:
    seen, out = set(), []
    for f in A.findings:
        if f.investigate and f.investigate not in seen and f.kind != "info" or (f.investigate and f.category in ("quality",)):
            if f.investigate not in seen:
                seen.add(f.investigate)
                out.append(f.investigate)
    return out[:6]


def _summary(A: Analysis) -> list[str]:
    ds, q = A.ds, A.quality
    scope = f"The dataset contains {len(ds.df):,} records across {ds.df.shape[1]} fields"
    rng = ds.date_range()
    if rng:
        scope += f", covering {rng[0]:%b %Y} to {rng[1]:%b %Y}" if ds.time_spec["kind"] != "year" else f", spanning {rng[0].year}–{rng[1].year}"
    scope += f". Data health is {q['band'].lower()} ({q['score']:.0f}/100)."
    paras = [scope]
    lead, seen_cat = [], set()
    for f in A.findings:                      # one headline per kind of evidence, not three versions of the same story
        if f.category not in seen_cat and f.category not in ("quality",) and f.importance >= 35:
            lead.append(f)
            seen_cat.add(f.category)
        if len(lead) == 3:
            break
    if lead:
        paras.append("Headline signals: " + " ".join(f.fact for f in lead))
    risks = A.risks[:3]
    if risks:
        paras.append("Areas requiring attention: " + "; ".join(r.title for r in risks) + ".")
    opps = A.opportunities[:3]
    if opps:
        paras.append("Worth investigating: " + "; ".join(o.title for o in opps) + ".")
    return paras
