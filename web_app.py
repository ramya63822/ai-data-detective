"""AI Data Detective — executive decision-intelligence workspace (Streamlit)."""
from __future__ import annotations
from detective.llm import test_gemini_connection

import hashlib
import html
from pathlib import Path

import pandas as pd
import streamlit as st

from detective import llm
from detective import performance as pf
from detective import timeseries as tsm
from detective import visuals as vz
from detective.copilot import Answer, Copilot
from detective.dataset import Dataset, load_data
from detective.engine import Analysis, analyze
from detective.fmt import pct, unit_word, value
from detective.report import build_html, build_markdown

from detective.llm import test_gemini_connection, provider_name

if provider_name() == "gemini":
    if st.button("Test Gemini connection"):
        with st.spinner("Testing Gemini connection..."):
            success, message = test_gemini_connection()

        if success:
            st.success(message)
        else:
            st.error(message)


SAMPLES = Path(__file__).parent / "sample_data"

st.set_page_config(page_title="AI Data Detective", page_icon="🔎", layout="wide")


def _stretch() -> dict:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 49) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


S = _stretch()
esc = html.escape

st.markdown("""
<style>
.block-container {max-width: 1380px; padding-top: 1.4rem;}
.hero {display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:2px;}
.hero h1 {font-size:2rem; margin:0; font-weight:760; letter-spacing:-.02em;}
.hero .sub {opacity:.65; font-size:.95rem;}
.card {border:1px solid rgba(128,138,160,.28); border-radius:14px; padding:14px 16px; background:rgba(128,138,160,.06);}
.kpi-label {font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; opacity:.65;}
.kpi-value {font-size:1.75rem; font-weight:720; line-height:1.15; margin:2px 0;}
.kpi-delta {font-size:.82rem; font-weight:600;}
.good {color:#2e9e6b;} .bad {color:#d64545;} .flat {color:#8a94a6;}
.chip {display:inline-block; font-size:.7rem; font-weight:700; padding:2px 9px; border-radius:99px; margin-right:8px;
       text-transform:uppercase; letter-spacing:.05em;}
.chip.risk {background:rgba(214,69,69,.14); color:#d64545;} .chip.positive {background:rgba(46,158,107,.15); color:#2e9e6b;}
.chip.opportunity {background:rgba(59,111,212,.15); color:#3b6fd4;} .chip.info {background:rgba(138,148,166,.18); color:#8a94a6;}
.finding {padding:10px 0; border-bottom:1px solid rgba(128,138,160,.2);}
.finding:last-child {border-bottom:none;}
.finding .t {font-weight:650;} .finding .f {opacity:.82; font-size:.92rem; margin-top:2px;}
.sect {font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; opacity:.6; font-weight:700; margin:14px 0 4px;}
.pill {display:inline-block; border-radius:99px; padding:2px 12px; font-size:.8rem; font-weight:650; background:rgba(59,111,212,.14); color:#3b6fd4;}
.basis {font-size:.78rem; opacity:.6;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Loading & caching
# --------------------------------------------------------------------------------------
def _load(name: str, raw: bytes | Path):
    key = hashlib.sha1(raw if isinstance(raw, bytes) else raw.read_bytes()).hexdigest()
    store = st.session_state.setdefault("_workspaces", {})
    if key not in store:
        import io

        src = io.BytesIO(raw) if isinstance(raw, bytes) else str(raw)
        df = load_data(src, name)
        ds = Dataset(df, name)
        store[key] = {"ds": ds, "A": analyze(ds)}
        store[key]["cp"] = Copilot(ds, store[key]["A"])
    return key, store[key]


with st.sidebar:
    st.markdown("### Workspace")
    up = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel dataset from any business domain.",
    )

    # Keep the user's own data upload as the primary action; demos are secondary.
    sample_choice = "—"
    with st.expander("Demo Workspace", expanded=False):
        st.caption("Explore the platform using built-in demo datasets.")
        samples = sorted(p for p in SAMPLES.glob("*.csv")) if SAMPLES.exists() else []
        if samples:
            sample_choice = st.selectbox(
                "Try a demo dataset",
                ["—"] + [p.stem.replace("_", " ").title() for p in samples],
                index=0,
            )
        else:
            st.caption("No demo datasets available.")

    st.divider()
    st.caption(llm.status_label())
    if llm.provider_name() == "openai":
        if st.button("Test OpenAI connection", key="test_openai_connection", **S):
            with st.spinner("Checking the OpenAI API…"):
                ok, message = llm.test_openai_connection()
            if ok:
                st.success(message)
            else:
                st.warning(message)
    st.caption(
        "All figures are computed by deterministic analysis. OpenAI is used for question planning "
        "and interpretation; figures are shown only when grounded in verified evidence."
    )

raw, fname = None, None
if up is not None:
    raw, fname = up.getvalue(), up.name
elif sample_choice and sample_choice != "—":
    path = next(p for p in samples if p.stem.replace("_", " ").title() == sample_choice)
    raw, fname = path.read_bytes(), path.name

if raw is None:
    st.markdown('<div class="hero"><h1>🔎 AI Data Detective</h1><span class="sub">Decision intelligence for any dataset</span></div>',
                unsafe_allow_html=True)
    st.info("Upload a CSV or Excel file in the sidebar (or pick a sample) to open an executive workspace.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Instant executive view**\n\nKPIs, trends, drivers, anomalies, risks and opportunities appear automatically — no questions needed.")
    c2.markdown("**Verified analysis**\n\nEvery number is computed deterministically. Questions like “why did revenue decline?” run multi-step analysis, not guesses.")
    c3.markdown("**Honest about limits**\n\nWhen the data can't support an analysis, the product says so and explains why.")
    st.stop()

try:
    with st.spinner("Profiling the dataset and running analyses…"):
        key, W = _load(fname, raw)
except Exception:
    st.error("This file could not be read as a dataset. Please check that it is a valid CSV or Excel file with a header row.")
    st.stop()

ds: Dataset = W["ds"]
A: Analysis = W["A"]
cp: Copilot = W["cp"]
if ds.df.empty or not ds.all_measures and not ds.categorical_dimensions:
    st.warning("This dataset has no numeric measures or categorical fields to analyse.")
    st.stop()

q = A.quality
rng = ds.date_range()
subtitle = f"{len(ds.df):,} records · {ds.df.shape[1]} fields" + (
    f" · {rng[0]:%b %Y} – {rng[1]:%b %Y}" if rng and ds.time_spec["kind"] != "year" else "")
st.markdown(f'<div class="hero"><h1>{esc(Path(fname).stem.replace("_", " ").title())}</h1>'
            f'<span class="sub">{esc(subtitle)}</span><span class="pill">Data health {q["score"]:.0f}/100 · {esc(q["band"])}</span></div>',
            unsafe_allow_html=True)

# ---- KPI strip --------------------------------------------------------------------
cols = st.columns(min(5, len(A.kpis)))
for i, (c, k) in enumerate(zip(cols, A.kpis[:5])):
    with c:
        cls = "flat" if k["favourable"] is None else ("good" if k["favourable"] else "bad")
        delta = f'<div class="kpi-delta {cls}">{esc(k["delta"])}</div>' if k["delta"] else f'<div class="kpi-delta flat">{esc(k["caption"])}</div>'
        st.markdown(f'<div class="card"><div class="kpi-label">{esc(k["label"])}</div><div class="kpi-value">{esc(k["value"])}</div>{delta}</div>',
                    unsafe_allow_html=True)
        if k["spark"]:
            st.plotly_chart(vz.sparkline(k["spark"], k["favourable"]), key=f"spark{i}", config={"displayModeBar": False}, **S)

tabs = st.tabs(["Executive Overview", "Performance & Trends", "Drivers & Segments", "Anomalies", "Data Health", "AI Analyst", "Report"])


def finding_html(fs) -> str:
    return "".join(
        f'<div class="finding"><span class="chip {f.kind}">{esc({"risk": "Attention", "positive": "Positive", "opportunity": "Opportunity", "info": "Insight"}[f.kind])}</span>'
        f'<span class="t">{esc(f.title)}</span><div class="f">{esc(f.fact)}</div></div>' for f in fs)


# ---- 1. Executive overview ----------------------------------------------------------
with tabs[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<div class="sect">Executive summary</div>', unsafe_allow_html=True)
        for p in A.summary:
            st.write(p)
        st.markdown('<div class="sect">Key findings</div>', unsafe_allow_html=True)
        top = [f for f in A.findings if f.importance >= 35][:8]
        st.markdown(f'<div class="card">{finding_html(top) if top else "No material findings stood out."}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="sect">Risks & areas requiring attention</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">{finding_html(A.risks[:4]) if A.risks else "No material risks identified."}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sect">Opportunities worth investigating</div>', unsafe_allow_html=True)
        opp = A.opportunities[:4] + A.positives[:1]
        st.markdown(f'<div class="card">{finding_html(opp) if opp else "No clear opportunities identified."}</div>', unsafe_allow_html=True)
    if A.investigations:
        st.markdown('<div class="sect">Recommended next investigations</div>', unsafe_allow_html=True)
        for i, t in enumerate(A.investigations, 1):
            st.markdown(f"**{i}.** {t}")
    if A.ts is not None and A.pop:
        st.markdown('<div class="sect">Significant changes</div>', unsafe_allow_html=True)
        from detective.report import _changes_table

        ch = _changes_table(A)
        if not ch.empty:
            st.dataframe(ch, hide_index=True, **S)
        st.plotly_chart(vz.trend_chart(ds, A.ts, A.forecast), key="ov_trend", **S)

# ---- 2. Performance & trends --------------------------------------------------------
with tabs[1]:
    measures = ds.all_measures
    if not measures:
        st.info("No numeric measures were detected, so performance analysis is unavailable.")
    else:
        M = st.selectbox("Measure", measures, format_func=ds.label, key="perf_measure")
        kind = ds.kind(M)
        if ds.has_time:
            ts = A.ts if M == A.primary and A.ts is not None else tsm.build_ts(ds, M)
            if ts is None or len(ts.values.dropna()) < 2:
                st.info("Not enough complete periods for time analysis of this measure.")
            else:
                pop = tsm.period_over_period(ts)
                se = A.season if M == A.primary else tsm.seasonality(ts)
                fc = A.forecast if M == A.primary else tsm.forecast(ts, se)
                c1, c2, c3 = st.columns(3)
                if pop and pop["pct"] is not None:
                    c1.metric(f"{pop['latest_label']} vs {pop['prev_label']}", value(kind, pop["latest"]), pct(pop["pct"], signed=True))
                if pop and pop["yoy_pct"] is not None:
                    c2.metric(f"vs {pop['yoy_label']}", value(kind, pop["latest"]), pct(pop["yoy_pct"], signed=True))
                tr = tsm.trend(ts)
                if tr:
                    c3.metric("Trend", "Significant" if tr["significant"] else "No clear trend",
                              f"{tr['pct_per_period']:+.1f}% per {unit_word(ts.freq)}" if tr["significant"] else None)
                if ts.partial_last:
                    st.caption(f"{ts.label(ts.partial_last['start'])} is incomplete ({ts.partial_last['coverage'] * 100:.0f}% of the period) and is excluded from comparisons.")
                st.plotly_chart(vz.trend_chart(ds, ts, fc), key="perf_trend", **S)
                if fc and not fc.get("available"):
                    st.caption(fc["reason"])
                elif fc:
                    st.caption(f"Forecast: {fc['model']} · reliability **{fc['reliability'].lower()}** (back-test error "
                               f"{pct((fc['backtest_mape'] or 0) * 100, decimals=0)} over {fc['backtest_periods']} {unit_word(fc['freq'])}s). "
                               "Assumes past patterns continue.")
                if se and se["meaningful"]:
                    st.plotly_chart(vz.seasonality_bar(se), key="perf_season", **S)
                if M == A.primary and A.change and A.change["dimensions"] and pop and pop["pct"] is not None and abs(pop["pct"]) >= 1:
                    st.markdown('<div class="sect">What moved the latest change</div>', unsafe_allow_html=True)
                    dsel = st.selectbox("Break down by", [d["dim"] for d in A.change["dimensions"]], format_func=ds.label, key="chg_dim")
                    d = next(x for x in A.change["dimensions"] if x["dim"] == dsel)
                    if kind == "additive":
                        st.plotly_chart(vz.waterfall(d["table"], A.change["pop"], ds.label(M), ds.label(dsel)), key="perf_wf", **S)
                    t = d["table"].head(10)
                    show = pd.DataFrame({ds.label(dsel): t["group"], pop["prev_label"]: t["v0"].map(lambda x: value(kind, x)),
                                         pop["latest_label"]: t["v1"].map(lambda x: value(kind, x)),
                                         "% change": t["pct_change"].map(lambda x: pct(x, signed=True)),
                                         "Share of change": t["share_of_change"].map(lambda x: pct(x))})
                    st.dataframe(show, hide_index=True, **S)
        else:
            st.info("Trend, period-over-period and forecasting views are unavailable because no suitable time/date column was detected.")
        dims = [d for d in ds.dimensions if d != M]
        if dims:
            st.markdown('<div class="sect">Segment performance</div>', unsafe_allow_html=True)
            D = st.selectbox("Segment by", dims, format_func=ds.label, key="perf_dim")
            top, bot = pf.rank_groups(ds, D, M, False, 8), pf.rank_groups(ds, D, M, True, 8)
            a, b = st.columns(2)
            if not top.empty:
                a.plotly_chart(vz.ranking_bar(top, "group", "value", f"Highest {ds.label(M)} by {ds.label(D)}", fmt_kind=kind), key="rk_top", **S)
                b.plotly_chart(vz.ranking_bar(bot, "group", "value", f"Lowest {ds.label(M)} by {ds.label(D)}", vz.RED, fmt_kind=kind), key="rk_bot", **S)
            con = pf.contribution(ds, D, M)
            if con:
                st.plotly_chart(vz.pareto(con["table"], f"Contribution to {'total ' + ds.label(M) if con['basis'] == 'total' else 'record count'} by {ds.label(D)}"),
                                key="pareto", **S)
                st.caption(f"{con['top1']} contributes {con['top1_share']:.0f}%; {con['pareto_groups']} of {con['groups']} values make up 80%.")

# ---- 3. Drivers & segments -----------------------------------------------------------
with tabs[2]:
    P = A.primary
    d = A.drivers
    st.markdown('<div class="sect">Potential drivers</div>', unsafe_allow_html=True)
    if d and d.get("available") and d["predictive"] and d["drivers"]:
        st.caption(f"Fields most associated with {ds.label(P)} (out-of-sample {'R²' if d['metric'] == 'R2' else 'AUC'} {d['score']:.2f}). Associations, not proven causes.")
        st.plotly_chart(vz.driver_bars(d), key="drivers", **S)
        rows = []
        for x in d["drivers"]:
            if x["type"] == "numeric":
                rows.append({"Field": x["label"], "Importance": f"{x['importance_pct']:.0f}%", "Relationship": f"{x['direction'].title()} (ρ={x['rho']:+.2f})"})
            else:
                rows.append({"Field": x["label"], "Importance": f"{x['importance_pct']:.0f}%",
                             "Relationship": f"{x.get('best', '')} highest, {x.get('worst', '')} lowest"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, **S)
    elif d and d.get("available"):
        st.info(f"No field in this dataset predicts {ds.label(P)} reliably, so no drivers are reported rather than weak guesses.")
    else:
        st.info(d["reason"] if d else "Driver analysis needs a numeric measure and at least one other field.")
    st.markdown('<div class="sect">Correlations</div>', unsafe_allow_html=True)
    if A.corr:
        st.plotly_chart(vz.corr_heatmap(A.corr["matrix"]), key="corr", **S)
        strong = pd.DataFrame([{"Field A": p["label_a"], "Field B": p["label_b"], "Correlation": f"{p['r']:+.2f}"} for p in A.corr["pairs"][:8]])
        st.dataframe(strong, hide_index=True, **S)
    else:
        st.info("At least two numeric fields are needed for correlation analysis.")
    st.markdown('<div class="sect">Segments</div>', unsafe_allow_html=True)
    sg_ = A.segments
    if sg_.get("available"):
        st.caption(f"{sg_['k']} statistically distinct groups (separation {sg_['silhouette']:.2f}). These are data-driven clusters, not predefined personas.")
        st.plotly_chart(vz.segment_scatter(sg_), key="segs", **S)
        rows = [{"Segment": p["name"], "Records": f"{p['records']:,}", "Share": pct(p["share_pct"]), "Characterised by": p["description"],
                 **({f"Avg {ds.label(P)}": value(ds.kind(P), p["primary_mean"])} if p.get("primary_mean") is not None else {})} for p in sg_["profiles"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, **S)
    else:
        st.info(sg_.get("reason", "Segmentation is not available for this dataset."))
    st.markdown('<div class="sect">Distributions</div>', unsafe_allow_html=True)
    if A.distributions:
        dm = st.selectbox("Measure", list(A.distributions), format_func=ds.label, key="dist_m")
        st.plotly_chart(vz.histogram(ds, dm), key="hist", **S)
        dd = A.distributions[dm]
        st.caption(f"Median {dd['median']:,.4g} · mean {dd['mean']:,.4g} · 5th–95th percentile {dd['p05']:,.4g}–{dd['p95']:,.4g}")

# ---- 4. Anomalies ---------------------------------------------------------------------
with tabs[3]:
    an_ = A.anomalies
    if not an_.get("available"):
        st.info(an_.get("reason", "Anomaly detection is unavailable for this dataset."))
    elif not an_["total"]:
        st.success("No records stood out as unusual using the available methods.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Unusual records", f"{an_['total']:,}")
        c2.metric("Share of data", pct(an_["share_pct"]))
        c3.metric("High priority", f"{an_['high']:,}")
        st.caption("“Unusual” means statistically atypical. These may be genuine large events, entry errors or misuse — only the data owner can confirm.")
        st.dataframe(an_["table"], hide_index=True, **S)
        if A.period_anoms:
            st.markdown('<div class="sect">Unusual periods</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"Period": p["period"], "Actual": value(ds.kind(A.primary), p["value"]),
                                        "Expected": value(ds.kind(A.primary), p["expected"]), "Direction": p["direction"].title()}
                                       for p in A.period_anoms]), hide_index=True, **S)

# ---- 5. Data health ---------------------------------------------------------------------
with tabs[4]:
    g, m = st.columns([1, 2])
    g.plotly_chart(vz.health_gauge(q["score"]), key="gauge", config={"displayModeBar": False}, **S)
    with m:
        a, b, c, d_ = st.columns(4)
        a.metric("Completeness", f"{q['completeness']:.0f}")
        b.metric("Uniqueness", f"{q['uniqueness']:.0f}")
        c.metric("Validity", f"{q['validity']:.0f}")
        d_.metric("Duplicate rows", f"{q['duplicate_rows']:,}")
        st.caption("Score = 40% completeness + 30% uniqueness + 30% validity. Completeness penalises missing cells, uniqueness penalises duplicate rows and identifiers, validity penalises constant fields, unexpected negatives, inconsistent labels and future dates.")
    st.markdown('<div class="sect">Issues</div>', unsafe_allow_html=True)
    if q["issues"]:
        for i in q["issues"]:
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}[i["severity"]]
            st.markdown(f"{icon} **{i['title']}** — {i['detail']}  \n<span class='basis'>Suggested action: {esc(i['action'])}</span>", unsafe_allow_html=True)
    else:
        st.success("No material data-quality issues detected.")
    if (q["columns_table"]["Missing %"] > 0).any():
        st.plotly_chart(vz.missing_bar(q["columns_table"]), key="missing", **S)
    st.markdown('<div class="sect">How the dataset was interpreted</div>', unsafe_allow_html=True)
    if ds.time_spec:
        st.caption(f"Time axis: {ds.time_spec['label']} · Primary measure: {ds.label(ds.primary) if ds.primary else 'none'}")
    st.dataframe(q["columns_table"], hide_index=True, **S)

# ---- 6. AI analyst -----------------------------------------------------------------------
def render_answer(a: Answer, key: str) -> None:
    st.markdown(f"#### {a.headline}")
    if a.facts:
        st.markdown('<div class="sect">What the data shows</div>', unsafe_allow_html=True)
        for f in a.facts:
            st.markdown(f"- {f}")
    if a.interpretation:
        st.markdown('<div class="sect">What it suggests</div>', unsafe_allow_html=True)
        st.write(a.interpretation)
    if a.recommendations:
        st.markdown('<div class="sect">What to investigate next</div>', unsafe_allow_html=True)
        for r in a.recommendations:
            st.markdown(f"- {r}")
    for lim in a.limitations:
        st.warning(lim)
    for j, (title, t) in enumerate(a.tables):
        st.caption(title)
        st.dataframe(t, hide_index=True, key=f"{key}_t{j}", **S)
    for j, fig in enumerate(a.figures):
        st.plotly_chart(fig, key=f"{key}_f{j}", **S)
    if a.basis:
        st.markdown(f'<div class="basis">Basis: {esc(a.basis)}</div>', unsafe_allow_html=True)


with tabs[5]:
    st.markdown("Ask business questions in plain language. Answers are computed from your data; the AI never invents numbers.")
    chat = st.session_state.setdefault(f"chat_{key}", [])
    prompt = None
    sug = cp.suggestions()
    scols = st.columns(4)
    for i, s_ in enumerate(sug[:8]):
        if scols[i % 4].button(s_, key=f"sug{i}_{key}", **S):
            prompt = s_
    for n, item in enumerate(chat):
        with st.chat_message(item["role"]):
            if item["role"] == "user":
                st.write(item["content"])
            else:
                render_answer(item["answer"], f"h{n}_{key}")
    typed = st.chat_input("Ask about performance, changes, drivers, risks, trends or the data itself…")
    prompt = typed or prompt
    if prompt:
        chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                ans = cp.ask(prompt)
            render_answer(ans, f"n{len(chat)}_{key}")
            if ans.suggestions and ans.limitations:
                st.caption("You could try: " + " · ".join(ans.suggestions[:3]))
        chat.append({"role": "assistant", "answer": ans})

# ---- 7. Report -----------------------------------------------------------------------------
with tabs[6]:
    st.markdown("A complete decision report built from the verified analyses above — no AI-generated figures.")
    md = build_markdown(fname, A)
    c1, c2 = st.columns(2)
    c1.download_button("Download report (HTML, with charts)", build_html(fname, A), file_name="decision_report.html", mime="text/html", **S)
    c2.download_button("Download report (Markdown)", md, file_name="decision_report.md", mime="text/markdown", **S)
    with st.expander("Preview", expanded=True):
        st.markdown(md)

