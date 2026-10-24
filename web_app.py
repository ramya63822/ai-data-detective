"""AI Data Detective — executive decision-intelligence workspace (Streamlit)."""
from __future__ import annotations

import hashlib
import html
import io
import re
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

SAMPLES = Path(__file__).parent / "sample_data"

# --------------------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------------------
INK, MUTED, LINE, CANVAS = "#0F1B2D", "#5B6B82", "#DCE2EB", "#F3F5F8"
ACCENT, RAIL = "#0B6E7F", "#0C1727"
GOOD, BAD, WARN, FLAT = "#1B7F5A", "#C0392B", "#A9701A", "#7A8799"
FONT = '"IBM Plex Sans","Segoe UI",system-ui,-apple-system,sans-serif'
PALETTE = [ACCENT, "#33507A", "#A9701A", "#7A8799", "#1B7F5A", "#C0392B"]


def _app_icon():
    try:
        from PIL import Image, ImageDraw

        im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, 127, 127), radius=28, fill=ACCENT)
        d.ellipse((30, 28, 80, 78), outline="white", width=9)
        d.line((72, 72, 98, 98), fill="white", width=11)
        return im
    except Exception:
        return "🔎"


st.set_page_config(page_title="AI Data Detective", page_icon=_app_icon(), layout="wide",
                   initial_sidebar_state="expanded")


def _stretch() -> dict:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 49) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


S = _stretch()
esc = html.escape

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root{{--ink:{INK};--muted:{MUTED};--line:{LINE};--canvas:{CANVAS};--accent:{ACCENT};--rail:{RAIL};
--good:{GOOD};--bad:{BAD};--warn:{WARN};--flat:{FLAT};--font:{FONT};}}
""" + """
.stApp,.stApp p,.stApp li,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp td,.stApp th,.stApp div{font-family:var(--font)}
.stApp{background:var(--canvas);color:var(--ink)}
#MainMenu,footer,.stDeployButton{display:none!important}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1440px;padding:1.5rem 2.4rem 4rem}
.stApp h4{font-size:1.12rem;font-weight:650;letter-spacing:-.005em;color:var(--ink);padding:.2rem 0 .4rem}
.stApp code{background:#E9EEF4;color:var(--ink);padding:1px 5px;border-radius:4px;font-size:.88em}
/* page header */
.ph{display:flex;justify-content:space-between;align-items:flex-end;gap:28px;flex-wrap:wrap;padding-bottom:20px;border-bottom:1px solid var(--line);margin-bottom:22px}
.ph .crumb{font-size:.82rem;color:var(--muted);margin-bottom:6px}
.ph .title{font-size:1.7rem;font-weight:650;letter-spacing:-.015em;line-height:1.2;color:var(--ink)}
.ph .sub{font-size:.95rem;color:var(--muted);margin-top:6px}
.ph .meta{display:flex;gap:26px;flex-wrap:wrap;margin-top:12px}
.ph .meta div{font-size:.8rem;color:var(--muted);line-height:1.35}
.ph .meta b{display:block;font-size:.95rem;font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
.health{min-width:190px}
.health .hl{font-size:.82rem;color:var(--muted)}
.health .hv{font-size:1.9rem;font-weight:650;line-height:1.15;font-variant-numeric:tabular-nums}
.health .hv span{font-size:.95rem;font-weight:500;color:var(--muted);margin-left:2px}
.health .hbar{height:4px;background:#E3E8EF;border-radius:2px;margin:6px 0 6px;overflow:hidden}
.health .hbar i{display:block;height:100%;border-radius:2px;background:var(--flat)}
.health .hb{font-size:.82rem;font-weight:600}
.health.good .hbar i{background:var(--good)} .health.good .hb{color:var(--good)}
.health.warn .hbar i{background:var(--warn)} .health.warn .hb{color:var(--warn)}
.health.bad .hbar i{background:var(--bad)} .health.bad .hb{color:var(--bad)}
/* KPI strip */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin-bottom:8px}
.kpi{padding:16px 20px 14px;box-shadow:1px 0 0 var(--line),0 1px 0 var(--line)}
.kpi-l{font-size:.84rem;color:var(--muted);font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi-v{font-size:1.75rem;font-weight:650;letter-spacing:-.02em;line-height:1.2;margin:4px 0 2px;font-variant-numeric:tabular-nums}
.kpi-d{font-size:.82rem;font-weight:600;min-height:1.2em}
.kpi-d.good{color:var(--good)} .kpi-d.bad{color:var(--bad)} .kpi-d.flat{color:var(--flat);font-weight:500}
.kpi .spark{display:block;width:100%;height:auto;margin-top:10px}
/* panels */
.panel{background:#fff;border:1px solid var(--line);border-radius:8px;margin-bottom:16px}
.panel-h{display:flex;justify-content:space-between;align-items:center;padding:12px 18px;border-bottom:1px solid var(--line);font-size:.95rem;font-weight:650}
.panel-h .count{font-size:.78rem;font-weight:600;color:var(--muted);background:#EDF1F6;border-radius:10px;padding:1px 9px;font-variant-numeric:tabular-nums}
.panel-b{padding:6px 18px 12px}
.panel p{margin:0 0 .85rem;line-height:1.62;font-size:.95rem;max-width:76ch}
.sum{padding-top:12px}
.sum p:first-child{font-size:1.04rem;font-weight:500}
.empty{color:var(--muted);font-size:.9rem;padding:12px 0}
.fi{border-left:3px solid var(--k,#7A8799);padding:1px 0 1px 12px;margin:14px 0}
.fi.risk{--k:var(--bad)} .fi.positive{--k:var(--good)} .fi.opportunity{--k:var(--accent)} .fi.info{--k:var(--flat)}
.fi-t{font-weight:600;font-size:.95rem;line-height:1.4}
.fi-f{color:var(--muted);font-size:.9rem;margin-top:3px;line-height:1.5}
.tag{display:inline-block;font-size:.74rem;font-weight:600;padding:1px 8px;border-radius:4px;margin-right:8px;vertical-align:1px}
.tag.risk,.tag.bad{background:#FBEAE8;color:var(--bad)} .tag.positive,.tag.good{background:#E4F3EC;color:var(--good)}
.tag.opportunity{background:#E0F0F3;color:var(--accent)} .tag.info,.tag.flat{background:#ECEFF4;color:#566478}
.tag.warn{background:#F8EEDB;color:var(--warn)}
.steps{list-style:none;counter-reset:s;margin:10px 0 4px;padding:0}
.steps li{counter-increment:s;display:flex;gap:14px;padding:9px 0;border-bottom:1px solid #EDF0F5;font-size:.95rem;line-height:1.5}
.steps li:last-child{border-bottom:none}
.steps li:before{content:counter(s);flex:0 0 24px;height:24px;border-radius:50%;background:#E0F0F3;color:var(--accent);font-weight:650;font-size:.82rem;display:flex;align-items:center;justify-content:center}
.sec{margin:30px 0 12px}
.sec .t{font-size:1.08rem;font-weight:650;letter-spacing:-.005em}
.sec .n{font-size:.88rem;color:var(--muted);margin-top:2px}
.sub-h{font-size:.86rem;font-weight:650;color:var(--muted);margin:16px 0 4px}
.basis{font-size:.8rem;color:var(--muted);margin-top:10px;padding-top:8px;border-top:1px solid #EDF0F5}
.issue{display:flex;gap:14px;align-items:flex-start;padding:12px 0;border-bottom:1px solid #EDF0F5}
.issue:last-child{border-bottom:none}
.issue .tag{flex:0 0 auto;min-width:58px;text-align:center;margin:2px 0 0}
.issue .it{font-weight:600;font-size:.95rem} .issue .id{color:var(--ink);font-size:.9rem;margin-top:2px;line-height:1.5}
.issue .ia{color:var(--muted);font-size:.85rem;margin-top:4px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-bottom:16px}
.grid3 .panel{margin:0}
.grid3 .panel-b{padding:14px 18px 8px}
/* native widgets */
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 18px}
[data-testid="stMetricLabel"] p{font-size:.84rem;color:var(--muted);font-weight:500}
[data-testid="stMetricValue"]{font-size:1.55rem;font-weight:650;letter-spacing:-.01em;font-variant-numeric:tabular-nums}
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px 10px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:8px;overflow:hidden}
[data-testid="stAlert"]{border-radius:8px;border:1px solid var(--line)}
[data-testid="stExpander"] details{background:#fff;border:1px solid var(--line);border-radius:8px}
[data-testid="stCaptionContainer"]{color:var(--muted)}
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{height:46px;padding:0 16px;background:transparent;color:var(--muted);font-weight:500}
.stTabs [data-baseweb="tab"] p{font-size:.95rem}
.stTabs [aria-selected="true"]{color:var(--ink);font-weight:650}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent);height:2px}
.stTabs [data-baseweb="tab-border"]{display:none}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.4rem}
.stButton>button,.stDownloadButton>button{border-radius:6px;border:1px solid var(--line);background:#fff;color:var(--ink);font-weight:500;padding:.5rem .9rem;transition:border-color .12s,background .12s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--accent);color:var(--accent);background:#F5FAFB}
.stButton>button:focus-visible,.stDownloadButton>button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.stButton>button p{font-size:.88rem;text-align:left;line-height:1.35}
.stDownloadButton>button[kind="primary"],.stDownloadButton>button[data-testid="stBaseButton-primary"]{background:var(--accent);border-color:var(--accent);color:#fff}
.stDownloadButton>button[kind="primary"]:hover,.stDownloadButton>button[data-testid="stBaseButton-primary"]:hover{background:#09596A;color:#fff}
[data-testid="stChatMessage"]{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin-bottom:12px}
[data-testid="stChatInput"]{border-radius:8px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
/* sidebar rail */
section[data-testid="stSidebar"]{background:var(--rail);border-right:1px solid #16253D}
section[data-testid="stSidebar"] *{color:#C3CEDF}
section[data-testid="stSidebar"] .brand{display:flex;align-items:center;gap:11px;padding:6px 0 14px;border-bottom:1px solid #1B2B45;margin-bottom:4px}
section[data-testid="stSidebar"] .brand b{display:block;font-size:1.02rem;font-weight:650;color:#fff;letter-spacing:-.005em}
section[data-testid="stSidebar"] .brand small{display:block;font-size:.78rem;color:#8396B3;margin-top:1px}
section[data-testid="stSidebar"] .side-h{font-size:.84rem;font-weight:650;color:#fff;margin:20px 0 8px}
section[data-testid="stSidebar"] .svc{font-size:.86rem;line-height:1.45;color:#C3CEDF}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],section[data-testid="stSidebar"] small{color:#8396B3}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:#122039;border:1px dashed #36507A;border-radius:8px}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button{background:#1C3253;border:1px solid #36507A;color:#fff}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover{background:#244069;color:#fff;border-color:#4A6A9C}
section[data-testid="stSidebar"] [data-testid="stExpander"] details{background:#101C31;border:1px solid #22344F;border-radius:8px}
section[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#122039;border-color:#36507A}
section[data-testid="stSidebar"] .stButton>button{background:#1C3253;border:1px solid #36507A;color:#fff}
section[data-testid="stSidebar"] .stButton>button:hover{background:#244069;border-color:#4A6A9C;color:#fff}
section[data-testid="stSidebar"] [data-testid="stAlert"] *{color:#0F1B2D}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------------------
def rich(text) -> str:
    """Escape text, keeping simple **bold** and `code` emphasis."""
    t = esc(str(text))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def section(title: str, note: str | None = None) -> None:
    n = f'<div class="n">{esc(note)}</div>' if note else ""
    st.markdown(f'<div class="sec"><div class="t">{esc(title)}</div>{n}</div>', unsafe_allow_html=True)


def sub(title: str) -> None:
    st.markdown(f'<div class="sub-h">{esc(title)}</div>', unsafe_allow_html=True)


def panel(title: str, body: str, count: int | None = None) -> str:
    badge = f'<span class="count">{count}</span>' if count else ""
    return f'<div class="panel"><div class="panel-h"><span>{esc(title)}</span>{badge}</div><div class="panel-b">{body}</div></div>'


def empty(text: str) -> str:
    return f'<div class="empty">{esc(text)}</div>'


KIND_LABEL = {"risk": "Attention", "positive": "Positive", "opportunity": "Opportunity", "info": "Insight"}


def findings_html(fs) -> str:
    return "".join(
        f'<div class="fi {f.kind}"><div class="fi-t"><span class="tag {f.kind}">{KIND_LABEL.get(f.kind, "Insight")}</span>{esc(f.title)}</div>'
        f'<div class="fi-f">{esc(f.fact)}</div></div>' for f in fs)


def _tone(fav) -> str:
    return "flat" if fav is None else ("good" if fav else "bad")


def _spark_points(spark) -> list[float]:
    try:
        if isinstance(spark, dict):
            spark = next((spark[k] for k in ("y", "values", "v") if k in spark), [])
        seq = spark.tolist() if hasattr(spark, "tolist") else list(spark)
        out = []
        for v in seq:
            if isinstance(v, (tuple, list)):
                v = v[-1]
            f = float(v)
            if f == f:
                out.append(f)
        return out if len(out) >= 2 else []
    except Exception:
        return []


def _spark_svg(spark, fav) -> str:
    pts = _spark_points(spark)
    if not pts:
        return ""
    w, h, pad = 160, 36, 3
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    n = len(pts)
    xy = [(pad + i * (w - 2 * pad) / (n - 1), pad + (h - 2 * pad) * (1 - (v - lo) / span)) for i, v in enumerate(pts)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    area = f"{xy[0][0]:.1f},{h} {line} {xy[-1][0]:.1f},{h}"
    col = {"good": GOOD, "bad": BAD, "flat": FLAT}[_tone(fav)]
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" aria-hidden="true">'
            f'<polygon points="{area}" fill="{col}" fill-opacity=".10"/>'
            f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{xy[-1][0]:.1f}" cy="{xy[-1][1]:.1f}" r="2.4" fill="{col}"/></svg>')


def kpi_strip(kpis) -> list:
    """Render KPI cells as one joined strip; returns KPIs whose sparkline could not be drawn inline."""
    cells, fallback = [], []
    for i, k in enumerate(kpis):
        tone = _tone(k["favourable"])
        d = str(k["delta"] or "").strip()
        arrow = "▲ " if d.startswith("+") else ("▼ " if d[:1] in "-−–" else "")
        line = f'<div class="kpi-d {tone}">{arrow}{esc(d)}</div>' if d else f'<div class="kpi-d flat">{esc(k["caption"] or "")}</div>'
        svg = _spark_svg(k["spark"], k["favourable"]) if k["spark"] else ""
        if k["spark"] and not svg:
            fallback.append((i, k))
        cells.append(f'<div class="kpi"><div class="kpi-l" title="{esc(k["label"])}">{esc(k["label"])}</div>'
                     f'<div class="kpi-v">{esc(k["value"])}</div>{line}{svg}</div>')
    st.markdown(f'<div class="kpis">{"".join(cells)}</div>', unsafe_allow_html=True)
    return fallback


def _style_fig(fig) -> None:
    kw = dict(font=dict(family=FONT, color=INK, size=12), paper_bgcolor="rgba(0,0,0,0)",
              hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(family=FONT, color="#FFFFFF", size=12)))
    if not fig.layout.colorway:
        kw["colorway"] = PALETTE
    fig.update_layout(**kw)
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title_font=dict(family=FONT, size=14, color=INK))
    axis = dict(gridcolor="#E7EBF1", linecolor="#C9D1DD", tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)


def chart(fig, key: str, **kw) -> None:
    try:
        _style_fig(fig)
    except Exception:
        pass
    st.plotly_chart(fig, key=key, **S, **kw)


def page_header(crumb: str, title: str, sub_text: str = "", meta: list[tuple[str, str]] | None = None, health: dict | None = None) -> None:
    m = "".join(f"<div>{esc(a)}<b>{esc(b)}</b></div>" for a, b in (meta or []))
    m = f'<div class="meta">{m}</div>' if m else ""
    s = f'<div class="sub">{esc(sub_text)}</div>' if sub_text else ""
    h = ""
    if health:
        sc = health["score"]
        tone = "good" if sc >= 80 else ("warn" if sc >= 60 else "bad")
        h = (f'<div class="health {tone}"><div class="hl">Data health</div><div class="hv">{sc:.0f}<span>/100</span></div>'
             f'<div class="hbar"><i style="width:{max(0, min(100, sc)):.0f}%"></i></div><div class="hb">{esc(health["band"])}</div></div>')
    st.markdown(f'<div class="ph"><div><div class="crumb">{esc(crumb)}</div><div class="title">{esc(title)}</div>{s}{m}</div>{h}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Loading & caching
# --------------------------------------------------------------------------------------
def _load(name: str, raw: bytes | Path):
    key = hashlib.sha1(raw if isinstance(raw, bytes) else raw.read_bytes()).hexdigest()
    store = st.session_state.setdefault("_workspaces", {})
    if key not in store:
        src = io.BytesIO(raw) if isinstance(raw, bytes) else str(raw)
        df = load_data(src, name)
        ds = Dataset(df, name)
        store[key] = {"ds": ds, "A": analyze(ds)}
        store[key]["cp"] = Copilot(ds, store[key]["A"])
    return key, store[key]


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
BRAND = ('<div class="brand"><svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="#0B6E7F"/>'
         '<circle cx="14" cy="14" r="6" fill="none" stroke="#fff" stroke-width="2.4"/><path d="M19 19l6 6" stroke="#fff" stroke-width="2.6" stroke-linecap="round"/></svg>'
         '<div><b>AI Data Detective</b><small>Decision intelligence</small></div></div>')

provider = llm.provider_name()

with st.sidebar:
    st.markdown(BRAND, unsafe_allow_html=True)
    st.markdown('<div class="side-h">Dataset</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    st.caption("CSV or Excel file from any business domain.")

    sample_choice = "—"
    samples = sorted(p for p in SAMPLES.glob("*.csv")) if SAMPLES.exists() else []
    with st.expander("Demo datasets", expanded=False):
        if samples:
            sample_choice = st.selectbox("Open a demo dataset", ["—"] + [p.stem.replace("_", " ").title() for p in samples], index=0)
        else:
            st.caption("No demo datasets are installed.")

    st.markdown('<div class="side-h">AI service</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="svc">{esc(llm.status_label())}</div>', unsafe_allow_html=True)
    if provider in ("openai", "gemini"):
        with st.expander("Connection check", expanded=False):
            if st.button("Test connection", key="test_connection", **S):
                test_fn = llm.test_gemini_connection if provider == "gemini" else llm.test_openai_connection
                with st.spinner("Checking the AI service…"):
                    ok, message = test_fn()
                (st.success if ok else st.error)(message)
    with st.expander("How figures are produced", expanded=False):
        st.caption("Every figure is computed by deterministic analysis. The AI assistant plans questions and explains results, "
                   "and a number is shown only when it is backed by verified evidence.")

raw, fname, source = None, None, ""
if up is not None:
    raw, fname, source = up.getvalue(), up.name, "Uploaded file"
elif sample_choice and sample_choice != "—":
    path = next(p for p in samples if p.stem.replace("_", " ").title() == sample_choice)
    raw, fname, source = path.read_bytes(), path.name, "Demo dataset"

# --------------------------------------------------------------------------------------
# Empty state
# --------------------------------------------------------------------------------------
if raw is None:
    page_header("Workspace", "AI Data Detective", "Decision intelligence for any dataset. Upload a file and the executive view builds itself.")
    caps = [
        ("Executive view, instantly", "KPIs, trends, drivers, anomalies, risks and opportunities appear automatically. No questions needed."),
        ("Verified analysis", "Every number is computed deterministically. Questions such as “why did revenue decline?” run multi-step analysis, not guesses."),
        ("Clear about its limits", "When the data cannot support an analysis, the workspace says so and explains why."),
    ]
    st.markdown('<div class="grid3">' + "".join(panel(t, f"<p>{esc(b)}</p>") for t, b in caps) + "</div>", unsafe_allow_html=True)
    demo_hint = " Or open one from Demo datasets." if samples else ""
    steps = [f"Upload a CSV or Excel file from the sidebar.{demo_hint}", "Review the executive view the workspace generates for your data.",
             "Ask the AI analyst why a metric moved or where to look next."]
    st.markdown(panel("Get started", '<ol class="steps">' + "".join(f"<li>{esc(s_)}</li>" for s_ in steps) + "</ol>"), unsafe_allow_html=True)
    st.stop()

try:
    with st.spinner("Profiling the dataset and running analyses…"):
        key, W = _load(fname, raw)
except Exception:
    st.error("This file couldn't be read as a dataset. Check that it is a valid CSV or Excel file with a header row.")
    st.stop()

ds: Dataset = W["ds"]
A: Analysis = W["A"]
cp: Copilot = W["cp"]
if ds.df.empty or not ds.all_measures and not ds.categorical_dimensions:
    st.warning("This dataset has no numeric measures or categorical fields to analyse.")
    st.stop()

q = A.quality
rng = ds.date_range()
meta = [("Records", f"{len(ds.df):,}"), ("Fields", f"{ds.df.shape[1]}")]
if rng and ds.time_spec["kind"] != "year":
    meta.append(("Period", f"{rng[0]:%b %Y} – {rng[1]:%b %Y}"))
page_header(f"Workspace  /  {source}", Path(fname).stem.replace("_", " ").title(), meta=meta, health=q)

# ---- KPI strip --------------------------------------------------------------------
fallback = kpi_strip(A.kpis[:5])
if fallback:
    fcols = st.columns(len(fallback))
    for c, (i, k) in zip(fcols, fallback):
        with c:
            chart(vz.sparkline(k["spark"], k["favourable"]), f"spark{i}", config={"displayModeBar": False})

tabs = st.tabs(["Executive Overview", "Performance & Trends", "Drivers & Segments", "Anomalies", "Data Health", "AI Analyst", "Report"])

# ---- 1. Executive overview ----------------------------------------------------------
with tabs[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        summary = "".join(f"<p>{rich(p)}</p>" for p in A.summary)
        st.markdown(panel("Executive summary", f'<div class="sum">{summary}</div>'), unsafe_allow_html=True)
        top = [f for f in A.findings if f.importance >= 35][:8]
        st.markdown(panel("Key findings", findings_html(top) or empty("No material findings stood out."), len(top)), unsafe_allow_html=True)
    with right:
        risks = A.risks[:4]
        st.markdown(panel("Risks and areas requiring attention", findings_html(risks) or empty("No material risks identified."), len(risks)),
                    unsafe_allow_html=True)
        opp = A.opportunities[:4] + A.positives[:1]
        st.markdown(panel("Opportunities worth investigating", findings_html(opp) or empty("No clear opportunities identified."), len(opp)),
                    unsafe_allow_html=True)
    if A.investigations:
        items = "".join(f"<li><span>{rich(t)}</span></li>" for t in A.investigations)
        st.markdown(panel("Recommended next investigations", f'<ol class="steps">{items}</ol>', len(A.investigations)), unsafe_allow_html=True)
    if A.ts is not None and A.pop:
        section("Significant changes")
        from detective.report import _changes_table

        ch = _changes_table(A)
        if not ch.empty:
            st.dataframe(ch, hide_index=True, **S)
        chart(vz.trend_chart(ds, A.ts, A.forecast), "ov_trend")

# ---- 2. Performance & trends --------------------------------------------------------
with tabs[1]:
    measures = ds.all_measures
    if not measures:
        st.info("No numeric measures were detected, so performance analysis is unavailable.")
    else:
        sel, _sp = st.columns([1, 2])
        M = sel.selectbox("Measure", measures, format_func=ds.label, key="perf_measure")
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
                section("Trend and forecast")
                chart(vz.trend_chart(ds, ts, fc), "perf_trend")
                if fc and not fc.get("available"):
                    st.caption(fc["reason"])
                elif fc:
                    st.caption(f"Forecast: {fc['model']}. Reliability **{fc['reliability'].lower()}** (back-test error "
                               f"{pct((fc['backtest_mape'] or 0) * 100, decimals=0)} over {fc['backtest_periods']} {unit_word(fc['freq'])}s). "
                               "Assumes past patterns continue.")
                if se and se["meaningful"]:
                    chart(vz.seasonality_bar(se), "perf_season")
                if M == A.primary and A.change and A.change["dimensions"] and pop and pop["pct"] is not None and abs(pop["pct"]) >= 1:
                    section("What moved the latest change")
                    dsel = st.selectbox("Break down by", [d["dim"] for d in A.change["dimensions"]], format_func=ds.label, key="chg_dim")
                    d = next(x for x in A.change["dimensions"] if x["dim"] == dsel)
                    if kind == "additive":
                        chart(vz.waterfall(d["table"], A.change["pop"], ds.label(M), ds.label(dsel)), "perf_wf")
                    t = d["table"].head(10)
                    show = pd.DataFrame({ds.label(dsel): t["group"], pop["prev_label"]: t["v0"].map(lambda x: value(kind, x)),
                                         pop["latest_label"]: t["v1"].map(lambda x: value(kind, x)),
                                         "% change": t["pct_change"].map(lambda x: pct(x, signed=True)),
                                         "Share of change": t["share_of_change"].map(lambda x: pct(x))})
                    st.dataframe(show, hide_index=True, **S)
        else:
            st.info("Trend, period-over-period and forecasting views are unavailable because no suitable time or date column was detected.")
        dims = [d for d in ds.dimensions if d != M]
        if dims:
            section("Segment performance")
            D = st.selectbox("Segment by", dims, format_func=ds.label, key="perf_dim")
            top, bot = pf.rank_groups(ds, D, M, False, 8), pf.rank_groups(ds, D, M, True, 8)
            a, b = st.columns(2)
            if not top.empty:
                with a:
                    chart(vz.ranking_bar(top, "group", "value", f"Highest {ds.label(M)} by {ds.label(D)}", fmt_kind=kind), "rk_top")
                with b:
                    chart(vz.ranking_bar(bot, "group", "value", f"Lowest {ds.label(M)} by {ds.label(D)}", vz.RED, fmt_kind=kind), "rk_bot")
            con = pf.contribution(ds, D, M)
            if con:
                chart(vz.pareto(con["table"], f"Contribution to {'total ' + ds.label(M) if con['basis'] == 'total' else 'record count'} by {ds.label(D)}"),
                      "pareto")
                st.caption(f"{con['top1']} contributes {con['top1_share']:.0f}%; {con['pareto_groups']} of {con['groups']} values make up 80%.")

# ---- 3. Drivers & segments -----------------------------------------------------------
with tabs[2]:
    P = A.primary
    d = A.drivers
    section("Potential drivers")
    if d and d.get("available") and d["predictive"] and d["drivers"]:
        st.caption(f"Fields most associated with {ds.label(P)} (out-of-sample {'R²' if d['metric'] == 'R2' else 'AUC'} {d['score']:.2f}). Associations, not proven causes.")
        chart(vz.driver_bars(d), "drivers")
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
    section("Correlations")
    if A.corr:
        chart(vz.corr_heatmap(A.corr["matrix"]), "corr")
        strong = pd.DataFrame([{"Field A": p["label_a"], "Field B": p["label_b"], "Correlation": f"{p['r']:+.2f}"} for p in A.corr["pairs"][:8]])
        st.dataframe(strong, hide_index=True, **S)
    else:
        st.info("At least two numeric fields are needed for correlation analysis.")
    section("Segments")
    sg_ = A.segments
    if sg_.get("available"):
        st.caption(f"{sg_['k']} statistically distinct groups (separation {sg_['silhouette']:.2f}). These are data-driven clusters, not predefined personas.")
        chart(vz.segment_scatter(sg_), "segs")
        rows = [{"Segment": p["name"], "Records": f"{p['records']:,}", "Share": pct(p["share_pct"]), "Characterised by": p["description"],
                 **({f"Avg {ds.label(P)}": value(ds.kind(P), p["primary_mean"])} if p.get("primary_mean") is not None else {})} for p in sg_["profiles"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, **S)
    else:
        st.info(sg_.get("reason", "Segmentation is not available for this dataset."))
    section("Distributions")
    if A.distributions:
        dsel_col, _sp2 = st.columns([1, 2])
        dm = dsel_col.selectbox("Measure", list(A.distributions), format_func=ds.label, key="dist_m")
        chart(vz.histogram(ds, dm), "hist")
        dd = A.distributions[dm]
        st.caption(f"Median {dd['median']:,.4g}, mean {dd['mean']:,.4g}, 5th–95th percentile {dd['p05']:,.4g}–{dd['p95']:,.4g}")

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
        st.caption("“Unusual” means statistically atypical. These may be genuine large events, entry errors or misuse. Only the data owner can confirm.")
        section("Flagged records")
        st.dataframe(an_["table"], hide_index=True, **S)
        if A.period_anoms:
            section("Unusual periods")
            st.dataframe(pd.DataFrame([{"Period": p["period"], "Actual": value(ds.kind(A.primary), p["value"]),
                                        "Expected": value(ds.kind(A.primary), p["expected"]), "Direction": p["direction"].title()}
                                       for p in A.period_anoms]), hide_index=True, **S)

# ---- 5. Data health ---------------------------------------------------------------------
SEV = {"high": ("High", "bad"), "medium": ("Medium", "warn"), "low": ("Low", "flat")}
with tabs[4]:
    g, m = st.columns([1, 2], gap="large")
    with g:
        chart(vz.health_gauge(q["score"]), "gauge", config={"displayModeBar": False})
    with m:
        a, b, c, d_ = st.columns(4)
        a.metric("Completeness", f"{q['completeness']:.0f}")
        b.metric("Uniqueness", f"{q['uniqueness']:.0f}")
        c.metric("Validity", f"{q['validity']:.0f}")
        d_.metric("Duplicate rows", f"{q['duplicate_rows']:,}")
        with st.expander("How the score is calculated"):
            st.caption("Score = 40% completeness + 30% uniqueness + 30% validity. Completeness penalises missing cells, uniqueness penalises duplicate "
                       "rows and identifiers, validity penalises constant fields, unexpected negatives, inconsistent labels and future dates.")
    if q["issues"]:
        rows = "".join(
            f'<div class="issue"><span class="tag {SEV[i["severity"]][1]}">{SEV[i["severity"]][0]}</span><div>'
            f'<div class="it">{rich(i["title"])}</div><div class="id">{rich(i["detail"])}</div>'
            f'<div class="ia">Suggested action: {esc(i["action"])}</div></div></div>' for i in q["issues"])
        st.markdown(panel("Issues", rows, len(q["issues"])), unsafe_allow_html=True)
    else:
        st.success("No material data-quality issues detected.")
    if (q["columns_table"]["Missing %"] > 0).any():
        chart(vz.missing_bar(q["columns_table"]), "missing")
    section("How the dataset was interpreted")
    if ds.time_spec:
        st.caption(f"Time axis: {ds.time_spec['label']}. Primary measure: {ds.label(ds.primary) if ds.primary else 'none'}.")
    st.dataframe(q["columns_table"], hide_index=True, **S)


# ---- 6. AI analyst -----------------------------------------------------------------------
def render_answer(a: Answer, key: str) -> None:
    st.markdown(f"#### {a.headline}")
    if a.facts:
        sub("What the data shows")
        for f in a.facts:
            st.markdown(f"- {f}")
    if a.interpretation:
        sub("What it suggests")
        st.write(a.interpretation)
    if a.recommendations:
        sub("What to investigate next")
        for r in a.recommendations:
            st.markdown(f"- {r}")
    for lim in a.limitations:
        st.warning(lim)
    for j, (title, t) in enumerate(a.tables):
        st.caption(title)
        st.dataframe(t, hide_index=True, key=f"{key}_t{j}", **S)
    for j, fig in enumerate(a.figures):
        chart(fig, f"{key}_f{j}")
    if a.basis:
        st.markdown(f'<div class="basis">Basis: {esc(a.basis)}</div>', unsafe_allow_html=True)


with tabs[5]:
    section("Ask the analyst", "Ask in plain language. Answers are computed from your data, and the AI never invents numbers.")
    chat = st.session_state.setdefault(f"chat_{key}", [])
    prompt = None
    sug = cp.suggestions()
    if sug:
        sub("Suggested questions")
        scols = st.columns(4)
        for i, s_ in enumerate(sug[:8]):
            if scols[i % 4].button(s_, key=f"sug{i}_{key}", **S):
                prompt = s_
        st.write("")
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
                st.caption("You could try: " + "; ".join(ans.suggestions[:3]))
        chat.append({"role": "assistant", "answer": ans})

# ---- 7. Report -----------------------------------------------------------------------------
with tabs[6]:
    section("Decision report", "A complete report built from the verified analyses in this workspace. No figure is AI-generated.")
    md = build_markdown(fname, A)
    c1, c2, _sp3 = st.columns([1, 1, 2])
    c1.download_button("Download report (HTML, with charts)", build_html(fname, A), file_name="decision_report.html",
                       mime="text/html", type="primary", **S)
    c2.download_button("Download report (Markdown)", md, file_name="decision_report.md", mime="text/markdown", **S)
    with st.expander("Preview", expanded=True):
        st.markdown(md)
