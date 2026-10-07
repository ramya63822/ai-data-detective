"""Executive decision report. Fully deterministic: every statement comes from verified evidence."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from . import visuals as vz
from .engine import Analysis
from .fmt import pct, value
from . import timeseries as tsm


def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_None._\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
    return "\n".join(lines) + "\n"


def _changes_table(A: Analysis) -> pd.DataFrame:
    ds, rows = A.ds, []
    if not ds.has_time:
        return pd.DataFrame()
    for m in ds.all_measures[:6]:
        ts = A.ts if m == A.primary and A.ts is not None else tsm.build_ts(ds, m)
        p = tsm.period_over_period(ts) if ts is not None else None
        if p and p["pct"] is not None:
            k = ds.kind(m)
            rows.append({"Measure": ds.label(m), p["prev_label"]: value(k, p["prev"]), p["latest_label"]: value(k, p["latest"]),
                         "Change": pct(p["pct"], signed=True),
                         "vs last year": pct(p["yoy_pct"], signed=True) if p["yoy_pct"] is not None else "n/a"})
    return pd.DataFrame(rows)


def build_markdown(name: str, A: Analysis) -> str:
    ds, q = A.ds, A.quality
    out = [f"# Decision Intelligence Report — {name}", f"_Generated {datetime.now():%d %b %Y %H:%M}_", ""]
    out += ["## Executive summary", *[p + "\n" for p in A.summary]]
    out += ["## Key findings"]
    for f in [x for x in A.findings if x.importance >= 35][:10]:
        out.append(f"- **{f.title}** ({f.kind}). {f.fact}")
    out += ["", "## Risks and areas requiring attention"]
    out += [f"- **{f.title}.** {f.fact}" for f in A.risks[:6]] or ["_No material risks identified._"]
    out += ["", "## Opportunities and areas worth investigating"]
    out += [f"- **{f.title}.** {f.fact}" for f in A.opportunities[:6]] or ["_No clear opportunities identified._"]
    out += ["", "## Recommended next investigations"]
    out += [f"{i}. {t}" for i, t in enumerate(A.investigations, 1)] or ["_None._"]
    ch = _changes_table(A)
    if not ch.empty:
        out += ["", "## Significant changes (latest complete period vs previous)", _md_table(ch)]
    if A.anomalies.get("available") and A.anomalies["total"]:
        out += ["", "## Unusual records (highest priority)", _md_table(A.anomalies["table"].head(10)),
                "_Unusual means statistically atypical; it does not confirm an error._"]
    if A.drivers and A.drivers.get("predictive") and A.drivers["drivers"]:
        out += ["", "## Potential drivers", *[f"- {d['label']} ({d['importance_pct']:.0f}% of relative importance)" for d in A.drivers["drivers"]],
                "_Associations, not proven causes._"]
    out += ["", "## Data health", f"Score **{q['score']:.0f}/100** ({q['band']}): completeness {q['completeness']:.0f}, "
            f"uniqueness {q['uniqueness']:.0f}, validity {q['validity']:.0f}."]
    out += [f"- **{i['title']}** — {i['detail']}" for i in q["issues"][:8]] or ["- No material issues."]
    out += ["", "## Method and caveats",
            "All figures are computed deterministically from the uploaded data. Incomplete trailing periods are excluded from comparisons. "
            "Drivers and correlations are statistical associations. Forecasts are back-tested and assume historical patterns continue. "
            "Anomalies are statistical outliers, not confirmed errors. Recommendations are suggested lines of inquiry derived from the findings above."]
    return "\n".join(out) + "\n"


def build_html(name: str, A: Analysis) -> str:
    ds = A.ds
    esc = html.escape
    figs = []
    if A.ts is not None:
        figs.append(vz.trend_chart(ds, A.ts, A.forecast))
    if A.change and A.change["dimensions"] and ds.kind(A.primary) == "additive":
        d0 = A.change["dimensions"][0]
        figs.append(vz.waterfall(d0["table"], A.change["pop"], ds.label(A.primary), ds.label(d0["dim"])))
    if A.drivers and A.drivers.get("predictive") and A.drivers["drivers"]:
        figs.append(vz.driver_bars(A.drivers))
    chart_html = "".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(figs))

    def items(fs):
        return "".join(f"<li><b>{esc(f.title)}.</b> {esc(f.fact)}</li>" for f in fs) or "<li>None identified.</li>"

    ch = _changes_table(A)
    rows = "".join("<tr>" + "".join(f"<td>{esc(str(v))}</td>" for v in r) + "</tr>" for r in ch.itertuples(index=False))
    head = "".join(f"<th>{esc(str(c))}</th>" for c in ch.columns)
    q = A.quality
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(name)} — Decision Intelligence Report</title>
<style>body{{font-family:Inter,Segoe UI,sans-serif;max-width:980px;margin:32px auto;color:#1f2a44;line-height:1.55;padding:0 16px}}
h1{{margin-bottom:0}}h2{{border-bottom:2px solid #e8ebf2;padding-bottom:4px;margin-top:32px}}.muted{{color:#6b7488}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #e1e5ee;padding:6px 10px;text-align:left}}th{{background:#f4f6fb}}
.pill{{display:inline-block;background:#eef2ff;border-radius:99px;padding:2px 12px}}</style></head><body>
<h1>Decision Intelligence Report</h1><p class="muted">{esc(name)} · generated {datetime.now():%d %b %Y %H:%M}</p>
<h2>Executive summary</h2>{''.join(f'<p>{esc(p)}</p>' for p in A.summary)}
<h2>Key findings</h2><ul>{items([f for f in A.findings if f.importance >= 35][:10])}</ul>
<h2>Risks and areas requiring attention</h2><ul>{items(A.risks[:6])}</ul>
<h2>Opportunities and areas worth investigating</h2><ul>{items(A.opportunities[:6])}</ul>
<h2>Recommended next investigations</h2><ol>{''.join(f'<li>{esc(i)}</li>' for i in A.investigations)}</ol>
{'<h2>Significant changes</h2><table><tr>' + head + '</tr>' + rows + '</table>' if not ch.empty else ''}
<h2>Charts</h2>{chart_html}
<h2>Data health</h2><p><span class="pill">{q['score']:.0f}/100 · {esc(q['band'])}</span></p>
<ul>{''.join(f"<li><b>{esc(i['title'])}</b> — {esc(i['detail'])}</li>" for i in q['issues'][:8]) or '<li>No material issues.</li>'}</ul>
<h2>Method and caveats</h2><p class="muted">All figures are computed deterministically from the uploaded data. Drivers and correlations are associations,
not proven causes. Anomalies are statistical outliers, not confirmed errors. Forecasts assume historical patterns continue.</p>
</body></html>"""
