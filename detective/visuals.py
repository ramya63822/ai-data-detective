"""Chart builders. All charts share one restrained executive theme."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .dataset import Dataset
from .fmt import num, pct, unit_word

NAVY, BLUE, TEAL, RED, AMBER, GREEN, GREY = "#1f2a44", "#3b6fd4", "#1aa3a3", "#d64545", "#e0a030", "#2e9e6b", "#8a94a6"
SEG_COLORS = [BLUE, TEAL, AMBER, RED, GREEN, "#8e5bd4"]


def _base(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15, "color": GREY}} if title else None,
        height=height, margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=GREY), hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#8a94a6", tickfont=dict(color=GREY), title_font=dict(color=GREY))
    fig.update_yaxes(gridcolor="rgba(138,148,166,0.25)", zeroline=False, tickfont=dict(color=GREY), title_font=dict(color=GREY))
    return fig


def trend_chart(ds: Dataset, ts, forecast: Optional[dict] = None, title: str | None = None) -> go.Figure:
    L = ds.label(ts.measure)
    v = ts.values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v.index, y=v.values, mode="lines+markers", name=L, line=dict(color=BLUE, width=2.5),
                             marker=dict(size=5), hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra></extra>"))
    if ts.partial_last:
        pl = ts.partial_last
        fig.add_trace(go.Scatter(x=[pl["start"]], y=[pl["value"]], mode="markers", name="Incomplete period (excluded)",
                                 marker=dict(color=GREY, symbol="circle-open", size=10)))
    if forecast and forecast.get("available"):
        f = forecast["frame"]
        fig.add_trace(go.Scatter(x=list(f["period"]) + list(f["period"])[::-1], y=list(f["upper"]) + list(f["lower"])[::-1],
                                 fill="toself", fillcolor="rgba(26,163,163,0.15)", line=dict(width=0), name="80% range",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[v.index[-1]] + list(f["period"]), y=[v.iloc[-1]] + list(f["forecast"]), mode="lines",
                                 name="Forecast", line=dict(color=TEAL, width=2.5, dash="dash")))
    return _base(fig, title or f"{L} over time", 380)


def waterfall(table: pd.DataFrame, pop: dict, label: str, group_label: str, top_n: int = 7) -> go.Figure:
    t = table.copy()
    t = t.reindex(t["delta"].abs().sort_values(ascending=False).index)
    head, rest = t.head(top_n), t.iloc[top_n:]
    names = [str(g) for g in head["group"]]
    vals = list(head["delta"])
    if len(rest):
        names.append("All other"), vals.append(float(rest["delta"].sum()))
    fig = go.Figure(go.Waterfall(
        x=[pop["prev_label"]] + names + [pop["latest_label"]], y=[pop["prev"]] + vals + [pop["latest"]],
        measure=["absolute"] + ["relative"] * len(vals) + ["total"],
        increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=NAVY)), connector=dict(line=dict(color="#c5cad6")),
        hovertemplate="%{x}: %{y:,.2f}<extra></extra>"))
    return _base(fig, f"What moved {label}, by {group_label}", 380)


def ranking_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, color: str = BLUE, horizontal: bool = True,
                fmt_kind: str = "intensity") -> go.Figure:
    d = df.iloc[::-1] if horizontal else df
    txt = [f"{x * 100:.1f}%" if fmt_kind == "rate" else num(x) for x in d[value_col]]
    fig = go.Figure(go.Bar(x=d[value_col] if horizontal else d[label_col], y=d[label_col] if horizontal else d[value_col],
                           orientation="h" if horizontal else "v", marker_color=color, text=txt, textposition="outside",
                           cliponaxis=False, hovertemplate="%{y}: %{x:,.2f}<extra></extra>" if horizontal else "%{x}: %{y:,.2f}<extra></extra>"))
    fig = _base(fig, title, max(260, 40 * len(d) + 90))
    fig.update_layout(showlegend=False)
    return fig


def pareto(table: pd.DataFrame, title: str, top_n: int = 12) -> go.Figure:
    t = table.head(top_n)
    cum = t["share"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["group"], y=t["share"], name="Share", marker_color=BLUE))
    fig.add_trace(go.Scatter(x=t["group"], y=cum, name="Cumulative", yaxis="y2", line=dict(color=AMBER, width=2.5),
                             mode="lines+markers"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False, ticksuffix="%"),
                      yaxis=dict(ticksuffix="%"))
    return _base(fig, title, 340)


def histogram(ds: Dataset, measure: str) -> go.Figure:
    v = ds.values(measure).dropna()
    kind = ds.kind(measure)
    fig = go.Figure(go.Histogram(x=v, nbinsx=40, marker_color=BLUE, opacity=0.85))
    fig.add_vline(x=float(v.median()), line_dash="dash", line_color=NAVY, annotation_text="median", annotation_position="top")
    return _base(fig, f"Distribution of {ds.label(measure)}", 300)


def seasonality_bar(season: dict) -> go.Figure:
    p = season["profile"]
    fig = go.Figure(go.Bar(x=p["position"], y=p["effect_pct"], marker_color=[GREEN if x >= 0 else RED for x in p["effect_pct"]],
                           hovertemplate="%{x}: %{y:+.1f}% vs average<extra></extra>"))
    fig.update_yaxes(ticksuffix="%")
    return _base(fig, "Seasonal effect versus average", 300)


def driver_bars(drivers: dict) -> go.Figure:
    d = pd.DataFrame(drivers["drivers"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=d["importance_pct"], y=d["label"], orientation="h", marker_color=TEAL,
                           text=[f"{x:.0f}%" for x in d["importance_pct"]], textposition="outside", cliponaxis=False))
    fig.update_xaxes(ticksuffix="%")
    return _base(fig, "Relative importance of associated fields", max(240, 52 * len(d) + 90))


def corr_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, zmin=-1, zmax=1,
                               colorscale=[[0, RED], [0.5, "#f5f6fa"], [1, BLUE]],
                               text=np.round(matrix.values, 2), texttemplate="%{text}", hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, "Correlation between numeric fields", max(320, 38 * len(matrix) + 120))


def segment_scatter(seg: dict) -> go.Figure:
    s = seg["scatter"]
    fig = go.Figure()
    for i, (name, g) in enumerate(s.groupby("segment")):
        fig.add_trace(go.Scattergl(x=g["x"], y=g["y"], mode="markers", name=name,
                                   marker=dict(size=5, opacity=0.55, color=SEG_COLORS[i % len(SEG_COLORS)])))
    fig.update_xaxes(title="Component 1", showgrid=False)
    fig.update_yaxes(title="Component 2")
    return _base(fig, "Record segments (2-D projection)", 380)


def missing_bar(qtable: pd.DataFrame) -> go.Figure:
    t = qtable[qtable["Missing %"] > 0].sort_values("Missing %", ascending=True)
    fig = go.Figure(go.Bar(x=t["Missing %"], y=t["Field"], orientation="h", marker_color=AMBER,
                           text=[f"{x:.1f}%" for x in t["Missing %"]], textposition="outside", cliponaxis=False))
    fig.update_xaxes(ticksuffix="%")
    return _base(fig, "Missing values by field", max(220, 36 * len(t) + 90))


def health_gauge(score: float) -> go.Figure:
    color = GREEN if score >= 90 else (BLUE if score >= 75 else (AMBER if score >= 60 else RED))
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, number=dict(suffix="/100", font=dict(size=30, color=NAVY)),
                                 gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color, thickness=0.3),
                                            bgcolor="#eef0f6", borderwidth=0)))
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def sparkline(values: list[float], favourable: Optional[bool]) -> go.Figure:
    color = GREY if favourable is None else (GREEN if favourable else RED)
    fig = go.Figure(go.Scatter(y=values, mode="lines", line=dict(color=color, width=2), fill="tozeroy",
                               fillcolor="rgba(138,148,166,0.10)"))
    fig.update_layout(height=46, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
    return fig
