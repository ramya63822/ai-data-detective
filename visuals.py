from __future__ import annotations

import pandas as pd
import plotly.express as px


def auto_charts(df: pd.DataFrame, max_charts: int = 4) -> list:
    charts = []
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    dates = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    if dates and numeric:
        date_col, value_col = dates[0], numeric[0]
        x = df[[date_col, value_col]].dropna().sort_values(date_col)
        if len(x) >= 3:
            trend = x.set_index(date_col)[value_col].resample("ME").mean().dropna().reset_index()
            if not trend.empty:
                charts.append(px.line(trend, x=date_col, y=value_col, markers=True, title=f"{value_col} — monthly trend"))

    if categorical and numeric:
        category = min(categorical, key=lambda c: df[c].nunique(dropna=True) if df[c].nunique(dropna=True) else 10)
        value = numeric[0]
        grouped = df.groupby(category, dropna=False)[value].mean().sort_values(ascending=False).head(15).reset_index()
        if not grouped.empty:
            charts.append(px.bar(grouped, x=category, y=value, title=f"Average {value} by {category}"))

    if len(numeric) >= 2:
        x, y = numeric[0], numeric[1]
        scatter = df[[x, y]].dropna()
        if not scatter.empty:
            charts.append(px.scatter(scatter, x=x, y=y, title=f"{y} vs {x}", trendline="ols" if len(scatter) >= 10 else None))

    if numeric:
        charts.append(px.histogram(df, x=numeric[0], title=f"Distribution of {numeric[0]}"))

    return charts[:max_charts]
