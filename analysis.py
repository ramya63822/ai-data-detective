from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _make_unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in columns:
        name = str(raw).strip() or "Unnamed"
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name}_{count + 1}")
    return out


def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if not pd.api.types.is_object_dtype(df[col]):
            continue
        non_null = df[col].dropna()
        if len(non_null) < 4:
            continue
        parsed = pd.to_datetime(non_null, errors="coerce")
        if float(parsed.notna().mean()) >= 0.90 and parsed.nunique() >= 2:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_data(source: Any, filename: str | None = None) -> pd.DataFrame:
    name = (filename or getattr(source, "name", "dataset.csv")).lower()
    if hasattr(source, "seek"):
        source.seek(0)
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)
    else:
        df = pd.read_csv(source)
    df = df.copy()
    df.columns = _make_unique_columns([str(c) for c in df.columns])
    df = _try_parse_dates(df)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df.reset_index(drop=True)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


def probable_id_columns(df: pd.DataFrame) -> list[str]:
    ids: list[str] = []
    for col in df.columns:
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        name = col.lower()
        if unique_ratio >= 0.98 and ("id" in name or "key" in name or name.endswith("_no")):
            ids.append(col)
    return ids


def probable_target_columns(df: pd.DataFrame) -> list[str]:
    """Heuristic ranking for important measures; never claims semantic certainty."""
    nums = numeric_columns(df)
    scored: list[tuple[float, str]] = []
    keywords = ("revenue", "sales", "price", "amount", "profit", "income", "salary", "cost", "value", "score", "target")
    for col in nums:
        name = col.lower()
        score = 0.0
        score += sum(2.0 for k in keywords if k in name)
        score += min(2.0, np.log10(max(df[col].nunique(dropna=True), 1) + 1) / 2)
        if df[col].nunique(dropna=True) <= 1:
            score -= 10
        scored.append((score, col))
    return [col for _, col in sorted(scored, reverse=True)]


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    dates = datetime_columns(df)

    columns_detail: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        detail: dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
            "unique_pct": round(float(s.nunique(dropna=True) / max(len(df), 1) * 100), 2),
        }
        if col in nums:
            x = s.dropna()
            if not x.empty:
                detail.update({
                    "min": float(x.min()), "mean": float(x.mean()), "median": float(x.median()),
                    "max": float(x.max()), "std": float(x.std()) if len(x) > 1 else 0.0,
                })
        elif col in cats:
            top = s.value_counts(dropna=True).head(5).to_dict()
            detail["top_values"] = {str(k): int(v) for k, v in top.items()}
        elif col in dates:
            x = s.dropna()
            if not x.empty:
                detail["min_date"] = x.min().strftime("%Y-%m-%d")
                detail["max_date"] = x.max().strftime("%Y-%m-%d")
        columns_detail.append(detail)

    completeness = max(0.0, 100.0 - missing / total_cells * 100)
    uniqueness = max(0.0, 100.0 - duplicates / max(len(df), 1) * 100)
    quality_score = round(0.65 * completeness + 0.35 * uniqueness, 1)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": nums,
        "categorical_columns": cats,
        "datetime_columns": dates,
        "probable_id_columns": probable_id_columns(df),
        "probable_measure_columns": probable_target_columns(df)[:8],
        "missing_cells": missing,
        "missing_pct": round(missing / total_cells * 100, 2),
        "duplicate_rows": duplicates,
        "duplicate_pct": round(duplicates / max(len(df), 1) * 100, 2),
        "quality_score": quality_score,
        "columns_detail": columns_detail,
    }
    
def top_group(df, group_by, measure, aggregation="sum", top_n=10):
    if group_by not in df.columns:
        raise ValueError(f"Unknown grouping column: {group_by}")

    if aggregation == "count":
        result = df.groupby(group_by).size()

    else:
        if not measure:
            raise ValueError(
                "A measure column is required for this operation."
            )

        if measure not in df.columns:
            raise ValueError(f"Unknown measure column: {measure}")

        if aggregation == "sum":
            result = df.groupby(group_by)[measure].sum()

        elif aggregation == "average":
            result = df.groupby(group_by)[measure].mean()

        elif aggregation == "maximum":
            result = df.groupby(group_by)[measure].max()

        elif aggregation == "minimum":
            result = df.groupby(group_by)[measure].min()

        else:
            raise ValueError(
                f"Unsupported aggregation: {aggregation}"
            )

    return (
        result
        .sort_values(ascending=False)
        .head(top_n)
        .to_dict()
    )


def execute_operation(df: pd.DataFrame, operation: dict[str, Any]) -> dict[str, Any]:
    op = operation.get("operation")
    column = operation.get("column")
    group_by = operation.get("group_by")
    second_column = operation.get("second_column")
    value = operation.get("value")
    operator = operation.get("operator")
    limit = max(1, min(int(operation.get("limit") or 10), 20))
    measure = operation.get("measure")
    aggregation = operation.get("aggregation", "sum")

    valid = set(df.columns)
    for requested in (column, group_by, second_column):
        if requested is not None and requested not in valid:
            raise ValueError(f"Unknown column: {requested}")

    numeric_ops = {"average", "sum", "minimum", "maximum", "median", "std"}
    if op in numeric_ops:
        if column is None or not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"'{column}' must be a numeric column.")
        s = df[column].dropna()
        if s.empty:
            raise ValueError(f"Column '{column}' has no usable numeric values.")
        if op == "average": result = s.mean()
        elif op == "sum": result = s.sum()
        elif op == "minimum": result = s.min()
        elif op == "maximum": result = s.max()
        elif op == "median": result = s.median()
        else: result = s.std() if len(s) > 1 else 0.0
        return {"operation": op, "column": column, "result": float(result), "rows_used": int(len(s))}

    if op == "count":
        return {"operation": op, "result": int(len(df))}

    if op in {"group_average", "group_sum", "group_count"}:
        if group_by is None:
            raise ValueError("group_by is required.")
        if op != "group_count" and (column is None or not pd.api.types.is_numeric_dtype(df[column])):
            raise ValueError("A numeric value column is required.")
        if op == "group_average": grouped = df.groupby(group_by, dropna=False)[column].mean()
        elif op == "group_sum": grouped = df.groupby(group_by, dropna=False)[column].sum()
        else: grouped = df.groupby(group_by, dropna=False).size()
        grouped = grouped.sort_values(ascending=False).head(limit)
        return {"operation": op, "group_by": group_by, "column": column,
                "result": {str(k): float(v) for k, v in grouped.to_dict().items()}}

    if op == "top_category":
        if column is None:
            raise ValueError("A categorical column is required.")
        counts = df[column].value_counts(dropna=False).head(limit)
        return {"operation": op, "column": column, "result": {str(k): int(v) for k, v in counts.to_dict().items()}}

    if op == "value_count":
        if column is None:
            raise ValueError("A column is required.")
        return {"operation": op, "column": column, "value": value,
                "result": int(df[column].astype(str).eq(str(value)).sum())}

    if op in {"filter_count", "filter_average", "filter_sum"}:
        if column is None or operator is None or not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError("Filtering requires a numeric column and comparison operator.")
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Filter value must be numeric.") from exc
        s = df[column]
        if operator == "eq": mask = s == threshold
        elif operator == "gt": mask = s > threshold
        elif operator == "gte": mask = s >= threshold
        elif operator == "lt": mask = s < threshold
        else: mask = s <= threshold
        filtered = df.loc[mask]
        if op == "filter_count": result = len(filtered)
        elif op == "filter_average": result = filtered[column].mean()
        else: result = filtered[column].sum()
        return {"operation": op, "column": column, "operator": operator, "value": threshold,
                "rows_matched": int(len(filtered)), "result": None if pd.isna(result) else float(result)}

    if op == "correlation":
        if column is None or second_column is None:
            raise ValueError("Two numeric columns are required.")
        if not pd.api.types.is_numeric_dtype(df[column]) or not pd.api.types.is_numeric_dtype(df[second_column]):
            raise ValueError("Correlation requires two numeric columns.")
        r = df[column].corr(df[second_column])
        return {"operation": op, "columns": [column, second_column], "result": None if pd.isna(r) else float(r)}

    if op == "missing_values":
        vals = df.isna().sum().sort_values(ascending=False).to_dict()
        return {"operation": op, "result": {str(k): int(v) for k, v in vals.items()}}

    if op == "duplicate_rows":
        return {"operation": op, "result": int(df.duplicated().sum())}

    if op == "dataset_summary":
        return {"operation": op, "result": profile_dataset(df)}

    if op == "period_change":
        if not datetime_columns(df) or column is None or not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError("Period comparison requires a datetime column and numeric measure.")
        date_col = operation.get("date_column") or datetime_columns(df)[0]
        if date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            raise ValueError("Invalid date column.")
        x = df[[date_col, column]].dropna().sort_values(date_col)
        if len(x) < 4:
            raise ValueError("Not enough observations for a period comparison.")
        midpoint = x[date_col].min() + (x[date_col].max() - x[date_col].min()) / 2
        first = x[x[date_col] <= midpoint][column].mean()
        second = x[x[date_col] > midpoint][column].mean()
        pct = None if first == 0 else (second - first) / abs(first) * 100
        return {"operation": op, "date_column": date_col, "column": column,
                "first_period_mean": float(first), "second_period_mean": float(second),
                "percent_change": None if pct is None else float(pct)}

    if op == "time_trend":
        dates = datetime_columns(df)
        if not dates or column is None or not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError("Time trend requires a datetime column and numeric measure.")
        date_col = operation.get("date_column") or dates[0]
        freq = operation.get("frequency") or "ME"
        x = df[[date_col, column]].dropna().set_index(date_col).resample(freq)[column].agg(["count", "mean", "sum"]).dropna()
        x = x.tail(limit)
        return {"operation": op, "date_column": date_col, "column": column,
                "frequency": freq, "result": x.round(4).reset_index().astype(str).to_dict(orient="records")}
        
    if op == "top_group":
        result = top_group(
        df=df,
        group_by=group_by,
        measure=measure,
        aggregation=aggregation)

    return {
        "operation": "top_group",
        "group_by": group_by,
        "measure": measure,
        "aggregation": aggregation,
        "result": result
    }

    raise ValueError(f"Unsupported operation: {op}")

