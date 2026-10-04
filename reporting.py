from __future__ import annotations

from datetime import datetime

from decision_engine import build_evidence


def build_report(dataset_name: str, df, investigation: str) -> str:
    evidence = build_evidence(df)
    p = evidence["profile"]
    return f"""# AI Data Detective — Decision Intelligence Report

**Dataset:** {dataset_name}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  

## Dataset Health

| Metric | Value |
|---|---:|
| Records | {p['rows']:,} |
| Features | {p['columns']:,} |
| Missing cells | {p['missing_cells']:,} |
| Duplicate rows | {p['duplicate_rows']:,} |
| Data health | {p['quality_score']}/100 |

## Schema

- Numeric: {', '.join(p['numeric_columns']) or 'None'}
- Categorical: {', '.join(p['categorical_columns']) or 'None'}
- Datetime: {', '.join(p['datetime_columns']) or 'None'}

## Executive Investigation

{investigation}

## Analytical Evidence

### Numeric Measures
{evidence['metrics']}

### Trends
{evidence['trends'] or 'No reliable time-series evidence detected.'}

### Potential Drivers
{evidence['drivers'] or 'No suitable correlation evidence detected.'}

*Calculations and anomaly detection are performed by Python. The LLM is used for analytical planning and explanation.*
"""
