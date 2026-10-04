from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(dataframe: pd.DataFrame, max_rows: int = 25) -> dict:
    """Run robust IQR checks and, when enough data exists, Isolation Forest."""
    results = {}
    numeric_columns = dataframe.select_dtypes(include="number").columns

    for column in numeric_columns:
        series = dataframe[column].dropna()
        if len(series) < 4 or series.nunique() < 2:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            iqr_mask = series != series.median()
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_mask = (series < lower) | (series > upper)

        iqr_rows = dataframe.loc[iqr_mask.index[iqr_mask]]

        method = "IQR"
        flagged = iqr_rows

        # Isolation Forest is useful only when there is enough data to learn a pattern.
        if len(series) >= 20:
            values = series.to_numpy().reshape(-1, 1)
            contamination = min(0.10, max(0.02, 5 / len(series)))
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
            )
            labels = model.fit_predict(values)
            iso_index = series.index[labels == -1]
            iso_rows = dataframe.loc[iso_index]
            combined_index = sorted(set(flagged.index).union(set(iso_rows.index)))
            flagged = dataframe.loc[combined_index]
            method = "IQR + Isolation Forest"

        if not flagged.empty:
            results[column] = {
                "method": method,
                "count": int(len(flagged)),
                "rows": flagged.head(max_rows),
            }

    return results
