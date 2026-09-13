def detect_anomalies(dataframe):
    anomalies = {}

    numeric_columns = dataframe.select_dtypes(include="number").columns

    for column in numeric_columns:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        unusual_rows = dataframe[
            (dataframe[column] < lower_bound)
            | (dataframe[column] > upper_bound)
        ]

        if not unusual_rows.empty:
            anomalies[column] = unusual_rows

    return anomalies