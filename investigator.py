from analysis import get_sales_by_product
from anomaly import detect_anomalies
from llm import ask_llm


def investigate_dataset(df):
    findings = []

    # Basic information
    findings.append(
        f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
    )

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        findings.append(
            f"Missing values were found in: {missing.to_dict()}"
        )
    else:
        findings.append("No missing values were detected.")

    # Highest-selling product
    if "product" in df.columns and "quantity" in df.columns:
        product_sales = get_sales_by_product(df)

        top_product = product_sales.idxmax()
        top_quantity = product_sales.max()

        findings.append(
            f"{top_product} has the highest quantity sold with "
            f"{top_quantity} units."
        )

    # Highest-selling region
    if "region" in df.columns and "quantity" in df.columns:
        region_sales = df.groupby("region")["quantity"].sum()

        top_region = region_sales.idxmax()
        top_region_quantity = region_sales.max()

        findings.append(
            f"{top_region} has the highest quantity sold with "
            f"{top_region_quantity} units."
        )

    # Price range
    if "price" in df.columns:
        min_price = df["price"].min()
        max_price = df["price"].max()

        findings.append(
            f"Prices range from ₹{min_price:,.2f} to ₹{max_price:,.2f}."
        )

    # Anomalies
    anomalies = detect_anomalies(df)

    if anomalies:
        for column, rows in anomalies.items():
            findings.append(
                f"Potential anomalies were detected in {column}: "
                f"{len(rows)} unusual row(s)."
            )
    else:
        findings.append("No obvious numerical anomalies were detected.")

    # Ask Ollama to interpret the findings
    findings_text = "\n".join(f"- {finding}" for finding in findings)

    prompt = f"""
You are an expert data analyst.

You have been given the following findings from a dataset:

{findings_text}

Create a concise investigation report with these sections:

1. KEY FINDINGS
2. DATA QUALITY
3. ANOMALIES
4. RECOMMENDATION

Use only the information provided.
Do not invent numbers or facts.
"""

    return ask_llm(prompt)