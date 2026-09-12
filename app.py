import pandas as pd
import ollama

# Load dataset
df = pd.read_csv("data/sales.csv")

# Detect numerical anomalies using IQR
def detect_anomalies(dataframe):
    anomalies = {}

    numeric_columns = dataframe.select_dtypes(include="number").columns

    for column in numeric_columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        unusual_rows = dataframe[
            (dataframe[column] < lower_bound)
            | (dataframe[column] > upper_bound)
        ]

        if not unusual_rows.empty:
            anomalies[column] = unusual_rows

    return anomalies


anomalies = detect_anomalies(df)

print("\n--- ANOMALY REPORT ---")

if not anomalies:
    print("No obvious numerical anomalies detected.")
else:
    for column, rows in anomalies.items():
        print(f"\nUnusual values found in: {column}")
        print(rows)

import matplotlib.pyplot as plt

# Create a simple sales chart
sales_by_product = df.groupby("product")["quantity"].sum()

plt.figure(figsize=(8, 5))
sales_by_product.plot(kind="bar")

plt.title("Quantity Sold by Product")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.tight_layout()

plt.savefig("sales_by_product.png")
plt.show()

insights = f"""
Product sales:

{sales_by_product.to_string()}

Detected anomalies:

{anomalies}
"""

response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": f"""
You are a data analyst.

Analyze these results and provide:
1. The most important finding
2. Any unusual behavior
3. A practical recommendation

Results:
{insights}
"""
        }
    ]
)

print("\n--- AI INSIGHTS ---")
print(response["message"]["content"])

print("Available columns:")
print(", ".join(df.columns))

question = input("\nAsk a question about the dataset: ")

# Ask Ollama to identify the type of question
response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": f"""
You are a data analysis assistant.

The dataset has these columns:
{list(df.columns)}

User question:
{question}

Classify the question into ONE of these categories:
1. highest_product_sales
2. total_sales
3. average_price
4. dataset_summary

Return ONLY the category name.
"""
        }
    ]
)

category = response["message"]["content"].strip().lower()

print("\nDetected question type:", category)

# Perform the actual calculation with Python
if category == "highest_product_sales":
    sales = df.groupby("product")["quantity"].sum()
    product = sales.idxmax()
    value = sales.max()

    print(f"\nHighest selling product: {product}")
    print(f"Quantity sold: {value}")

elif category == "total_sales":
    total = (df["quantity"] * df["price"]).sum()

    print(f"\nTotal sales: ₹{total:,.2f}")

elif category == "average_price":
    average = df["price"].mean()

    print(f"\nAverage price: ₹{average:,.2f}")

elif category == "dataset_summary":
    print("\nDataset Summary:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(df.describe())

else:
    print("\nSorry, I don't understand that question yet.")