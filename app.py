import pandas as pd
import ollama

# Load dataset
df = pd.read_csv("data/sales.csv")

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