from analysis import load_data, get_sales_by_product
from anomaly import detect_anomalies
from llm import ask_llm


df = load_data("data/sales.csv")

print("\n--- DATASET ---")
print(df)

anomalies = detect_anomalies(df)

print("\n--- ANOMALIES ---")

if not anomalies:
    print("No obvious numerical anomalies detected.")
else:
    for column, rows in anomalies.items():
        print(f"\nUnusual values found in: {column}")
        print(rows)


sales_by_product = get_sales_by_product(df)

prompt = f"""
You are a data analyst.

Product sales:
{sales_by_product.to_string()}

Detected anomalies:
{anomalies}

Provide:
1. The most important finding
2. Any unusual behavior
3. One practical recommendation
"""

insights = ask_llm(prompt)

print("\n--- AI INSIGHTS ---")
print(insights)

from query import answer_question

# Ask the user a question
question = input("\nAsk a question about the dataset: ")

answer = answer_question(df, question)

print("\n--- ANSWER ---")
print(answer)