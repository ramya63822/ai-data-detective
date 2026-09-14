from llm import ask_llm
from analysis import get_sales_by_product, get_total_sales, get_average_price


def classify_question(question, columns):
    prompt = f"""
You are a data analysis assistant.

Dataset columns:
{list(columns)}

User question:
{question}

Classify the question into ONE category:

highest_product_sales
total_sales
average_price
dataset_summary

Return ONLY the category name.
"""

    return ask_llm(prompt).strip().lower()


def answer_question(df, question):
    operation = classify_question(question, df.columns)

    if operation == "highest_product_sales":
        sales = get_sales_by_product(df)
        product = sales.idxmax()
        value = sales.max()

        result = f"""
Highest selling product: {product}
Quantity sold: {value}
"""

    elif operation == "total_sales":
        total = get_total_sales(df)

        result = f"""
Total sales: ₹{total:,.2f}
"""

    elif operation == "average_price":
        average = get_average_price(df)

        result = f"""
Average price: ₹{average:,.2f}
"""

    elif operation == "dataset_summary":
        result = f"""
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Statistics:
{df.describe().to_string()}
"""

    else:
        return "Sorry, I don't understand that question yet."

    # Ask Ollama to explain the result
    explanation_prompt = f"""
You are a helpful data analyst.

User question:
{question}

Exact result calculated by Python:
{result}

Explain this result clearly in 2-4 sentences.
Do not invent any numbers or facts.
"""

    explanation = ask_llm(explanation_prompt)

    return explanation