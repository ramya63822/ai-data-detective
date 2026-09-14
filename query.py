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
    category = classify_question(question, df.columns)

    if category == "highest_product_sales":
        sales = get_sales_by_product(df)
        product = sales.idxmax()
        value = sales.max()

        return f"Highest selling product: {product}\nQuantity sold: {value}"

    elif category == "total_sales":
        total = get_total_sales(df)
        return f"Total sales: ₹{total:,.2f}"

    elif category == "average_price":
        average = get_average_price(df)
        return f"Average price: ₹{average:,.2f}"

    elif category == "dataset_summary":
        return (
            f"Rows: {df.shape[0]}\n"
            f"Columns: {df.shape[1]}\n\n"
            f"{df.describe()}"
        )

    return "Sorry, I don't understand that question yet."