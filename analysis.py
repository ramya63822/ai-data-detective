import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def get_sales_by_product(df):
    return df.groupby("product")["quantity"].sum()


def get_total_sales(df):
    return (df["quantity"] * df["price"]).sum()


def get_average_price(df):
    return df["price"].mean()