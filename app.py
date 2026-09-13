import pandas as pd

df = pd.read_csv("data/sales.csv")

print("Dataset:")
print(df)

print("\nDataset Information:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())