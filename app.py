import pandas as pd

# Load dataset
df = pd.read_csv("data/sales.csv")

print("\n--- DATASET PREVIEW ---")
print(df.head())

print("\n--- SHAPE ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- COLUMN TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- UNIQUE VALUES ---")
print(df.nunique())

print("\n--- NUMERICAL SUMMARY ---")
print(df.describe())