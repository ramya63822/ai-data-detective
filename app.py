import pandas as pd
import ollama

# Load dataset
df = pd.read_csv("data/sales.csv")

# Generate dataset summary
summary = f"""
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column types:
{df.dtypes}

Missing values:
{df.isnull().sum()}

Statistics:
{df.describe()}
"""

# Send the summary to Ollama
response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": f"""
You are a data analyst.

Analyze this dataset information and explain:
1. What kind of data this appears to contain
2. Important patterns
3. Potential data-quality issues
4. Three useful insights

Dataset information:
{summary}
"""
        }
    ]
)

print("\n--- AI ANALYSIS ---")
print(response["message"]["content"])