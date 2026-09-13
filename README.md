# AI Data Detective

A local AI-powered data analysis tool that uses Python, Pandas, Matplotlib, and Ollama to analyze datasets and generate useful insights.

## Features

* Load CSV datasets
* Profile datasets
* Answer natural-language questions
* Detect numerical anomalies
* Generate data visualizations
* Generate AI-powered insights using Ollama

## Tech Stack

* Python
* Pandas
* Matplotlib
* Ollama
* Qwen 3 4B

## How It Works

```text
CSV Dataset
     ↓
   Pandas
     ↓
Data Analysis
     ↓
┌────┼────────────┐
↓    ↓            ↓
Queries  Anomalies  Visualization
└────┼────────────┘
     ↓
   Ollama
     ↓
 AI-generated Insights
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the environment

Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the Ollama model

```bash
ollama pull qwen3:4b
```

### 5. Run the application

```bash
python app.py
```

## Future Improvements

* Excel support
* SQLite database support
* Automatic chart selection
* More advanced anomaly detection
* Streamlit web interface
* Support for cloud LLM providers
