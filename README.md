# 🔎 AI Data Detective

**AI-powered decision intelligence for heterogeneous CSV and Excel datasets.**

AI Data Detective is not a generic CSV chatbot. It combines deterministic analytics with an LLM orchestration layer to produce evidence-grounded executive insights, data-quality diagnostics, anomaly findings, adaptive visualizations, and natural-language analytical workflows.

## Core Architecture

```text
CSV / Excel
    ↓
Schema & Data Profiling
    ↓
Data Quality + Anomaly Detection
    ↓
Evidence Engine
    ├── Metrics
    ├── Trends
    ├── Potential Drivers
    └── Concentration / Segmentation Signals
    ↓
LLM Query Planner
    ↓
Structured + Validated Plan
    ↓
Whitelisted Pandas Analytics
    ↓
Verified Evidence
    ↓
LLM Explanation / Executive Brief
    ↓
Streamlit Decision Workspace
```

## Product Capabilities

- CSV/XLSX/XLS ingestion
- Automatic schema/type/date detection
- Data health scoring
- Missing-value and duplicate diagnostics
- IQR + Isolation Forest anomaly detection
- Adaptive Plotly visualizations
- Period-over-period movement when dates are available
- Correlation-based driver screening
- Categorical concentration signals
- Natural-language analytics across arbitrary schemas
- Structured LLM query planning using Pydantic validation
- Whitelisted analytical operations only; no arbitrary code execution
- Transparent analytical trace
- Executive investigation brief
- Downloadable Markdown decision report
- Cloud LLM support through OpenAI
- Local fallback through Ollama

## Example Business Questions

The system is designed around analytical workflows, not a fixed list of questions:

- What changed over the available period?
- Which segment has the highest average value?
- What are the strongest observed numeric relationships?
- Which groups contribute the most records or value?
- How many records meet a threshold?
- Which fields have data-quality issues?
- Summarize the most important risks and patterns.
- What should I investigate next?

## Engineering Principles

### Evidence first
Python computes statistics and anomaly signals. The LLM does not calculate business metrics from memory.

### Constrained AI
The model returns a structured analytical plan. The application validates columns and operations against an explicit allow-list before execution.

### Provider agnostic
The application can use OpenAI in production and Ollama locally. The analytics engine is independent of the model provider.

### Graceful degradation
If the cloud model is unavailable, the configured local model can be used for AI explanations. Deterministic analytics remain available independently of the LLM.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your provider.

For OpenAI:

```text
AI_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-5.6-luna
```

For local development:

```text
AI_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
```

Then run:

```powershell
streamlit run web_app.py
```

## Portfolio Positioning

> Built an AI-powered decision-intelligence platform that profiles heterogeneous datasets, scores data quality, detects anomalies, analyzes trends and potential drivers, supports natural-language analytical workflows through structured LLM planning, and generates evidence-grounded executive reports.

## Important Limitation

Correlation and anomaly detection are analytical signals, not proof of causation or confirmed data errors. Forecasting and causal inference should be added as separate validated analytical modules rather than inferred by the LLM.
