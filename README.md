# 🔎 AI Data Detective — Decision Intelligence & Analytics Copilot

AI Data Detective turns a single CSV or Excel dataset into an exploratory analytics workspace, then lets users ask follow-up questions in plain language. It is designed for many structured datasets; the available analyses depend on the columns and data quality.

## What it does

- **Proactive analytics:** KPIs, executive summary, important findings, risks, opportunities, and recommended investigations.
- **Data preparation and quality:** common text cleanup, type/date detection, schema inference, missingness, duplicates, identifier checks, and a data-health score.
- **Performance analysis:** rankings, contribution/concentration, changes, trends, and period comparisons when a suitable time field exists.
- **Advanced analysis when supported:** seasonality and back-tested forecasting, anomaly detection, potential drivers, correlations, and clustering.
- **AI Analyst:** natural-language questions, exact record/ID lookup, and evidence-grounded interpretation through OpenAI.
- **Reports:** downloadable HTML with charts and Markdown.

## Design principle: verified numbers first

```text
Upload → clean/profile → deterministic analytics → verified evidence
       → optional OpenAI planning/interpretation → dashboard and answers
```

Pandas, SciPy, statsmodels, and scikit-learn calculate the figures. OpenAI helps interpret questions and phrase explanations; it is not trusted to perform arithmetic. If a field or time axis required for a question is unavailable, the app should state the limitation rather than substitute an unrelated field. Correlation and model importance are associations, not proof of causation.

The app can still run deterministic analytics without an OpenAI key; AI planning and narration will be unavailable in that case.

## Run locally (Windows PowerShell)

Use Python 3.11 or newer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your key:

```env
AI_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-6-luna
```

Then run:

```powershell
streamlit run web_app.py
```

In the sidebar, select **Test OpenAI connection**. A successful test verifies that the app can reach the API with the configured key and model. It makes one small API request and may incur a small usage charge. If you have not configured a key yet, the deterministic analytics still work.

Run tests with:

```powershell
python -m pytest -q
```

## Deploy to Streamlit Community Cloud

1. Push the project to a GitHub repository. Keep `.env` and `.streamlit/secrets.toml` untracked; `.gitignore` excludes them.
2. In Streamlit Community Cloud, create an app from that repository, select your branch, and set the entrypoint to `web_app.py`.
3. In the app's **Secrets** settings, enter:

```toml
AI_PROVIDER = "openai"
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL = "gpt-6-luna"
```

4. Deploy, open the sidebar, and click **Test OpenAI connection** before testing the AI Analyst.

`requirements.txt` is in the repository root for the cloud build. Do not paste API keys into Python source code or commit them to GitHub. The app reads Streamlit secrets on cloud deployments and `.env`/environment variables locally.

## Main UI

1. **Executive Overview** — summary, key findings, risks, opportunities, next investigations, and significant changes.
2. **Performance & Trends** — measure selection, period comparisons, trends, forecasts, seasonality, rankings, and contributions where supported.
3. **Drivers & Segments** — potential predictors, correlations, clustering, and distributions.
4. **Anomalies** — unusual records and periods.
5. **Data Health** — quality score, data issues, missingness, and inferred column roles.
6. **AI Analyst** — natural-language questions and evidence-backed tables/charts.
7. **Report** — downloadable HTML and Markdown reports.

The sidebar contains dataset upload, an optional **Demo Workspace**, provider status, and a manual OpenAI connection check.

## Project layout

- `web_app.py` — Streamlit interface
- `detective/dataset.py` — ingestion, cleaning, and schema/role inference
- `detective/quality.py` — data-quality assessment
- `detective/engine.py` — analytics orchestration and findings
- `detective/timeseries.py` — period comparisons, trends, seasonality, forecasts
- `detective/performance.py` — rankings, contribution, and change decomposition
- `detective/drivers.py` — predictive driver analysis
- `detective/anomalies.py` — unusual-record detection
- `detective/segments.py` — clustering and segment profiling
- `detective/copilot.py` — natural-language intent, plan validation, analysis handlers, answer composition
- `detective/llm.py` — OpenAI integration, connection check, and grounded narration
- `detective/visuals.py` — Plotly charts
- `detective/report.py` — HTML and Markdown reports
- `tests/` — analytics and LLM integration/configuration tests

## Important limitations

- Analyses operate on one table at a time; multiple files are not joined automatically.
- Column-role inference is heuristic. Clear column names help.
- Driver importance and clusters are associations, not verified causes or predefined personas.
- Forecasts assume historical patterns continue and are shown only when enough suitable time data exists.
- Do not upload confidential customer, employee, student, or production data to a public portfolio demo. The app is not an authenticated, multi-tenant production service.
