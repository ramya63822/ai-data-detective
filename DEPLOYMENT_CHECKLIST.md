# AI Data Detective — release smoke-test checklist

Run this checklist on the same commit you intend to publish.

## Local verification

- [ ] Create/activate a clean Python 3.11+ virtual environment.
- [ ] Install dependencies with `pip install -r requirements.txt`.
- [ ] Run `python -m pytest -q` and verify all tests pass.
- [ ] Start `streamlit run web_app.py` and verify the landing page opens.
- [ ] Expand **Demo Workspace** and load each built-in sample.
- [ ] Open all seven tabs for each sample and look for errors or empty visualizations that should not be empty.
- [ ] Upload a small CSV and an Excel file of your own; verify row/column counts and column roles.
- [ ] In **AI Analyst**, test an aggregate, a ranking ("Which product sold the most?"), a direct lookup ("Give me some employee IDs"), a time-based question where dates exist, and an unsupported question.
- [ ] Download and open both HTML and Markdown reports.

## OpenAI verification

- [ ] Set `AI_PROVIDER=openai`, `OPENAI_API_KEY`, and `OPENAI_MODEL=gpt-6-luna` locally or in the deployment's secret manager.
- [ ] Click **Test OpenAI connection** and confirm it succeeds. This sends one small API request and may incur a small usage charge.
- [ ] Ask the AI Analyst two or three questions and verify answers include actual figures/records from the data.
- [ ] Temporarily remove the key and confirm the deterministic dashboard still loads without LLM interpretation.
- [ ] Test an invalid key or unavailable model in a non-production session; confirm the app falls back without exposing keys or stack traces.

## Before making the demo public

- [ ] Confirm `.env` and `.streamlit/secrets.toml` are not tracked by Git.
- [ ] Do not use confidential company, student, customer, or employee data in the public demo.
- [ ] Add screenshots and a short demo walkthrough to the GitHub README.
- [ ] Deploy from the intended repository, branch, and `web_app.py` entrypoint.
- [ ] Repeat the upload, AI connection, and report checks on the public URL.

Passing unit tests alone does not prove the hosted UI or a live OpenAI request works. Record those as separate checks.
