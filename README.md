# DataMind - AI🤖


**An autonomous multi-agent analytics copilot.** Upload any CSV, and a pipeline of specialized AI agents profiles it, cleans it, trains and compares ML models, explains *why* the model predicts what it predicts, scores data quality, checks its own outputs for consistency, and writes an executive report — no code required.

Built with Flask + Groq LLaMA 3.3 70B + scikit-learn/XGBoost/LightGBM/CatBoost + SHAP + statsmodels.

**Try it without uploading anything**: the upload page has a "Try it instantly" button that runs the full pipeline on a bundled synthetic HR attrition dataset.

## 🌐 Live Demo

**Portfolio:** https://datamind-ai-887682911552.asia-south1.run.app/

# 📸 Screenshots
<img width="1907" height="875" alt="datamind-3" src="https://github.com/user-attachments/assets/7817257e-9b3a-4437-a7e2-59478f89400f" />

<img width="1909" height="881" alt="datamind-1" src="https://github.com/user-attachments/assets/6bf0fd2f-6a3d-4a65-a78c-a3b429a3d975" />



---

## What makes this more than a student ML project

Most CSV-analysis tools stop at "train a model, print a summary." DataMind adds three things most similar projects skip entirely:

- **RAG-grounded chat** — ask follow-up questions about your analysis and get answers retrieved from the *full* analysis output, not a fixed slice of the first few insights
- **Guardrails/Eval agent** — before any report reaches you, a dedicated agent checks whether the reporting LLM's own claims (confidence, risk score, "should I worry") are internally consistent and actually supported by the real numbers
- **LLMOps tracking** — every LLM call (Analyst, Reporter, Chat) is logged with latency, token usage, and success/failure, queryable per job — not just a job-level summary

Every one of these is real, working code — not a prompt asking the LLM to "use RAG" or "be careful about hallucination." See [ARCHITECTURE.md](ARCHITECTURE.md) for the full reasoning.

---

## Project Structure

```
DataMind/
├── app.py                       ← Flask backend (orchestrator, routes, auth, job queue)
├── models.py                     ← SQLAlchemy models: users, experiment history, LLM call logs
├── agents/
│   ├── detective.py             ← Agent 1: cleaning, profiling, Data Quality Center (5-dim score)
│   ├── analyst.py                ← Agent 2: Groq LLM insights + target/task detection
│   ├── ml_engineer.py            ← Agent 3: AutoML — classification & regression leaderboard
│   ├── clustering.py             ← Agent 3b: KMeans / DBSCAN / Hierarchical clustering
│   ├── forecasting.py            ← Agent 3c: ARIMA / SARIMA / Prophet time-series forecasting
│   ├── explainability.py         ← Agent 4: SHAP summary + waterfall plots, plain-English captions
│   ├── reporter.py               ← Agent 5: Groq LLM executive report
│   ├── evaluator.py              ← Agent 6: Guardrails — validates Reporter's claims for consistency
│   ├── rag_agent.py              ← Agent 7: RAG — TF-IDF retrieval over the full analysis output
│   ├── llmops.py                 ← Tracks latency/tokens/success for every Groq call
│   ├── industry_intelligence.py  ← Agent 8: industry-specific KPI mapping (banking, retail, HR, ...)
│   ├── dashboard_agent.py        ← Agent 9: self-contained Plotly HTML dashboard generator
│   └── report_generator.py       ← PDF report (reportlab) — quality, risk, execution summary, industry
├── templates/
│   ├── index.html / dist/        ← built frontend bundle
│   └── src/                      ← React source
├── sample_data/                  ← bundled demo dataset for the "Try it instantly" button
├── tests/                        ← pytest suite (45 tests across every agent + auth/history)
├── .github/workflows/tests.yml   ← CI — runs the suite on every push/PR
├── uploads/                      ← CSV files stored here (auto-created, gitignored)
├── requirements.txt / requirements-dev.txt
├── Dockerfile / docker-compose.yml
├── render.yaml                   ← Render Blueprint
├── railway.json / Procfile       ← Railway / Heroku-style deploy
├── .env.example                  ← copy to .env for local dev
├── ARCHITECTURE.md               ← system design, diagrams, and the reasoning behind key decisions
└── README.md
```

---

## Setup & Run (local)

### 1 — Configure environment variables
```bash
cp .env.example .env
```
Edit `.env`:
```
GROQ_API_KEY=gsk_your_actual_key_here   # https://console.groq.com/keys
SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
```

### 2 — Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```
XGBoost, LightGBM, CatBoost, statsmodels, and SHAP are all listed as dependencies but the app degrades gracefully if any single one fails to install — it just drops that model/feature rather than crashing. Prophet is commented out by default since it needs a C++ build toolchain; ARIMA/SARIMA (statsmodels) already cover forecasting without it. RAG retrieval uses scikit-learn's TF-IDF (already a dependency) rather than sentence-transformers/FAISS, for the same reason — no heavy model download at runtime.

### 4 — Run
```bash
python app.py
```
Open **http://localhost:5000**.

---

## The Agent Pipeline

| Agent | What it does |
|---|---|
| 🕵️ Detective | Loads the CSV, fixes missing values, detects outliers/duplicates/date columns, and computes a **Data Quality Center** score: Completeness, Accuracy, Consistency, Validity, Uniqueness → one composite /100 with a grade and concrete recommendations |
| 📊 Analyst | Sends the data profile to Groq LLaMA 3.3 70B for structured insights, and recommends a target column + task type |
| 🤖 ML Engineer | Trains and compares Logistic/Ridge Regression, Random Forest, Gradient Boosting, KNN, XGBoost, LightGBM, and CatBoost (whichever are installed), and auto-selects the best by cross-validated score |
| 🔵 Clustering | When there's no clear target (or `mode=clustering`), segments records with KMeans (auto-k via silhouette), DBSCAN, and Hierarchical clustering, with a PCA 2D projection and per-segment feature profiles |
| 📈 Forecasting | For date + numeric-target data (or `mode=forecasting`), backtests ARIMA/SARIMA/Prophet and forecasts forward with confidence intervals |
| 🔍 Explainability | Generates SHAP global summary + local waterfall plots with plain-English captions — "why does the model think X?" |
| 📝 Reporter | Asks Groq for a plain-English executive summary, risk score, confidence, and next steps |
| 🛡️ Guardrails / Evaluator | Checks Reporter's own claims against the real numbers — does "high confidence" actually match the quality score and sample size? Does the risk score agree with "should I worry"? Flags contradictions and small-sample overconfidence before the report is shown |
| 🔎 RAG Agent | Indexes every insight, action, quality recommendation, and SHAP caption from the run so the chat assistant can retrieve exactly what's relevant to a given question, instead of always repeating the same fixed slice |
| 🏭 Industry Intelligence | Optionally maps results onto industry-specific KPIs (Banking, Finance, Retail, Healthcare, Education, Manufacturing, HR) |
| 📊 Dashboard | Assembles everything into one self-contained Plotly HTML dashboard (gauge, radar, leaderboard, SHAP plots, cluster/forecast charts) |
| 📄 PDF Report | Executive Summary, AI Agent Contributions, Data Quality Center, Key Insights, ML Leaderboard, Prediction Confidence, Risk Analysis, AI Execution Summary (RAG/Guardrails/LLMOps), Executive Conclusion |

### Requesting a specific mode
`POST /api/analyze` accepts:
- `mode` — `auto` (default), `classification`, `regression`, `clustering`, or `forecasting`
- `industry` — `banking`, `finance`, `retail`, `healthcare`, `education`, `manufacturing`, or `hr` (optional)
- `target_column` — override automatic target detection

---

## Human-in-the-Loop (HIL) review

If Guardrails/Eval flags or fails a run (e.g. the LLM claims high confidence on a low-quality, small-sample dataset), the job doesn't auto-complete. It holds at `needs_review` instead:

- `GET /api/status/<job_id>` returns `review_required: true` plus the exact flags raised, so a reviewer can see *why* it's held
- `POST /api/jobs/<job_id>/review` with `{"decision": "approve" | "reject", "note": "..."}` lets a human make the final call
- Approving unlocks the job exactly like a normal completed run (dashboard, PDF, chat) — rejecting keeps results locked with the reviewer's note attached

---

## RAG-Powered Chat

`/api/chat` retrieves the top-6 most relevant chunks from the full analysis (not just the first few insights) using TF-IDF + cosine similarity, and grounds every answer in that retrieved evidence. Ask about a specific feature's importance, a quality recommendation, or an industry KPI that wasn't in the headline insights — it can find and answer from it.

---

## LLMOps

Every Groq call (Analyst, Reporter, Chat) is wrapped and logged to `LLMCallLog`: which agent, latency, prompt/completion tokens, and success/failure. Query it per job:

```
GET /api/llm_calls/<job_id>
```

This is separate from the job-level `ExperimentRun` summary — one gives you the end-to-end job duration, the other gives you per-call observability into which agent is slow or failing.

---

## Testing

45 tests across the agent layer and the API — Data Quality Center scoring, AutoML leaderboard construction, clustering, forecasting, SHAP explanations, industry KPI matching, PDF/dashboard generation, database-backed auth, and per-user history isolation.

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=agents --cov-report=term-missing
```
`.github/workflows/tests.yml` runs this on every push/PR against Python 3.11 and 3.12.

---

## Experiment Tracking

Every completed analysis logs a row to the database — dataset, mode, task type, best model, score, quality score, and duration — visible on the **History** page (or via `GET /api/history`). It's a deliberately small version of what MLflow/Weights & Biases do: the point isn't to replace those tools, it's to not throw away a model's performance the moment the job finishes.

---

## Live Agent Monitoring

The "Agent Pipeline" panel in the UI isn't just a progress bar — `/api/status/<job_id>` returns real per-stage timestamps (`step_history`), so the frontend shows live elapsed time for whichever agent is currently running, and final durations for each completed stage once the job is done.

---

## Deployment

### Docker (any host)
```bash
docker build -t datamind .
docker run -p 5000:5000 -e GROQ_API_KEY=gsk_xxx -e SECRET_KEY=$(python -c "import secrets;print(secrets.token_hex(32))") datamind
```
Or with Compose: `GROQ_API_KEY=gsk_xxx docker compose up --build`

### Render
1. Push this repo to GitHub.
2. Render → **New** → **Blueprint** → select the repo. `render.yaml` configures the service automatically.
3. Set `GROQ_API_KEY` in the Render dashboard (marked `sync: false` in the blueprint so it's never committed).

### Railway
1. Push to GitHub, then **New Project → Deploy from GitHub repo** in Railway.
2. Railway reads `railway.json` automatically. Set `GROQ_API_KEY` and `SECRET_KEY` as variables in the Railway dashboard.

### Notes for any host
- Set `FLASK_ENV=production` so debug mode is off.
- The app uses `gunicorn` as the WSGI server in all deployment paths above — don't use `python app.py` (the Flask dev server) in production.
- User accounts, experiment history, and LLM call logs live in a real database (SQLAlchemy, SQLite by default) — set `DATABASE_URL` to a Postgres connection string for real multi-instance hosting. On Render/Railway's free tiers, an unset `DATABASE_URL` means SQLite writes to local disk, which gets wiped on every redeploy; fine for a demo, but add a managed Postgres add-on if you need accounts/history to persist.

---

## Design System

The UI is a real glassmorphism redesign tied to what the product actually does — frosted, layered glass panels you can see depth through, echoing SHAP explainability (seeing through a model's decision instead of treating it as opaque). Amber/cyan duotone palette — amber for "instrument/measurement," cyan for "data." Space Grotesk for headings, Inter for UI text, JetBrains Mono for every numeric/data value.

The same palette and type system carry through to the standalone HTML dashboard and the PNG charts embedded in the PDF report, so a screenshot of the live app, the dashboard export, or the PDF all read as the same product.

---

## Known limitations / not yet done

- **Prophet** is optional/commented out by default — ARIMA/SARIMA cover forecasting without the heavier build.
- **CatBoost** results will vary by platform; drop it from `requirements.txt` if a host's build minutes make it slow to install.
- **RAG** uses TF-IDF retrieval rather than dense embeddings — a reasonable trade-off for free-tier hosting, but won't catch semantic similarity the way an embeddings-based approach would.
- **MCP/A2A** (agent-to-agent protocol standardization) not implemented — considered out of scope for the current build.

## Python Version
Tested with Python 3.11–3.12.
