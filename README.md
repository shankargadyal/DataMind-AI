# DataMind V.01 🤖

[![Tests](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/tests.yml)

*(Replace `YOUR-USERNAME/YOUR-REPO` above with your actual GitHub path once you push this — then the badge goes green automatically on every commit.)*

AI-powered multi-agent data analysis platform — Flask + Groq LLaMA 3.3 70B + scikit-learn/XGBoost/LightGBM/CatBoost + SHAP + statsmodels.

Upload a CSV, and the agent pipeline cleans it, profiles it, trains and compares models (or clusters/forecasts it, depending on what's in the data), explains *why* the model predicts what it predicts, scores the data quality, and writes an executive report — no code required.

**Try it without uploading anything**: the upload page has a "Try it instantly" button that runs the full pipeline on a bundled synthetic HR attrition dataset — useful for demos where someone won't upload their own data or paste in a Groq key. See [ARCHITECTURE.md](ARCHITECTURE.md) for how the system fits together and why it's built this way.

## Project Structure

```
DATAMIND_V.01/
├── app.py                       ← Flask backend (orchestrator, routes, auth, job queue)
├── models.py                     ← SQLAlchemy models: User accounts + experiment run history
├── agents/
│   ├── detective.py             ← Agent 1: cleaning, profiling, Data Quality Center (5-dim score)
│   ├── analyst.py                ← Agent 2: Groq LLM insights + target/task detection
│   ├── ml_engineer.py            ← Agent 3: AutoML — classification & regression leaderboard
│   ├── clustering.py             ← Agent 3b: KMeans / DBSCAN / Hierarchical clustering
│   ├── forecasting.py            ← Agent 3c: ARIMA / SARIMA / Prophet time-series forecasting
│   ├── explainability.py         ← Agent 4: SHAP summary + waterfall plots, plain-English captions
│   ├── reporter.py               ← Agent 5: Groq LLM executive report
│   ├── industry_intelligence.py  ← Agent 5b: industry-specific KPI mapping (banking, retail, HR, ...)
│   ├── dashboard_agent.py        ← Agent 6: self-contained Plotly HTML dashboard generator
│   └── report_generator.py       ← PDF report (reportlab) — quality, risk, industry sections
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

XGBoost, LightGBM, CatBoost, statsmodels, and SHAP are all listed as dependencies but the app degrades gracefully if any single one fails to install — it just drops that model/feature from the leaderboard rather than crashing. Prophet is commented out by default since it needs a C++ build toolchain; ARIMA/SARIMA (statsmodels) already cover forecasting without it.

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
| 🔵 Clustering | When there's no clear target (or `mode=clustering` is requested), segments records with KMeans (auto-k via silhouette), DBSCAN, and Hierarchical clustering, with a PCA 2D projection and per-segment feature profiles |
| 📈 Forecasting | For date + numeric-target data (or `mode=forecasting`), backtests ARIMA/SARIMA/Prophet and forecasts forward with confidence intervals |
| 🔍 Explainability | Generates SHAP global summary + local waterfall plots with plain-English captions — "why does the model think X?" |
| 🏭 Industry Intelligence | Optionally maps results onto industry-specific KPIs (Banking, Finance, Retail, Healthcare, Education, Manufacturing, HR) |
| 📝 Reporter | Asks Groq for a plain-English executive summary, risk score, and next steps |
| 📊 Dashboard | Assembles everything into one self-contained Plotly HTML dashboard (gauge, radar, leaderboard, SHAP plots, cluster/forecast charts) |
| 📄 PDF Report | Builds a multi-section PDF: Executive Summary, Data Quality Center, Key Insights, ML Leaderboard, Feature Importance, Risk Analysis, Future Opportunities, Industry Intelligence |

### Requesting a specific mode

`POST /api/analyze` accepts:
- `mode` — `auto` (default), `classification`, `regression`, `clustering`, or `forecasting`
- `industry` — `banking`, `finance`, `retail`, `healthcare`, `education`, `manufacturing`, or `hr` (optional)
- `target_column` — override automatic target detection

---

## Testing

45 tests across the agent layer and the API — Data Quality Center scoring, AutoML leaderboard construction, clustering, forecasting, SHAP explanations, industry KPI matching, PDF/dashboard generation, database-backed auth, and per-user history isolation. Several are direct regression tests for real bugs found while building this (a missing dashboard module, a leaderboard crash on clustering/forecasting results, a metric mislabeling bug, SHAP images that were computed but never embedded in the PDF, a categorical chart that was computed but never rendered).

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=agents --cov-report=term-missing
```

`.github/workflows/tests.yml` runs this on every push/PR against Python 3.11 and 3.12.

---

## Experiment Tracking

Every completed analysis logs a row to the database — dataset, mode, task type, best model, score, quality score, and duration — visible on the **History** page (or via `GET /api/history`). It's a deliberately small version of what MLflow/Weights & Biases do: the point isn't to replace those tools, it's to not throw away a model's performance the moment the job finishes, which is the same habit real MLOps work requires at any scale. History is scoped per-user — see `tests/test_models_and_history.py` for the isolation guarantee.

---

## Live Agent Monitoring

The "Agent Pipeline" panel in the UI isn't just a progress bar — `/api/status/<job_id>` returns real per-stage timestamps (`step_history`), so the frontend shows live elapsed time for whichever agent is currently running, and final durations for each completed stage once the job is done. Useful both for the demo "wow" factor and for actually seeing which agent is the bottleneck on a given dataset (usually the LLM calls in Analyst/Reporter).

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
3. Set `GROQ_API_KEY` in the Render dashboard (it's marked `sync: false` in the blueprint so it's never committed).

### Railway
1. Push to GitHub, then **New Project → Deploy from GitHub repo** in Railway.
2. Railway reads `railway.json` automatically. Set `GROQ_API_KEY` and `SECRET_KEY` as variables in the Railway dashboard.

### Notes for any host
- Set `FLASK_ENV=production` so debug mode is off.
- The app uses `gunicorn` as the WSGI server in all deployment paths above — don't use `python app.py` (the Flask dev server) in production.
- User accounts and the experiment-tracking log live in a real database (SQLAlchemy, SQLite by default) rather than a flat file — set `DATABASE_URL` to a Postgres connection string for real multi-instance hosting. On Render/Railway's free tiers, an unset `DATABASE_URL` means SQLite writes to local disk, which gets wiped on every redeploy; that's fine for a demo, but add a managed Postgres add-on if you need accounts/history to persist.

---

---

## Design System

The UI is a real glassmorphism redesign, not a decorative skin — and it's tied to what the product actually does. The signature visual idea is frosted, layered glass panels you can see depth through, which is the same idea as SHAP explainability (seeing through a model's decision instead of treating it as opaque). The palette is a deliberate amber/cyan duotone — amber reads as "instrument/measurement" (calibration dials, oscilloscope phosphor), cyan as "data" — chosen specifically to avoid the generic violet-on-black look most AI tool clones default to. Type system: Space Grotesk for headings, Inter for UI text, JetBrains Mono for every numeric/data value, so data reads as data at a glance.

The one signature motion element: whichever agent is currently running in the live pipeline gets a slowly rotating conic-gradient ring around its card — light visibly refracting through glass, tied directly to "an agent is actively working." Both theme variants (light/dark) were built and contrast-checked together (WCAG ratios computed numerically, not eyeballed) rather than treating light mode as an afterthought.

The same palette and type system carry through to the standalone HTML dashboard (`dashboard_agent.py`) and the PNG charts embedded in the PDF report, so a screenshot of any of the three (live app, dashboard export, PDF) reads as the same product.

---

## Known limitations / not yet done

- ~~**UI**: still the existing dark-themed UI~~ — redesigned (see Design System below).
- **Prophet** is optional/commented out by default (see above) — ARIMA/SARIMA cover forecasting without the heavier build.
- **CatBoost** model results will vary by platform; some hosts' free build minutes can make the wheel slow to install. Drop it from `requirements.txt` if it's an issue — the leaderboard just trains without it.

## Python Version
Tested with Python 3.11–3.12.
