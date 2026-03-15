# LoanSense — Improvements Backlog

Items from the original suggestions. **Done** = implemented; **Todo** = still to do.

---

## 1. Data & ML

| Item | Status | Notes |
|------|--------|--------|
| **Real dataset** — Add a link or script to a public loan dataset (e.g. UCI Credit, Kaggle) and document training from `data/loan_data.csv` in the UI or CLI. | Done | `scripts/download_loan_data.py` fetches UCI Credit Approval, maps to schema, saves `data/loan_data.csv`. README updated; UI sidebar has "Train from data/loan_data.csv" when file exists. |
| **Model interpretability** — Use SHAP or feature importance to explain why the model approved/denied a specific application (beyond the rule-based `explain_decision`), and show it in the UI or API. | Done | `src/models/explain.py`: SHAP TreeExplainer per-prediction contributions. `/score` returns `feature_contributions`; Streamlit shows "Why this decision?" expander. |
| **Hyperparameter tuning** — Optional grid search or Optuna for the Gradient Boosting (and RF) model and mention it in the README. | Done | `scripts/tune.py`: Optuna over n_estimators, max_depth, learning_rate (GB) or n_estimators, max_depth (RF); saves best pipeline. README updated. |

---

## 2. LLM / agent

| Item | Status | Notes |
|------|--------|--------|
| **Multi-provider** — Support Anthropic Claude (Sonnet) as an option (e.g. env or model selector). | Done | `LLM_PROVIDER=openai|anthropic`, `ANTHROPIC_API_KEY`, `src/llm/client.py`. |
| **Structured logging** — Log anonymized pipeline steps (e.g. "email generated", "bias score", "escalated") to a file or stdout. | Done | `log_llm_event()`, `LOG_LEVEL`, `ENV` in `src/utils/log.py`. |
| **Resilience** — Retries with backoff and optional rate limiting for OpenAI/Anthropic calls. | Done | Retries with backoff in `completion()`; rate limiting not added (optional). |

---

## 3. Production & DevOps

| Item | Status | Notes |
|------|--------|--------|
| **Docker Compose** — One file to run API + Streamlit (and optional DB) together for local demos. | Done | `docker-compose.yml`: api (8000) + app (8501). |
| **Health check** — `/health` that checks "model loaded" and optionally "LLM key configured". | Done | `GET /health` returns `model_loaded`, `llm_configured`. |
| **Config by environment** — e.g. `ENV=development` vs `production` for logging level and base URL. | Done | `ENV`, `LOG_LEVEL` in `.env` and `src/utils/log.py`. |

---

## 4. Testing

| Item | Status | Notes |
|------|--------|--------|
| **Integration test** — One test that runs the full flow (score → generate email → agent pipeline) with mocked LLM and asserts structure. | Done | `tests/test_integration.py`: full flow with mocked `completion`. |
| **API test for /score-and-email** — Request/response validation and status codes. | Done | `test_score_and_email_*`, `test_score_and_email_validation_error` in `tests/test_api.py`. |

---

## 5. Docs & interview prep

| Item | Status | Notes |
|------|--------|--------|
| **Architecture diagram** — Mermaid in the README: data → train → score → LLM → bias agent → next-best-offer. | Done | Added "Architecture" section with Mermaid flowchart in README. |
| **Talking points** — Short "How to present this in an interview" section: problem, design choices, trade-offs, what you'd do next. | Todo | New subsection in README or separate INTERVIEW.md. |

---

## 6. Security & validation

| Item | Status | Notes |
|------|--------|--------|
| **API validation** — Stricter Pydantic rules (e.g. income ≥ 0, credit_score 300–850) and clear error messages. | Done | `Field(ge=0, le=...)` on score/score-and-email; `pattern` for decision; `.env.example` updated. |
| **Optional API auth** — API key or simple OAuth for `/score`, `/generate-email`, `/agent-pipeline` for production-like security. | Done | Set `API_KEY` or `LOANSENSE_API_KEY` in `.env`; endpoints require `X-API-Key` header. |

---

## 7. UX

| Item | Status | Notes |
|------|--------|--------|
| **Compare view** — In Streamlit, show "Email only" vs "Email + agent" side-by-side for the same decision. | Done | Action "Generate both (compare side-by-side)" + two columns with Download per column. |
| **Export/copy** — "Copy email" button or "Download as PDF" for the generated email. | Done | "Download as .txt" on all email outputs (single, compare, manual). |

---

## Summary

- **Done:** 16 (multi-provider, logging, retries, Docker Compose, health check, env config, integration test, /score-and-email API tests, architecture diagram, API validation, optional API auth, compare view, export/copy, real dataset, model interpretability, hyperparameter tuning).
- **Todo:** 2 (talking points).
