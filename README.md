# LoanSense

A three-level AI loan approval project: **ML scoring** → **LLM email automation** → **AI agents with guardrails** (bias detection, next-best-offer). Built to showcase production-ready AI skills for recruiters.

## Project structure

```
LoanSense/
├── app.py                 # Streamlit UI (train, score, email/agent)
├── scripts/               # CLI: train, score, generate_email, run_agent_pipeline
├── src/
│   ├── data/              # Load CSV/sample, preprocess, feature engineering
│   ├── models/            # Train (GB/RF), evaluate, predict, save pipeline
│   ├── llm/               # LLM customer email from approve/deny
│   ├── agents/            # Bias detection, next-best-offer, pipeline
│   └── api/               # FastAPI: /score, /generate-email, /agent-pipeline
├── tests/                 # Pytest suite (data, models, llm, agents, API)
├── data/                  # Optional: loan_data.csv
└── models/                # Saved pipeline (after train)
```

## Levels

| Level | What it does |
|-------|----------------|
| **1. Beginner** | Gradient Boosting (and Random Forest) model for approve/deny; train/validation/test splits; feature engineering; simple deployment API. |
| **2. Intermediate** | LLM takes the ML decision and generates a personalized email to the customer (adds probabilistic component). |
| **3. Advanced** | Agent detects bias/discrimination in the email → scores it → escalates to human or re-runs through a stricter agent; optional next-best-offer agent for denied applicants. |

## Skills demonstrated

- Data analysis, preprocessing, feature engineering  
- ML model training (Gradient Boosting, Random Forest), validation, testing  
- Deployment (API, scoring in production)  
- LLM integration and prompt design  
- AI agents, guardrails, bias detection  
- Deterministic vs probabilistic system design  

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env     # Add your OPENAI_API_KEY for Level 2/3
```

## Usage

Run all commands from the project root. On Windows PowerShell, set the project root in Python path first:
```powershell
$env:PYTHONPATH = (Get-Location).Path
```

### Level 1: Train and run ML model

```bash
# Train (uses sample data or your CSV; omit --data to generate sample data)
python scripts/train.py --data data/loan_data.csv

# Score one application (deployment simulation)
python scripts/score.py --income 50000 --debt 10000 --employment_years 5 --credit_score 650
```

### Level 2: Generate email with LLM

```bash
python scripts/generate_email.py --decision approve --applicant_name "Jane Doe"
```

### Level 3: Full agent pipeline (bias check + next-best-offer)

```bash
python scripts/run_agent_pipeline.py --decision deny --applicant_name "Jane Doe"
```

### API (optional)

```bash
uvicorn src.api.main:app --reload
```

Interactive API docs: **http://127.0.0.1:8000/docs** — try `POST /score`, `POST /generate-email`, `POST /agent-pipeline`.

### Web UI (Streamlit)

```bash
streamlit run app.py
```

Opens a browser: score an application, and (if `OPENAI_API_KEY` is set) generate customer emails or run the full agent pipeline.

## Testing

Run the test suite (from project root, with `PYTHONPATH` set as above):

```bash
python -m pytest tests/ -v
```

All tests use mocks for the OpenAI API, so no API key is required for tests. Level 2/3 scripts require a valid `OPENAI_API_KEY` in `.env`.

## Data

- Place your loan dataset in `data/loan_data.csv` (or use the sample generation script).
- Expected columns (adjust in `src/data/schema.py`): income, debt, employment_years, credit_score, and a target like `approved`.

## License

MIT
