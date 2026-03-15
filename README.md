# LoanSense

A three-level AI loan approval project: **ML scoring** → **LLM email automation** → **AI agents with guardrails** (bias detection, next-best-offer). Built to showcase production-ready AI skills for recruiters.

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
python scripts/score.py --income 50000 --debt 10000 --employment_years 5
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
# POST /score, POST /generate-email, POST /agent-pipeline
```

## Data

- Place your loan dataset in `data/loan_data.csv` (or use the sample generation script).
- Expected columns (adjust in `src/data/schema.py`): income, debt, employment_years, credit_score, and a target like `approved`.

## License

MIT
