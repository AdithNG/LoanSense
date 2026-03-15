"""Simple deployment API: score, generate email, agent pipeline."""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load project .env so OPENAI_API_KEY from repo is used (overrides system/shell)
_load_env = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_load_env, override=True)

from src.data.preprocess import preprocess_features
from src.models.predict import load_pipeline, predict, predict_proba, explain_decision

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

app = FastAPI(title="LoanSense API", description="Loan approval ML + LLM + agents")

# Lazy-load pipeline
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not (MODEL_DIR / "pipeline.joblib").exists():
            raise HTTPException(503, "Model not trained. Run: python scripts/train.py")
        _pipeline = load_pipeline(MODEL_DIR)
    return _pipeline


class ScoreRequest(BaseModel):
    income: float
    debt: float
    employment_years: int
    credit_score: int
    loan_amount: float = 50_000.0
    savings_balance: float = 10_000.0


class ScoreResponse(BaseModel):
    approval_probability: float
    decision: str  # "approved" | "denied"
    reason: str  # short explanation for LLM/email


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Level 1: Score one loan application. Returns decision and reason for downstream email."""
    row = pd.DataFrame([{
        "income": req.income,
        "debt": req.debt,
        "employment_years": req.employment_years,
        "credit_score": req.credit_score,
        "loan_amount": req.loan_amount,
        "savings_balance": req.savings_balance,
        "approved": 0,
    }])
    row = preprocess_features(row)
    model, feature_cols = get_pipeline()
    prob = float(predict_proba(model, feature_cols, row)[0])
    decision_int = int(predict(model, feature_cols, row)[0])
    reason = explain_decision(row, decision_int)
    return ScoreResponse(
        approval_probability=prob,
        decision="approved" if decision_int == 1 else "denied",
        reason=reason,
    )


class ScoreAndEmailRequest(BaseModel):
    applicant_name: str = "Valued Customer"
    income: float
    debt: float
    employment_years: int
    credit_score: int
    loan_amount: float = 50_000.0
    savings_balance: float = 10_000.0
    run_agent_pipeline: bool = False


@app.post("/score-and-email")
def score_and_email(req: ScoreAndEmailRequest):
    """Full flow: score application → get decision + reason → LLM generates email (optionally with agent pipeline)."""
    row = pd.DataFrame([{
        "income": req.income,
        "debt": req.debt,
        "employment_years": req.employment_years,
        "credit_score": req.credit_score,
        "loan_amount": req.loan_amount,
        "savings_balance": req.savings_balance,
        "approved": 0,
    }])
    row = preprocess_features(row)
    model, feature_cols = get_pipeline()
    prob = float(predict_proba(model, feature_cols, row)[0])
    decision_int = int(predict(model, feature_cols, row)[0])
    decision = "approved" if decision_int == 1 else "denied"
    reason = explain_decision(row, decision_int)
    try:
        from src.llm.email import generate_customer_email
        from src.agents.pipeline import run_agent_pipeline
        if req.run_agent_pipeline:
            result = run_agent_pipeline(
                "approve" if decision_int == 1 else "deny",
                req.applicant_name,
                reason=reason,
            )
            return {
                "approval_probability": prob,
                "decision": decision,
                "reason": reason,
                "email": result.email,
                "bias_score": result.bias_score,
                "escalated": result.escalated,
                "next_best_offer": result.next_best_offer,
            }
        email = generate_customer_email(decision, req.applicant_name, reason=reason)
        return {"approval_probability": prob, "decision": decision, "reason": reason, "email": email}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Level 2: LLM email ---
class GenerateEmailRequest(BaseModel):
    decision: str  # approved | denied
    applicant_name: str = "Valued Customer"
    reason: str | None = None


class GenerateEmailResponse(BaseModel):
    email: str


@app.post("/generate-email", response_model=GenerateEmailResponse)
def generate_email(req: GenerateEmailRequest):
    """Level 2: Generate customer email from decision (LLM). Optionally include reason for the decision."""
    try:
        from src.llm.email import generate_customer_email
        email = generate_customer_email(req.decision, req.applicant_name, reason=req.reason)
        return GenerateEmailResponse(email=email)
    except Exception as e:
        raise HTTPException(500, str(e))


# --- Level 3: Agent pipeline ---
class AgentPipelineRequest(BaseModel):
    decision: str
    applicant_name: str = "Valued Customer"
    reason: str | None = None
    include_next_best_offer: bool = True


@app.post("/agent-pipeline")
def agent_pipeline(req: AgentPipelineRequest):
    """Level 3: Full pipeline with bias detection and optional next-best-offer."""
    try:
        from src.agents.pipeline import run_agent_pipeline
        result = run_agent_pipeline(
            req.decision,
            req.applicant_name,
            reason=req.reason,
            include_next_best_offer_on_deny=req.include_next_best_offer,
        )
        return {
            "email": result.email,
            "bias_score": result.bias_score,
            "escalated": result.escalated,
            "passed_tough_check": result.passed_tough_check,
            "next_best_offer": result.next_best_offer,
            "final_email_sent": result.final_email_sent,
        }
    except Exception as e:
        raise HTTPException(500, str(e))
