"""Simple deployment API: score, generate email, agent pipeline."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# Load project .env so OPENAI_API_KEY from repo is used (overrides system/shell)
_load_env = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_load_env, override=True)

from src.data.preprocess import preprocess_features
from src.models.predict import load_pipeline, predict, predict_proba, explain_decision, apply_guardrails
from src.models.explain import get_prediction_contributions

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Optional API key auth: set API_KEY or LOANSENSE_API_KEY in .env to require X-API-Key header
REQUIRED_API_KEY = os.environ.get("API_KEY") or os.environ.get("LOANSENSE_API_KEY")


def require_api_key(request: Request) -> None:
    """If REQUIRED_API_KEY is set, require X-API-Key header to match."""
    if not REQUIRED_API_KEY:
        return
    key = request.headers.get("X-API-Key")
    if key != REQUIRED_API_KEY:
        raise HTTPException(401, "Missing or invalid X-API-Key header")


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
    income: float = Field(ge=0, description="Annual income (non-negative)")
    debt: float = Field(ge=0, description="Existing debt (non-negative)")
    employment_years: int = Field(ge=0, le=70, description="Years employed (0-70)")
    credit_score: int = Field(ge=300, le=850, description="Credit score (300-850)")
    loan_amount: float = Field(default=50_000.0, ge=0, description="Requested loan amount")
    savings_balance: float = Field(default=10_000.0, ge=0, description="Savings balance")


class ScoreResponse(BaseModel):
    approval_probability: float
    decision: str  # "approved" | "denied"
    reason: str  # short explanation for LLM/email
    feature_contributions: dict[str, float] | None = None  # SHAP-style per-feature contribution (optional)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, _: None = Depends(require_api_key)):
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
    guardrail_decision, guardrail_reason = apply_guardrails(row)
    if guardrail_decision is not None:
        return ScoreResponse(
            approval_probability=0.0,
            decision="denied",
            reason=guardrail_reason or "application does not meet guidelines",
        )
    model, feature_cols = get_pipeline()
    prob = float(predict_proba(model, feature_cols, row)[0])
    decision_int = int(predict(model, feature_cols, row)[0])
    reason = explain_decision(row, decision_int)
    contributions = get_prediction_contributions(model, feature_cols, row)
    return ScoreResponse(
        approval_probability=prob,
        decision="approved" if decision_int == 1 else "denied",
        reason=reason,
        feature_contributions=contributions if contributions else None,
    )


class ScoreAndEmailRequest(BaseModel):
    applicant_name: str = Field(default="Valued Customer", min_length=1, max_length=200)
    income: float = Field(ge=0, description="Annual income (non-negative)")
    debt: float = Field(ge=0, description="Existing debt (non-negative)")
    employment_years: int = Field(ge=0, le=70, description="Years employed (0-70)")
    credit_score: int = Field(ge=300, le=850, description="Credit score (300-850)")
    loan_amount: float = Field(default=50_000.0, ge=0)
    savings_balance: float = Field(default=10_000.0, ge=0)
    run_agent_pipeline: bool = False


@app.post("/score-and-email")
def score_and_email(req: ScoreAndEmailRequest, _: None = Depends(require_api_key)):
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
    guardrail_decision, guardrail_reason = apply_guardrails(row)
    if guardrail_decision is not None:
        decision = "denied"
        reason = guardrail_reason or "application does not meet guidelines"
        prob = 0.0
        decision_int = 0
    else:
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
    """Model loaded and LLM key configured (optional for Level 1-only)."""
    model_loaded = (MODEL_DIR / "pipeline.joblib").exists()
    llm_configured = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "llm_configured": llm_configured,
    }


# --- Level 2: LLM email ---
class GenerateEmailRequest(BaseModel):
    decision: str = Field(..., pattern="^(approve|approved|deny|denied)$", description="approve or deny")
    applicant_name: str = Field(default="Valued Customer", min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=500)


class GenerateEmailResponse(BaseModel):
    email: str


@app.post("/generate-email", response_model=GenerateEmailResponse)
def generate_email(req: GenerateEmailRequest, _: None = Depends(require_api_key)):
    """Level 2: Generate customer email from decision (LLM). Optionally include reason for the decision."""
    try:
        from src.llm.email import generate_customer_email
        email = generate_customer_email(req.decision, req.applicant_name, reason=req.reason)
        return GenerateEmailResponse(email=email)
    except Exception as e:
        raise HTTPException(500, str(e))


# --- Level 3: Agent pipeline ---
class AgentPipelineRequest(BaseModel):
    decision: str = Field(..., pattern="^(approve|approved|deny|denied)$")
    applicant_name: str = Field(default="Valued Customer", min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=500)
    include_next_best_offer: bool = True


@app.post("/agent-pipeline")
def agent_pipeline(req: AgentPipelineRequest, _: None = Depends(require_api_key)):
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
