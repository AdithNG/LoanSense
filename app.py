"""
LoanSense demo UI (Streamlit).
Run from project root: streamlit run app.py
"""
import sys
from pathlib import Path

# Ensure project root is on path when running streamlit run app.py
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv
load_dotenv(_root / ".env", override=True)

import streamlit as st
import pandas as pd

from src.data.preprocess import preprocess_features
from src.models.predict import load_pipeline, predict, predict_proba

MODEL_DIR = _root / "models"

st.set_page_config(page_title="LoanSense", page_icon="📋", layout="centered")
st.title("📋 LoanSense — Loan Approval Demo")

# Train model (Level 1)
st.header("Train model")
with st.expander("Train or retrain the approval model (sample data)", expanded=False):
    alg = st.selectbox("Algorithm", ["gradient_boosting", "random_forest"], format_func=lambda x: "Gradient Boosting" if x == "gradient_boosting" else "Random Forest")
    n_samples = st.slider("Sample size", 500, 5000, 2000, 500)
    if st.button("Train model"):
        try:
            with st.spinner("Training..."):
                from src.data import load_sample_data, preprocess_features, prepare_splits
                from src.models.train import train_model, evaluate_model, save_pipeline
                df = load_sample_data(n=n_samples, seed=42)
                df = preprocess_features(df)
                train_df, val_df, test_df = prepare_splits(df, 0.8, 0.1, 0.1, seed=42)
                model, X_val, y_val, feature_cols = train_model(train_df, val_df, algorithm=alg, seed=42)
                X_test = test_df[feature_cols]
                y_test = test_df["approved"]
                metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
                save_pipeline(model, feature_cols, metrics, MODEL_DIR)
            st.success("Model saved to `models/`")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Validation accuracy", f"{metrics['validation']['accuracy']:.2%}")
                st.metric("Validation F1", f"{metrics['validation']['f1']:.3f}")
            with col2:
                st.metric("Test accuracy", f"{metrics['test']['accuracy']:.2%}")
                st.metric("Test F1", f"{metrics['test']['f1']:.3f}")
        except Exception as e:
            st.error(str(e))

# Level 1: Score
st.header("Score an application")
with st.form("score_form"):
    income = st.number_input("Annual income ($)", min_value=0, value=50_000, step=1000)
    debt = st.number_input("Existing debt ($)", min_value=0, value=10_000, step=500)
    employment_years = st.number_input("Years employed", min_value=0, value=5, step=1)
    credit_score = st.slider("Credit score", 300, 850, 650)
    loan_amount = st.number_input("Loan amount ($)", min_value=0, value=50_000, step=5000)
    savings_balance = st.number_input("Savings balance ($)", min_value=0, value=10_000, step=1000)
    submitted = st.form_submit_button("Score")

if submitted:
    if not (MODEL_DIR / "pipeline.joblib").exists():
        st.error("Model not trained. Run: `python scripts/train.py`")
    else:
        row = pd.DataFrame([{
            "income": income, "debt": debt, "employment_years": employment_years,
            "credit_score": credit_score, "loan_amount": loan_amount,
            "savings_balance": savings_balance, "approved": 0,
        }])
        row = preprocess_features(row)
        model, feature_cols = load_pipeline(MODEL_DIR)
        prob = float(predict_proba(model, feature_cols, row)[0])
        decision = "Approved" if predict(model, feature_cols, row)[0] == 1 else "Denied"
        st.metric("Decision", decision)
        st.progress(prob)
        st.caption(f"Approval probability: {prob:.1%}")

# Level 2 / 3: Email & agent pipeline (optional, needs OPENAI_API_KEY)
st.divider()
st.header("Generate customer email (Level 2/3)")
import os
if not os.environ.get("OPENAI_API_KEY"):
    st.info("Set OPENAI_API_KEY in .env to use email generation and agent pipeline.")
else:
    with st.form("email_form"):
        decision = st.selectbox("Decision", ["approve", "deny"], format_func=lambda x: "Approve" if x == "approve" else "Deny")
        applicant_name = st.text_input("Applicant name", value="Jane Doe")
        use_agent = st.checkbox("Run full agent pipeline (bias check + next-best-offer for deny)", value=False)
        email_submitted = st.form_submit_button("Generate")

    if email_submitted:
        try:
            if use_agent:
                from src.agents.pipeline import run_agent_pipeline
                result = run_agent_pipeline(decision, applicant_name)
                st.write("**Bias score:**", result.bias_score)
                st.write("**Escalated to human:**", result.escalated)
                if result.next_best_offer:
                    st.write("**Next best offer:**", result.next_best_offer)
                st.text_area("Email", value=result.email, height=200)
            else:
                from src.llm.email import generate_customer_email
                email = generate_customer_email(decision, applicant_name)
                st.text_area("Email", value=email, height=200)
        except Exception as e:
            st.error(str(e))
