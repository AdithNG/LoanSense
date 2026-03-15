"""
LoanSense demo UI (Streamlit).
Run from project root: streamlit run app.py
"""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv
load_dotenv(_root / ".env", override=True)

import streamlit as st
import pandas as pd

from src.data.preprocess import preprocess_features
from src.models.predict import load_pipeline, predict, predict_proba, explain_decision

MODEL_DIR = _root / "models"

# Persist last N scores for dashboard (session only)
if "score_history" not in st.session_state:
    st.session_state["score_history"] = []

st.set_page_config(page_title="LoanSense", page_icon="📋", layout="centered")
st.title("📋 LoanSense")
st.caption("Loan approval: score → email with reason")

# ----- Sidebar: Train model -----
with st.sidebar:
    st.header("⚙️ Setup")
    with st.expander("Train model", expanded=False):
        if (MODEL_DIR / "pipeline.joblib").exists():
            st.success("Model loaded")
        else:
            st.warning("No model yet")
        alg = st.selectbox(
            "Algorithm",
            ["gradient_boosting", "random_forest"],
            format_func=lambda x: "Gradient Boosting" if x == "gradient_boosting" else "Random Forest",
        )
        n_samples = st.slider("Sample size", 500, 5000, 2000, 500)
        if st.button("Train"):
            try:
                with st.spinner("Training…"):
                    from src.data import load_sample_data, preprocess_features, prepare_splits
                    from src.models.train import train_model, evaluate_model, save_pipeline
                    df = load_sample_data(n=n_samples, seed=42)
                    df = preprocess_features(df)
                    train_df, val_df, test_df = prepare_splits(df, 0.8, 0.1, 0.1, seed=42)
                    model, X_val, y_val, feature_cols = train_model(train_df, val_df, algorithm=alg, seed=42)
                    X_test, y_test = test_df[feature_cols], test_df["approved"]
                    metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
                    save_pipeline(model, feature_cols, metrics, MODEL_DIR)
                st.success("Done")
                st.caption(f"Val accuracy: {metrics['validation']['accuracy']:.2%}")
                if hasattr(model, "feature_importances_"):
                    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                    st.caption("**Top features:** " + ", ".join(imp.head(3).index.tolist()))
            except Exception as e:
                st.error(str(e))

# ----- Main: Score application -----
st.header("1. Score application")

with st.form("score_form"):
    c1, c2 = st.columns(2)
    with c1:
        applicant_name = st.text_input("Applicant name", value="Jane Doe")
        income = st.number_input("Annual income ($)", min_value=0, value=50_000, step=1000)
        debt = st.number_input("Existing debt ($)", min_value=0, value=10_000, step=500)
        employment_years = st.number_input("Years employed", min_value=0, value=5, step=1)
    with c2:
        credit_score = st.slider("Credit score", 300, 850, 650)
        loan_amount = st.number_input("Loan amount ($)", min_value=0, value=50_000, step=5000)
        savings_balance = st.number_input("Savings balance ($)", min_value=0, value=10_000, step=1000)
    submitted = st.form_submit_button("Score")

if submitted:
    if not (MODEL_DIR / "pipeline.joblib").exists():
        st.error("Train a model first (sidebar → Train model).")
    else:
        row = pd.DataFrame([{
            "income": income, "debt": debt, "employment_years": employment_years,
            "credit_score": credit_score, "loan_amount": loan_amount,
            "savings_balance": savings_balance, "approved": 0,
        }])
        row = preprocess_features(row)
        model, feature_cols = load_pipeline(MODEL_DIR)
        prob = float(predict_proba(model, feature_cols, row)[0])
        decision_int = int(predict(model, feature_cols, row)[0])
        reason = explain_decision(row, decision_int)
        decision_label = "Approved" if decision_int == 1 else "Denied"

        st.session_state["last_decision"] = "approve" if decision_int == 1 else "deny"
        st.session_state["last_reason"] = reason
        st.session_state["last_applicant_name"] = applicant_name
        # New score → clear previous email so we don't show stale content
        for key in ("email_output", "email_from_agent", "email_meta"):
            st.session_state.pop(key, None)
        # Append to dashboard history (keep last 20)
        st.session_state["score_history"] = (
            [{"applicant": applicant_name, "decision": decision_label, "prob": prob, "reason": reason}] + st.session_state["score_history"]
        )[:20]

        # Result card
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Decision", decision_label)
            st.progress(prob)
            st.caption(f"Approval probability: {prob:.1%}")
        with col_b:
            reason_display = reason if len(reason) <= 60 else reason[:57] + "..."
            st.caption("**Reason for email**")
            st.write(reason_display)

# ----- Dashboard: recent scores -----
if st.session_state.get("score_history"):
    with st.expander("📊 Dashboard — recent scores", expanded=False):
        hist = st.session_state["score_history"]
        df_hist = pd.DataFrame(hist)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        approved = sum(1 for h in hist if h["decision"] == "Approved")
        st.caption(f"Approval rate (this session): {approved}/{len(hist)} = {approved/len(hist):.0%}")

# ----- Generate email (only if we have a decision or API key) -----
st.header("2. Generate customer email")

has_key = bool(os.environ.get("OPENAI_API_KEY"))
has_decision = st.session_state.get("last_decision") is not None

if not has_key:
    st.info("Set `OPENAI_API_KEY` in `.env` to generate emails.")
else:
    if has_decision:
        st.caption("Using the decision from the score above.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📧 Generate email", type="primary"):
                try:
                    from src.llm.email import generate_customer_email
                    d = st.session_state["last_decision"]
                    name = st.session_state.get("last_applicant_name", "Valued Customer")
                    reason = st.session_state.get("last_reason")
                    email = generate_customer_email(d, name, reason=reason)
                    st.session_state["email_output"] = email
                    st.session_state["email_from_agent"] = False
                except Exception as e:
                    st.error(str(e))
        with col2:
            if st.button("🛡️ Email + agent pipeline"):
                try:
                    from src.agents.pipeline import run_agent_pipeline
                    d = st.session_state["last_decision"]
                    name = st.session_state.get("last_applicant_name", "Valued Customer")
                    reason = st.session_state.get("last_reason")
                    result = run_agent_pipeline(d, name, reason=reason)
                    st.session_state["email_output"] = result.email
                    st.session_state["email_meta"] = {
                        "bias_score": result.bias_score,
                        "escalated": result.escalated,
                        "next_best_offer": result.next_best_offer,
                    }
                    st.session_state["email_from_agent"] = True
                except Exception as e:
                    st.error(str(e))

        if st.session_state.get("email_output"):
            st.text_area("Email", value=st.session_state["email_output"], height=180, key="email_display")
            if st.session_state.get("email_from_agent") and st.session_state.get("email_meta"):
                m = st.session_state["email_meta"]
                st.caption(f"Bias score: {m['bias_score']:.2f} · Escalated: {m['escalated']}")
                if m.get("next_best_offer"):
                    st.caption(f"Next best offer: {m['next_best_offer']}")
    else:
        st.caption("Score an application above first, or generate manually below.")

    with st.expander("Generate email manually (no score)"):
        with st.form("email_manual"):
            d_manual = st.selectbox("Decision", ["approve", "deny"], format_func=lambda x: "Approve" if x == "approve" else "Deny")
            name_manual = st.text_input("Applicant name", value="Jane Doe")
            reason_manual = st.text_input("Reason (optional)", placeholder="e.g. strong credit profile")
            use_agent = st.checkbox("Run agent pipeline", value=False)
            if st.form_submit_button("Generate"):
                try:
                    reason_val = reason_manual.strip() or None
                    if use_agent:
                        from src.agents.pipeline import run_agent_pipeline
                        r = run_agent_pipeline(d_manual, name_manual, reason=reason_val)
                        st.session_state["email_manual_output"] = r.email
                        st.session_state["email_manual_meta"] = {"bias_score": r.bias_score, "escalated": r.escalated, "next_best_offer": r.next_best_offer}
                    else:
                        from src.llm.email import generate_customer_email
                        st.session_state["email_manual_output"] = generate_customer_email(d_manual, name_manual, reason=reason_val)
                        st.session_state["email_manual_meta"] = None
                except Exception as e:
                    st.error(str(e))
        if st.session_state.get("email_manual_output"):
            st.text_area("Email", value=st.session_state["email_manual_output"], height=180, key="email_manual_display")
            if st.session_state.get("email_manual_meta"):
                m = st.session_state["email_manual_meta"]
                st.caption(f"Bias score: {m['bias_score']:.2f} · Escalated: {m['escalated']}")
                if m.get("next_best_offer"):
                    st.caption(f"Next best offer: {m['next_best_offer']}")
