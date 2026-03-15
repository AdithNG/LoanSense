"""
LoanSense — Professional loan approval UI.
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
from src.models.predict import load_pipeline, predict, predict_proba, explain_decision, apply_guardrails

MODEL_DIR = _root / "models"

if "score_history" not in st.session_state:
    st.session_state["score_history"] = []

# ---- Page config ----
st.set_page_config(
    page_title="LoanSense — Loan Approval",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS: professional look ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* App background */
    .stApp { 
        background: #f1f5f9 !important; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    [data-testid="stAppViewContainer"] { background: #f1f5f9 !important; }
    .block-container { 
        padding: 2rem 3rem 3rem !important; 
        max-width: 720px !important;
    }
    
    /* Header */
    .loansense-header {
        font-family: 'Inter', sans-serif !important;
        padding: 0 0 1.75rem 0 !important;
        margin-bottom: 1.5rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    .loansense-header h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.02em !important;
        margin: 0 !important;
    }
    .loansense-header p {
        color: #64748b !important;
        font-size: 0.95rem !important;
        margin: 0.35rem 0 0 0 !important;
        font-weight: 400 !important;
    }
    
    /* Section card (markdown wrapper) */
    .section-card {
        background: #ffffff !important;
        border-radius: 12px !important;
        padding: 1.5rem 1.75rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        border: 1px solid #e2e8f0 !important;
    }
    .section-title {
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        margin-bottom: 1rem !important;
    }
    
    /* Forms as cards - Streamlit forms */
    [data-testid="stForm"] {
        background: #ffffff !important;
        border-radius: 12px !important;
        padding: 1.5rem 1.75rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Inputs */
    [data-testid="stTextInput"] input, 
    [data-testid="stNumberInput"] input {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }
    label { font-weight: 500 !important; color: #334155 !important; font-size: 0.9rem !important; }
    
    /* Captions and helper text: darker grey for better contrast */
    [data-testid="stCaptionContainer"] { color: #475569 !important; }
    [data-testid="stCaptionContainer"] p { color: #475569 !important; font-weight: 500 !important; }
    .stCaptionContainer p { color: #475569 !important; font-weight: 500 !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #334155 !important; }
    
    /* Manual email expander: more visible (border + background) */
    .manual-email-section [data-testid="stExpander"] { 
        margin-top: 0.5rem !important; 
        border: 1px solid #cbd5e1 !important; 
        border-radius: 8px !important; 
        background: #f8fafc !important;
        overflow: hidden !important;
    }
    .manual-email-section [data-testid="stExpander"] > div:first-child { 
        background: #f1f5f9 !important; 
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
        color: #0f172a !important;
    }
    
    /* Buttons */
    [data-testid="stFormSubmitButton"] button,
    button[kind="primary"] {
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.25rem !important;
        border: none !important;
        background: #2563eb !important;
        color: white !important;
    }
    [data-testid="stFormSubmitButton"] button:hover,
    button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] button {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    /* Slider - softer look */
    [data-testid="stSlider"] [role="slider"] {
        background: #2563eb !important;
    }
    [data-testid="stSlider"] span {
        color: #334155 !important;
        font-weight: 500 !important;
    }
    
    /* Metrics and result area */
    [data-testid="stMetricValue"] { font-size: 1.25rem !important; font-weight: 600 !important; }
    [data-testid="stProgressBar"] > div { border-radius: 6px !important; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] label { color: #94a3b8 !important; }
    [data-testid="stSidebar"] .stSuccess { background: rgba(34, 197, 94, 0.2) !important; color: #86efac !important; border-radius: 8px !important; }
    [data-testid="stSidebar"] .stWarning { background: rgba(234, 179, 8, 0.2) !important; color: #fde047 !important; border-radius: 8px !important; }
    
    /* Dividers */
    hr { margin: 1.5rem 0 !important; border-color: #e2e8f0 !important; }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        margin-bottom: 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<div class="loansense-header"><h1>LoanSense</h1><p>Loan approval · ML score → customer email with guardrails</p></div>', unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### Setup")
    st.markdown("Train or load the approval model.")
    if (MODEL_DIR / "pipeline.joblib").exists():
        st.success("Model loaded")
    else:
        st.warning("No model yet")
    with st.expander("Train model", expanded=False):
        data_csv = _root / "data" / "loan_data.csv"
        use_real_data = data_csv.exists()
        if use_real_data:
            st.caption("Use **real data** (data/loan_data.csv) or synthetic below.")
        alg = st.selectbox(
            "Algorithm",
            ["gradient_boosting", "random_forest"],
            format_func=lambda x: "Gradient Boosting" if x == "gradient_boosting" else "Random Forest",
        )
        n_samples = st.slider("Sample size", 500, 5000, 2000, 500)
        if st.button("Train (synthetic)", key="sidebar_train_synth"):
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
        if use_real_data and st.button("Train from data/loan_data.csv", key="sidebar_train_csv"):
            try:
                with st.spinner("Training on real data…"):
                    from src.data import load_loan_data, preprocess_features, prepare_splits
                    from src.models.train import train_model, evaluate_model, save_pipeline
                    df = load_loan_data(str(data_csv))
                    df = preprocess_features(df)
                    train_df, val_df, test_df = prepare_splits(df, 0.8, 0.1, 0.1, seed=42)
                    model, X_val, y_val, feature_cols = train_model(train_df, val_df, algorithm=alg, seed=42)
                    X_test, y_test = test_df[feature_cols], test_df["approved"]
                    metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
                    save_pipeline(model, feature_cols, metrics, MODEL_DIR)
                st.success("Done")
                st.caption(f"Val accuracy: {metrics['validation']['accuracy']:.2%} (real data)")
                if hasattr(model, "feature_importances_"):
                    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                    st.caption("**Top features:** " + ", ".join(imp.head(3).index.tolist()))
            except Exception as e:
                st.error(str(e))
        if not use_real_data:
            st.caption("Run `python scripts/download_loan_data.py` to fetch UCI Credit data, then re-open this panel.")

# ---- Main: Score application ----
st.markdown('<div class="section-card"><p class="section-title">Score application</p>', unsafe_allow_html=True)
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
        guardrail_decision, guardrail_reason = apply_guardrails(row)
        if guardrail_decision is not None:
            prob = 0.0
            decision_int = 0
            reason = guardrail_reason or "Application does not meet guidelines."
        else:
            model, feature_cols = load_pipeline(MODEL_DIR)
            prob = float(predict_proba(model, feature_cols, row)[0])
            decision_int = int(predict(model, feature_cols, row)[0])
            reason = explain_decision(row, decision_int)
        decision_label = "Approved" if decision_int == 1 else "Denied"

        st.session_state["last_decision"] = "approve" if decision_int == 1 else "deny"
        st.session_state["last_reason"] = reason
        st.session_state["last_applicant_name"] = applicant_name
        for key in ("email_output", "email_from_agent", "email_meta"):
            st.session_state.pop(key, None)
        st.session_state["score_history"] = (
            [{"applicant": applicant_name, "decision": decision_label, "prob": prob, "reason": reason}] + st.session_state["score_history"]
        )[:20]

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Decision", decision_label)
            st.progress(prob)
            st.caption(f"Approval probability: {prob:.1%}")
        with col_b:
            st.caption("**Reason for email**")
            st.write(reason if len(reason) <= 60 else reason[:57] + "...")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ---- Dashboard (collapsible) ----
if st.session_state.get("score_history"):
    with st.expander("Dashboard — recent scores", expanded=False):
        hist = st.session_state["score_history"]
        df_hist = pd.DataFrame(hist)
        try:
            st.dataframe(df_hist, width="stretch", hide_index=True)
        except TypeError:
            st.dataframe(df_hist, hide_index=True)
        approved = sum(1 for h in hist if h["decision"] == "Approved")
        st.caption(f"Approval rate (this session): {approved}/{len(hist)} = {approved/len(hist):.0%}")

# ---- Generate customer email: single control (dropdown + one button) ----
st.markdown('<div class="section-card"><p class="section-title">Generate customer email</p>', unsafe_allow_html=True)
has_key = bool(os.environ.get("OPENAI_API_KEY"))
has_decision = st.session_state.get("last_decision") is not None

if not has_key:
    st.info("Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` to generate emails.")
else:
    if has_decision:
        # Mode: single (email only or agent) or both for compare view
        email_mode = st.selectbox(
            "Action",
            ["simple", "agent", "compare"],
            format_func=lambda x: {
                "simple": "Generate email only",
                "agent": "Generate email + agent pipeline",
                "compare": "Generate both (compare side-by-side)",
            }[x],
            key="email_mode_select",
        )
        st.caption("**Agent:** LLM scores the email for bias; if denied, appends a next-best-offer. **Compare** generates both versions so you can see the difference.")
        if st.button("Generate", type="primary", key="email_generate_btn"):
            try:
                d = st.session_state["last_decision"]
                name = st.session_state.get("last_applicant_name", "Valued Customer")
                reason = st.session_state.get("last_reason")
                from src.llm.email import generate_customer_email
                from src.agents.pipeline import run_agent_pipeline
                if email_mode == "simple":
                    email = generate_customer_email(d, name, reason=reason)
                    st.session_state["email_output"] = email
                    st.session_state["email_simple"] = email
                    st.session_state["email_agent"] = None
                    st.session_state["email_from_agent"] = False
                    st.session_state["email_meta"] = None
                elif email_mode == "agent":
                    result = run_agent_pipeline(d, name, reason=reason)
                    st.session_state["email_output"] = result.email
                    st.session_state["email_simple"] = None
                    st.session_state["email_agent"] = result.email
                    st.session_state["email_from_agent"] = True
                    st.session_state["email_meta"] = {
                        "bias_score": result.bias_score,
                        "escalated": result.escalated,
                        "next_best_offer": result.next_best_offer,
                    }
                else:
                    # compare: generate both
                    email_simple = generate_customer_email(d, name, reason=reason)
                    result = run_agent_pipeline(d, name, reason=reason)
                    st.session_state["email_simple"] = email_simple
                    st.session_state["email_agent"] = result.email
                    st.session_state["email_output"] = result.email
                    st.session_state["email_from_agent"] = True
                    st.session_state["email_meta"] = {
                        "bias_score": result.bias_score,
                        "escalated": result.escalated,
                        "next_best_offer": result.next_best_offer,
                    }
            except Exception as e:
                st.error(str(e))

        # Compare view: side-by-side when both are available
        if st.session_state.get("email_simple") is not None and st.session_state.get("email_agent") is not None:
            st.subheader("Compare: Email only vs Email + agent")
            col_left, col_right = st.columns(2)
            with col_left:
                st.caption("**Email only** (LLM-generated, no bias check or next-best-offer)")
                st.text_area("Email only", value=st.session_state["email_simple"], height=220, key="email_compare_simple", label_visibility="collapsed")
                st.download_button("Download as .txt", data=st.session_state["email_simple"], file_name="loansense_email_only.txt", mime="text/plain", key="dl_simple")
            with col_right:
                st.caption("**Email + agent** (bias-scored; if denied, next-best-offer appended)")
                st.text_area("Email + agent", value=st.session_state["email_agent"], height=220, key="email_compare_agent", label_visibility="collapsed")
                st.download_button("Download as .txt", data=st.session_state["email_agent"], file_name="loansense_email_with_agent.txt", mime="text/plain", key="dl_agent")
            if st.session_state.get("email_meta"):
                m = st.session_state["email_meta"]
                st.info(f"**Agent pipeline:** Bias score **{m['bias_score']:.2f}** · Escalated: **{m['escalated']}**" + (f" · Next-best-offer appended in right column." if m.get("next_best_offer") else ""))
        elif st.session_state.get("email_output"):
            if st.session_state.get("email_from_agent") and st.session_state.get("email_meta"):
                m = st.session_state["email_meta"]
                st.info(f"**Agent pipeline:** Bias score **{m['bias_score']:.2f}** (0–1, lower is safer). Escalated to human: **{m['escalated']}**. " + (f"Next-best-offer appended below." if m.get("next_best_offer") else "No next-best-offer (applicant approved)."))
                if m.get("next_best_offer"):
                    st.caption(f"Recommendation added to email: *{m['next_best_offer']}*")
            st.text_area("Email", value=st.session_state["email_output"], height=200, key="email_display", label_visibility="collapsed")
            st.download_button("Download email as .txt", data=st.session_state["email_output"], file_name="loansense_email.txt", mime="text/plain", key="dl_single")
    else:
        st.caption("Score an application above first, or use the manual form below.")

    st.markdown("---")
    st.markdown("#### 📧 Generate email manually (no score)")
    st.caption("Use this when you want to generate an email without scoring an application first.")
    st.markdown('<div class="manual-email-section">', unsafe_allow_html=True)
    with st.expander("Open manual options", expanded=True):
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
            st.text_area("Email", value=st.session_state["email_manual_output"], height=180, key="email_manual_display", label_visibility="collapsed")
            st.download_button("Download as .txt", data=st.session_state["email_manual_output"], file_name="loansense_email_manual.txt", mime="text/plain", key="dl_manual")
            if st.session_state.get("email_manual_meta"):
                m = st.session_state["email_manual_meta"]
                st.caption(f"Bias score: {m['bias_score']:.2f} · Escalated: {m['escalated']}")
                if m.get("next_best_offer"):
                    st.caption(f"Next best offer: {m['next_best_offer']}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
