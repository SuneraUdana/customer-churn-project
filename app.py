import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pipeline import run_pipeline
import io

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .risk-high  { color: #ff4b4b; font-weight: 700; }
    .risk-med   { color: #ffa500; font-weight: 700; }
    .risk-low   { color: #00c853; font-weight: 700; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2e3250;
    }
    .agent-step {
        background: #1a1d2e;
        border-left: 3px solid #4f8ff7;
        padding: 8px 16px;
        margin: 4px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    div[data-testid="stExpander"] {
        background: #1a1d2e;
        border: 1px solid #2e3250;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## 🛡️")
with col_title:
    st.markdown("# ChurnGuard AI")
    st.caption("Agentic Customer Churn Prevention · LangGraph + XGBoost + Groq")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Upload Data")
    uploaded = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="Use Telco churn format or compatible CSV"
    )
    st.divider()

    st.markdown("### 🤖 Pipeline Agents")
    agents = [
        ("🧹", "Agent 1", "Data Validator & Cleaner"),
        ("⚙️", "Agent 2", "Feature Engineer"),
        ("🤖", "Agent 3", "Churn Predictor (XGBoost)"),
        ("🧠", "Agent 4", "Reason Explainer (SHAP)"),
        ("✉️", "Agent 5", "Retention Email Drafter"),
    ]
    for icon, label, desc in agents:
        st.markdown(f"{icon} **{label}** — {desc}")

    st.divider()
    st.markdown("### 🎯 Risk Tiers")
    st.markdown("🔴 **High Risk** — churn prob > 70%")
    st.markdown("🟡 **Medium Risk** — 40% to 70%")
    st.markdown("🟢 **Low Risk** — below 40%")
    st.divider()
    st.caption("Model: XGBoost · ROC-AUC: 0.84")
    st.caption("LLM: LLaMA 3.3 70B via Groq")

# ── Main area ─────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("### 👋 Welcome to ChurnGuard AI")
    st.markdown("""
Upload a customer CSV file in the sidebar to get started.
The pipeline will automatically:
1. **Validate & clean** your data
2. **Engineer features** for prediction
3. **Score every customer** with churn probability
4. **Explain the top risk factors** using SHAP
5. **Draft personalized retention emails** for at-risk customers
""")
    st.info("💡 Use the IBM Telco Customer Churn dataset to test — download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)", icon="📊")

    # Sample preview
    st.markdown("#### 📋 Expected CSV Format")
    sample = pd.DataFrame({
        "customerID":     ["7590-VHVEG", "5575-GNVDE"],
        "gender":         ["Female", "Male"],
        "tenure":         [1, 34],
        "Contract":       ["Month-to-month", "One year"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges":   [29.85, 1889.5],
        "Churn":          ["No", "No"]
    })
    st.dataframe(sample, use_container_width=True)

else:
    # ── Load CSV ──────────────────────────────────────────────────
    df_raw = pd.read_csv(uploaded)
    st.success(f"✅ File loaded — {len(df_raw):,} customers · {len(df_raw.columns)} columns")

    run_btn = st.button("🚀 Run ChurnGuard Pipeline", type="primary", use_container_width=True)

    if run_btn:
        with st.status("🤖 Running 5-agent pipeline...", expanded=True) as status:
            st.write("🧹 Agent 1 — Validating & cleaning data...")
            st.write("⚙️  Agent 2 — Engineering features...")
            st.write("🤖 Agent 3 — Scoring churn probability...")
            st.write("🧠 Agent 4 — Generating SHAP explanations...")
            st.write("✉️  Agent 5 — Drafting retention emails...")

            try:
                result = run_pipeline(df_raw)
                st.session_state["result"] = result
                status.update(label="✅ Pipeline complete!", state="complete")

            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    status.update(label="⚠️ Rate limit reached", state="error")
                    st.warning(
                        "⏳ **Groq API daily token limit reached.**\n\n"
                        "Churn scoring and SHAP explanations completed. "
                        "Only email drafting was affected.\n\n"
                        "✅ Try again after **midnight UTC** — the limit resets daily.",
                        icon="⏳"
                    )
                else:
                    status.update(label="❌ Pipeline failed", state="error")
                    st.error(f"Pipeline error: {str(e)}")
                st.stop()

    # ── Show results ──────────────────────────────────────────────
    if "result" in st.session_state:
        result = st.session_state["result"]

        if result.get("error"):
            st.error(f"❌ Pipeline error: {result['error']}")
            st.stop()

        emails_df = result.get("emails_df")
        quality   = result.get("data_quality_report", {})

        if emails_df is None or emails_df.empty:
            st.warning("No results returned. Check your CSV format.")
            st.stop()

        # ── KPI Metrics ───────────────────────────────────────────
        st.markdown("## 📊 Overview")
        high   = (emails_df["risk_tier"] == "🔴 High Risk").sum()
        medium = (emails_df["risk_tier"] == "🟡 Medium Risk").sum()
        low    = (emails_df["risk_tier"] == "🟢 Low Risk").sum()
        total  = len(emails_df)
        avg_prob = emails_df["churn_prob"].mean()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Customers",  f"{total:,}")
        k2.metric("🔴 High Risk",     f"{high:,}",   f"{high/total:.0%} of base")
        k3.metric("🟡 Medium Risk",   f"{medium:,}", f"{medium/total:.0%} of base")
        k4.metric("🟢 Low Risk",      f"{low:,}",    f"{low/total:.0%} of base")
        k5.metric("Avg Churn Prob",   f"{avg_prob:.1%}")

        st.divider()

        # ── Charts row ────────────────────────────────────────────
        st.markdown("## 📈 Risk Distribution")
        ch1, ch2 = st.columns(2)

        with ch1:
            fig_pie = px.pie(
                values=[high, medium, low],
                names=["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
                color_discrete_sequence=["#ff4b4b", "#ffa500", "#00c853"],
                title="Customer Risk Breakdown"
            )
            fig_pie.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font_color="white", title_font_color="white"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            fig_hist = px.histogram(
                emails_df, x="churn_prob", nbins=30,
                title="Churn Probability Distribution",
                color_discrete_sequence=["#4f8ff7"]
            )
            fig_hist.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
                font_color="white", title_font_color="white",
                xaxis_title="Churn Probability",
                yaxis_title="Number of Customers"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # ── Customer Table with Filters ───────────────────────────
        st.markdown("## 🔍 Customer Risk Table")

        f1, f2, f3 = st.columns(3)
        with f1:
            tier_filter = st.multiselect(
                "Filter by Risk Tier",
                ["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
                default=["🔴 High Risk", "🟡 Medium Risk"]
            )
        with f2:
            prob_min = st.slider("Min Churn Probability", 0.0, 1.0, 0.0, 0.05)
        with f3:
            top_n = st.selectbox("Show top N customers", [25, 50, 100, 250, "All"], index=1)

        filtered = emails_df[emails_df["risk_tier"].isin(tier_filter)]
        filtered = filtered[filtered["churn_prob"] >= prob_min]
        if top_n != "All":
            filtered = filtered.head(int(top_n))

        st.dataframe(
            filtered[["customerID", "churn_prob", "risk_tier", "top_reasons"]].style.format(
                {"churn_prob": "{:.1%}"}
            ),
            use_container_width=True,
            height=350
        )

        st.caption(f"Showing {len(filtered):,} customers")
        st.divider()

        # ── Individual Customer Drilldown ─────────────────────────
        st.markdown("## 🔎 Customer Drilldown + Retention Email")

        at_risk_ids = emails_df[
            emails_df["risk_tier"].isin(["🔴 High Risk", "🟡 Medium Risk"])
        ]["customerID"].tolist()

        if at_risk_ids:
            selected_id = st.selectbox("Select a customer", at_risk_ids)
            row = emails_df[emails_df["customerID"] == selected_id].iloc[0]

            d1, d2, d3 = st.columns(3)
            d1.metric("Churn Probability", f"{row['churn_prob']:.1%}")
            d2.metric("Risk Tier", row["risk_tier"])
            d3.metric("Customer ID", row["customerID"])

            st.markdown("#### 🧠 Top Risk Factors (SHAP)")
            reasons = row["top_reasons"].split(" | ")
            for r in reasons:
                st.markdown(f"- {r}")

            st.markdown("#### ✉️ Draft Retention Email")
            st.info(row["retention_email"])

            if st.button("📋 Copy Email to Clipboard"):
                st.code(row["retention_email"])

        st.divider()

        # ── Export ────────────────────────────────────────────────
        st.markdown("## 💾 Export Results")

        e1, e2 = st.columns(2)
        with e1:
            csv_out = emails_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Full Results (CSV)",
                csv_out,
                "churnguard_results.csv",
                "text/csv",
                use_container_width=True
            )
        with e2:
            at_risk_only = emails_df[
                emails_df["risk_tier"].isin(["🔴 High Risk", "🟡 Medium Risk"])
            ]
            csv_risk = at_risk_only.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download At-Risk Only (CSV)",
                csv_risk,
                "churnguard_atrisk.csv",
                "text/csv",
                use_container_width=True
            )

        st.divider()

        # ── Agent Log ─────────────────────────────────────────────
        with st.expander("⚙️ Agent Pipeline Log", expanded=False):
            for log in result.get("agent_log", []):
                st.markdown(f'<div class="agent-step">{log}</div>', unsafe_allow_html=True)

        # ── Data Quality Report ───────────────────────────────────
        with st.expander("📋 Data Quality Report", expanded=False):
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Total Rows",         quality.get("total_rows", "-"))
            q2.metric("Nulls Fixed",         quality.get("missing_fixed", "-"))
            q3.metric("Duplicates Dropped",  quality.get("duplicates_dropped", "-"))
            q4.metric("Columns",             quality.get("columns", "-"))