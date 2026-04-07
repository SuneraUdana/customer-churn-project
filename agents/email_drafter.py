from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import pandas as pd
import time
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Switch to smaller model — 8B uses far fewer tokens
def draft_email(customer_id, risk_tier, churn_prob, reasons):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",   # ← was llama-3.3-70b-versatile
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.4,
        max_tokens=150                   # ← cap response length
    )

    prompt = f"""Write a 2-sentence retention email for this customer.
Risk: {risk_tier} | Churn Prob: {churn_prob:.0%} | Risk Factors: {reasons}
Be warm, specific, and end with one clear action."""

    response = llm.invoke([
        SystemMessage(content="You are a customer retention specialist. Be concise."),
        HumanMessage(content=prompt)
    ])
    return response.content.strip()

def run_email_drafter(state):
    df = state["explained_df"].copy()

    at_risk     = df[df["risk_tier"].isin(["🔴 High Risk", "🟡 Medium Risk"])].copy()
    not_at_risk = df[df["risk_tier"] == "🟢 Low Risk"].copy()
    not_at_risk["retention_email"] = "No action needed."

    # ✅ Only draft for top 20 highest-risk customers
    at_risk = at_risk.sort_values("churn_prob", ascending=False)
    top_at_risk    = at_risk.head(20).copy()
    rest_at_risk   = at_risk.iloc[20:].copy()
    rest_at_risk["retention_email"] = "⏳ Email not generated (upgrade plan for full batch)"

    emails = []
    for i, (_, row) in enumerate(top_at_risk.iterrows()):
        try:
            email = draft_email(
                row["customerID"],
                row["risk_tier"],
                row["churn_prob"],
                row["top_reasons"]
            )
            emails.append(email)
            # ✅ Small delay to avoid burst rate limits
            if i < len(top_at_risk) - 1:
                time.sleep(0.5)
        except Exception as e:
            emails.append(f"⚠️ Email generation failed: {str(e)[:60]}")

    top_at_risk["retention_email"] = emails

    final_df = pd.concat([top_at_risk, rest_at_risk, not_at_risk]).sort_values(
        "churn_prob", ascending=False
    ).reset_index(drop=True)

    log = f"✉️ Agent 5 — Drafted {len(emails)} retention emails (top 20 high-risk)"
    return {"emails_df": final_df, "agent_log": [log]}
