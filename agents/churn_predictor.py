import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/churn_model.pkl"

RISK_TIERS = {
    "🔴 High Risk":   (0.70, 1.01),
    "🟡 Medium Risk": (0.40, 0.70),
    "🟢 Low Risk":    (0.00, 0.40),
}

def get_risk_tier(prob):
    for label, (low, high) in RISK_TIERS.items():
        if low <= prob < high:
            return label
    return "🟢 Low Risk"

def run_churn_predictor(state):
    artifact   = joblib.load(MODEL_PATH)
    model      = artifact["model"]
    feat_cols  = artifact["feature_cols"]

    feature_df = state["feature_df"].copy()
    customer_ids = feature_df["customerID"].values if "customerID" in feature_df.columns else None
    X = feature_df[feat_cols]

    probs = model.predict_proba(X)[:, 1]

    pred_df = pd.DataFrame({
        "customerID":   customer_ids if customer_ids is not None else range(len(probs)),
        "churn_prob":   np.round(probs, 4),
        "risk_tier":    [get_risk_tier(p) for p in probs],
    })
    pred_df = pred_df.sort_values("churn_prob", ascending=False).reset_index(drop=True)

    high   = (pred_df["risk_tier"] == "🔴 High Risk").sum()
    medium = (pred_df["risk_tier"] == "🟡 Medium Risk").sum()
    low    = (pred_df["risk_tier"] == "🟢 Low Risk").sum()

    log = f"🤖 Agent 3 — Scored {len(pred_df)} customers | 🔴 {high} | 🟡 {medium} | 🟢 {low}"
    return {"predictions_df": pred_df, "agent_log": [log]}