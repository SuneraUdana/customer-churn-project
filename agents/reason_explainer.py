import joblib
import shap
import pandas as pd
import numpy as np

MODEL_PATH = "models/churn_model.pkl"

FEATURE_LABELS = {
    "tenure":           "Account tenure (months)",
    "MonthlyCharges":   "Monthly charges ($)",
    "TotalCharges":     "Total charges ($)",
    "Contract":         "Contract type",
    "InternetService":  "Internet service type",
    "TechSupport":      "Tech support",
    "OnlineSecurity":   "Online security add-on",
    "PaymentMethod":    "Payment method",
    "PaperlessBilling": "Paperless billing",
    "SeniorCitizen":    "Senior citizen status",
}

def run_reason_explainer(state):
    artifact  = joblib.load(MODEL_PATH)
    model     = artifact["model"]
    feat_cols = artifact["feature_cols"]

    feature_df   = state["feature_df"].copy()
    predictions  = state["predictions_df"].copy()

    X = feature_df[feat_cols]

    # SHAP values
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)

    reasons_list = []
    for i in range(len(X)):
        row_shap  = shap_vals[i]
        # Top 3 features pushing toward churn (positive SHAP)
        top_idx   = np.argsort(row_shap)[::-1][:3]
        reasons   = []
        for idx in top_idx:
            fname = feat_cols[idx]
            label = FEATURE_LABELS.get(fname, fname)
            val   = X.iloc[i][fname]
            reasons.append(f"{label} = {val}")
        reasons_list.append(" | ".join(reasons))

    predictions["top_reasons"] = reasons_list

    log = f"🧠 Agent 4 — SHAP reasons generated for {len(predictions)} customers"
    return {"explained_df": predictions, "agent_log": [log]}