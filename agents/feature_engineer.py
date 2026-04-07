import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

def run_feature_engineer(state):
    df = state["cleaned_df"].copy()

    # Keep customerID aside
    customer_ids = df["customerID"].copy() if "customerID" in df.columns else None

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    feature_df = df[FEATURE_COLS].copy()
    if customer_ids is not None:
        feature_df.insert(0, "customerID", customer_ids.values)

    log = f"⚙️ Agent 2 — Engineered {len(FEATURE_COLS)} features for {len(feature_df)} customers"
    return {"feature_df": feature_df, "feature_cols": FEATURE_COLS, "agent_log": [log]}