import pandas as pd

REQUIRED_COLS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

def run_data_validator(state):
    df = state.get("raw_df")
    report = {}

    if df is None or df.empty:
        return {"error": "No data uploaded.", "agent_log": ["🧹 Agent 1 — No data found"]}

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_before = df.isnull().sum().sum()
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return {
            "error": f"Missing columns: {missing_cols}",
            "agent_log": [f"🧹 Agent 1 — Missing columns: {missing_cols}"]
        }

    # Drop duplicates
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()

    # Strip whitespace from categoricals
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    report = {
        "total_rows":      len(df),
        "missing_fixed":   int(missing_before),
        "duplicates_dropped": int(dupes),
        "columns":         len(df.columns)
    }

    log = (f"🧹 Agent 1 — Validated {len(df)} rows | "
           f"Fixed {missing_before} nulls | Dropped {dupes} dupes")

    return {"cleaned_df": df, "data_quality_report": report,
            "error": None, "agent_log": [log]}