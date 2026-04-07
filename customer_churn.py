import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib, os

os.makedirs("models", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── Clean ─────────────────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.drop(columns=["customerID"], inplace=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ── Encode categoricals ───────────────────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ── Train / Test split ────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train XGBoost ─────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=50)

# ── Evaluate ──────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("\n", classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── Save model + encoders + feature names ─────────────────────────
joblib.dump({
    "model":        model,
    "encoders":     encoders,
    "feature_cols": X.columns.tolist()
}, "models/churn_model.pkl")

print("\n✅ Model saved to models/churn_model.pkl")
print(f"   Features: {len(X.columns)}")
print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")