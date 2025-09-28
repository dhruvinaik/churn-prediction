# train_auto_taxi_churn.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# ===========================
# Config
# ===========================
DATA_FILE = "auto_taxi_churn.csv"              # dataset generated earlier
MODEL_OUT = "auto_taxi_churn_model.pkl"        # trained model
ENC_OUT = "auto_taxi_encoders.pkl"             # encoders dict

# ===========================
# Training pipeline
# ===========================
def train():
    # 1. Load dataset
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Drop non-feature columns
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    if "churn_prob" in df.columns:
        df = df.drop(columns=["churn_prob"])

    # 3. Target
    y = df["churn"]
    X = df.drop(columns=["churn"])

    # 4. Encode categoricals
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Train model
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nEvaluation metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1:", f1_score(y_test, y_pred, zero_division=0))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # 8. Save model + encoders
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    with open(ENC_OUT, "wb") as f:
        pickle.dump(encoders, f)

    print(f"\n✅ Model saved as {MODEL_OUT}")
    print(f"✅ Encoders saved as {ENC_OUT}")

# ===========================
if __name__ == "__main__":
    train()
