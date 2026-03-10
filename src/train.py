import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# 1. Initialize your specific DagsHub repo
dagshub.init(repo_owner="RanimmmHisham", repo_name="DVC-DIABETES2-PROJECT", mlflow=True)

os.makedirs("models", exist_ok=True)

# 2. Load Diabetes data (using 'Outcome' instead of 'label')
train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")

X_train, y_train = train_df.drop("Outcome", axis=1), train_df["Outcome"]
X_test,  y_test  = test_df.drop("Outcome", axis=1),  test_df["Outcome"]

# 3. Train and Log Logistic Regression
with mlflow.start_run(run_name="Logistic_Baseline"):
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train, y_train)

    preds = model_lr.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    # This creates the clean columns in your DagsHub table
    mlflow.log_param("model_type", "Logistic")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model_lr, "logistic_model")
    joblib.dump(model_lr, "models/logistic_model.pkl")
    print(f"Logistic Accuracy: {acc:.4f}")

# 4. Train and Log Random Forest (to show the comparison)
with mlflow.start_run(run_name="RandomForest_Baseline"):
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    preds_rf = model_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, preds_rf)
    f1_rf = f1_score(y_test, preds_rf, average="weighted")

    # Logging the second row for the table
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.log_metric("f1_score", f1_rf)
    mlflow.sklearn.log_model(model_rf, "randomforest_model")
    joblib.dump(model_rf, "models/randomforest_model.pkl")
    print(f"RF Accuracy: {acc_rf:.4f}")