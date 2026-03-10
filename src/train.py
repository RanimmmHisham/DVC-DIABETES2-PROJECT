import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# 1. Initialize for YOUR repository
dagshub.init(repo_owner="RanimmmHisham", repo_name="DVC-DIABETES2-PROJECT", mlflow=True)

os.makedirs("models", exist_ok=True)

# 2. Load data - using "Outcome" to match your Diabetes dataset
train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")

# Change 'label' to 'Outcome'
X_train, y_train = train_df.drop("Outcome", axis=1), train_df["Outcome"]
X_test,  y_test  = test_df.drop("Outcome", axis=1),  test_df["Outcome"]

# 3. Start MLflow Run
with mlflow.start_run(run_name="Logistic_Baseline"):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    preds    = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1       = f1_score(y_test, preds, average="weighted")

    # Log parameters and metrics for the DagsHub Table
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # Log the model to MLflow (DagsHub Models tab)
    mlflow.sklearn.log_model(model, "logistic_model")

    # Save the physical file for DVC to track
    joblib.dump(model, "models/logistic_model.pkl")

    print(f"Logistic Baseline -> Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
