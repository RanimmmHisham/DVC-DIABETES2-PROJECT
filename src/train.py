import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# Initialize DagsHub
dagshub.init(repo_owner="RanimmmHisham", repo_name="DVC-DIABETES2-PROJECT", mlflow=True)

os.makedirs("models", exist_ok=True)

# Load data
train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")

X_train, y_train = train_df.drop("Outcome", axis=1), train_df["Outcome"]
X_test,  y_test  = test_df.drop("Outcome", axis=1),  test_df["Outcome"]

def train_and_log(model, model_name, run_name):
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        # Log to MLflow Table
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Save file for DVC
        model_path = f"models/{model_name}_model.pkl"
        joblib.dump(model, model_path)
        
        # Log artifact to MLflow
        mlflow.sklearn.log_model(model, model_name)
        print(f"{run_name} -> Accuracy: {acc:.4f}")

if __name__ == "__main__":
    # Train Logistic
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    train_and_log(lr_model, "logistic", "Logistic_Baseline")

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_log(rf_model, "random_forest", "RF_Baseline")
