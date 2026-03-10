import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import itertools
import joblib
import os

# 1. Initialize DagsHub
dagshub.init(repo_owner="RanimmmHisham", repo_name="DVC-DIABETES2-PROJECT", mlflow=True)

os.makedirs("models", exist_ok=True)

# 2. Load data
train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")

X_train, y_train = train_df.drop("Outcome", axis=1), train_df["Outcome"]
X_test,  y_test  = test_df.drop("Outcome", axis=1),  test_df["Outcome"]

# 3. Hyperparameter grid
n_estimators_list = [50, 100, 150]
max_depth_list    = [None, 5, 10]

# Parent run to group all trials
with mlflow.start_run(run_name="RandomForest_Tuning_Grid"):

    best_accuracy = 0
    best_model = None
    best_params = {}

    for n_est, depth in itertools.product(n_estimators_list, max_depth_list):
        trial_name = f"RandomForest_n{n_est}_d{depth}"
        
        # Nested=True ensures these trials stay "under" the parent row
        with mlflow.start_run(run_name=trial_name, nested=True):
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                random_state=42
            )
            model.fit(X_train, y_train)

            preds    = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            f1       = f1_score(y_test, preds, average="weighted")

            # Log parameters for THIS specific trial
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("max_depth", str(depth)) # Convert None to string for safety
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            # Log the model to artifacts
            mlflow.sklearn.log_model(model, "model")

            print(f"{trial_name} -> Acc: {accuracy:.4f}")

            # Track the winner
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_params = {"n_estimators": n_est, "max_depth": depth}

    # === LOG SUMMARY TO PARENT RUN (The First Row) ===
    # We use "best_" prefix to avoid naming collisions with the child runs
    mlflow.log_metric("best_accuracy", best_accuracy)  
    mlflow.log_param("best_n_estimators", best_params["n_estimators"])
    mlflow.log_param("best_max_depth", str(best_params["max_depth"]))
    
    # Tagging the parent run as the summary
    mlflow.set_tag("stage", "hyperparameter_tuning")

    # 4. Save the absolute best model physically for DVC tracking
    if best_model:
        joblib.dump(best_model, "models/best_randomf_model.pkl")
        print(f"\nSUCCESS: Best Model saved with Accuracy: {best_accuracy:.4f}")