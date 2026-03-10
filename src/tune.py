import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import itertools
import os

# 1. Initialize your specific DagsHub repo
dagshub.init(repo_owner="RanimmmHisham", repo_name="DVC-DIABETES2-PROJECT", mlflow=True)

def tune():
    # 2. Load Diabetes data (Outcome instead of label)
    train_df = pd.read_csv("data/processed/train.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    X_train, y_train = train_df.drop("Outcome", axis=1), train_df["Outcome"]
    X_test,  y_test  = test_df.drop("Outcome", axis=1),  test_df["Outcome"]

    # 3. Hyperparameter grid
    n_estimators_list = [50, 100, 150]
    max_depth_list    = [5, 10, None]

    # Parent Run
    with mlflow.start_run(run_name="Tuning_Exp"):
        best_accuracy = 0
        
        # itertools.product creates combinations like (50, 5), (50, 10), etc.
        for n_est, depth in itertools.product(n_estimators_list, max_depth_list):
            
            # DESCRIPTIVE RUN NAME: This is what makes the 'Name' column look good
            run_name = f"RandomForest_n{n_est}_d{str(depth)}"
            
            with mlflow.start_run(run_name=run_name, nested=True):
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    random_state=42
                )
                model.fit(X_train, y_train)

                # Predict on test set for true performance
                preds    = model.predict(X_test)
                accuracy = accuracy_score(y_test, preds)
                f1       = f1_score(y_test, preds, average="weighted")

                # LOGGING - Creates the clean columns
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", depth)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)
                
                # Optional: log the model itself to MLflow
                mlflow.sklearn.log_model(model, "model")

                print(f"{run_name} → Acc: {accuracy:.4f} | F1: {f1:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy

        mlflow.log_metric("best_overall_accuracy", best_accuracy)
        print(f"\nTuning Complete. Best Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    tune()