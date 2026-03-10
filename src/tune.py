import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import itertools
import joblib
import os

dagshub.init(repo_owner="RanimmmHisham", repo_name="DVC-DIABETES2-PROJECT", mlflow=True)

os.makedirs("models", exist_ok=True)

# 2. Load data - using "Outcome" for the Diabetes dataset
train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")

# Target column is 'Outcome' in the diabetes dataset
X_train, y_train = train_df.drop("Outcome", axis=1), train_df["Outcome"]
X_test,  y_test  = test_df.drop("Outcome", axis=1),  test_df["Outcome"]

# 3. Hyperparameter grid
n_estimators_list = [50, 100, 150]
max_depth_list    = [None, 5, 10]

# Parent run to group all trials
with mlflow.start_run(run_name="RandomForest_Tuning_Grid"):

    best_accuracy = 0
    best_model = None

    for n_est, depth in itertools.product(n_estimators_list, max_depth_list):
        # Create a descriptive name for each trial row in DagsHub
        trial_name = f"RF_n{n_est}_d{depth}"
        
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

            # Logging parameters and metrics for the clean table view
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            # Log the individual model to MLflow artifacts
            mlflow.sklearn.log_model(model, "model")

            print(f"{trial_name} -> Acc: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

    # 4. Save the absolute best model physically for DVC tracking
    if best_model:
        joblib.dump(best_model, "models/best_rf_model.pkl")
        print(f"\nSaved Best Model to models/best_rf_model.pkl with Accuracy: {best_accuracy:.4f}")
