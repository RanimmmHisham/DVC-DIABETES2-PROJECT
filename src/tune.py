import pandas as pd
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize DagsHub connection
dagshub.init(repo_owner='RanimmmHisham', repo_name='DVC-DIABETES2-PROJECT', mlflow=True)

def tune():
    # Load training data
    train_data = pd.read_csv('data/processed/train.csv')
    X_train = train_data.drop("Outcome", axis=1)
    y_train = train_data["Outcome"]
    
    with mlflow.start_run(run_name="Hyperparameter_Tuning"):
        # Hyperparameters to test
        param_grid = [
            {'n_estimators': 10, 'max_depth': 5},
            {'n_estimators': 50, 'max_depth': 10}
        ]
        
        for i, params in enumerate(param_grid):
            # Nested runs for side-by-side comparison on DagsHub
            with mlflow.start_run(run_name=f"Trial_{i}", nested=True):
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                
                acc = accuracy_score(y_train, model.predict(X_train))
                
                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                print(f"Tune: Trial {i} logged with Accuracy {acc}")

if __name__ == "__main__":
    tune()