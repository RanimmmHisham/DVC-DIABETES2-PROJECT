import pandas as pd
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier # Changed to Classifier for Diabetes
from sklearn.metrics import accuracy_score # Better metric for classification

# Connect to DagsHub MLflow
dagshub.init(repo_owner='RanimmmHisham', repo_name='DVC-DIABETES2-PROJECT', mlflow=True)

def tune():
    train_data = pd.read_csv('data/processed/train.csv')
    # Use 'Outcome' instead of 'quality'
    X_train = train_data.drop("Outcome", axis=1)
    y_train = train_data["Outcome"]
    
    with mlflow.start_run(run_name="Hyperparameter_Tuning"):
        param_grid = [
            {'n_estimators': 10, 'max_depth': 5},
            {'n_estimators': 50, 'max_depth': 10}
        ]
        
        for i, params in enumerate(param_grid):
            with mlflow.start_run(run_name=f"Trial_{i}", nested=True):
                # Using Classifier because diabetes is a classification task
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                
                acc = accuracy_score(y_train, model.predict(X_train))
                
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                print(f"Tune: Trial {i} logged with Accuracy {acc}")

if __name__ == "__main__":
    tune()