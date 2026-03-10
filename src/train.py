import pandas as pd
import joblib
import yaml
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    os.makedirs("models", exist_ok=True)

    # Load parameters from root
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    if params is None or "models" not in params:
        raise ValueError("params.yaml is empty or missing the 'models' key!")

    model_list = params["models"]

    # Load processed training data
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df.drop("Outcome", axis=1)
    y = train_df["Outcome"]

    # Iterate through models defined in params.yaml
    for model_type in model_list:
        print(f"Training {model_type}...")
        
        if model_type == "logistic":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "random_forest":
            model = RandomForestClassifier()
        else:
            print(f"Unknown model type: {model_type}")
            continue

        model.fit(X, y)
        
        # Save the model file (overwrites for the last model in the list)
        joblib.dump(model, "models/model.pkl")

        # Calculate and save metrics
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        
        metrics = {"accuracy": acc}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
        
        print(f"Saved metrics for {model_type}: Accuracy = {acc}")

    print("All training complete.")

if __name__ == "__main__":
    train()