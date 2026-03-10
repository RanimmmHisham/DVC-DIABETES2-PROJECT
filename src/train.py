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

    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    model_list = params.get("models", [])
    
    # Load data
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df.drop("Outcome", axis=1)
    y = train_df["Outcome"]

    # This dictionary will store results for ALL models
    all_metrics = {}

    for model_type in model_list:
        print(f"Training {model_type}...")
        
        if model_type == "logistic":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "random_forest":
            model = RandomForestClassifier()
        else:
            continue

        model.fit(X, y)
        
        # Save each model uniquely
        joblib.dump(model, f"models/{model_type}_model.pkl")

        # Calculate accuracy
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        
        # Store in our dictionary
        all_metrics[model_type] = {"accuracy": acc}

    # Save the final combined metrics file
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    print("Metrics saved for all models:", all_metrics)

if __name__ == "__main__":
    train()