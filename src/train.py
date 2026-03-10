import pandas as pd
import joblib
import yaml
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

# Load from root
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

if params is None or "models" not in params:
    raise ValueError("params.yaml is empty or missing the 'models' key!")

model_list = params["models"]

train = pd.read_csv("data/processed/train.csv")
# Ensure column name matches your CSV (Diabetes dataset uses 'Outcome')
X = train.drop("Outcome", axis=1)
y = train["Outcome"]

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
    # Note: This overwrites model.pkl each time. 
    # Usually, you'd save a unique name per model_type.
    joblib.dump(model, "models/model.pkl")

print("All training complete.")