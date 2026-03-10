import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess():

    # Load the raw dataset
    # Ensure diabetes.csv is in data/raw/
    df = pd.read_csv("data/raw/diabetes.csv")

    # Basic cleaning
    df = df.dropna()

    # Split into train and test
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Save to CSV
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("Preprocessing complete: Data split into train and test.")

if __name__ == "__main__":
    preprocess()