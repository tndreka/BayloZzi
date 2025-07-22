# core/model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = "models/model.pkl"

def train_enhanced_model(df, label_col='Signal', test_size=0.2, random_state=42):
    """
    Train a classification model using the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): Dataset with features and a 'Signal' label.
        label_col (str): Column name for labels.
    
    Returns:
        model: Trained classifier
        X_test, y_test: Test features and labels
        preds, preds_proba: Model predictions and probabilities
    """
    features = [col for col in df.columns if col != label_col]

    X = df[features]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)

    print("\nModel Performance:\n", classification_report(y_test, preds))
    save_model(model)

    return model, X_test, y_test, preds, preds_proba

def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model
