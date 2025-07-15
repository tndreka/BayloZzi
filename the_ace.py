import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Download Data
def download_data():
    df = yf.download("EURUSD=X", start="2022-01-01", end="2024-12-31", interval="1d")
    df = df.dropna()
    return df

# Step 2: Feature Engineering
def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = np.where(df['Return'].shift(-1) > 0, 1, 0)  # 1 = Up, 0 = Down
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.copy()
    loss = delta.copy()
    
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = loss.abs()
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Step 3: Train Model
def train_model(df):
    features = ['MA5', 'MA10', 'RSI']
    X = df[features]
    y = df['Direction']

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.2f}")

    return model, X_test, y_test, preds

# Step 4: Plot Results
def plot_predictions(y_test, preds):
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(preds, label="Predicted", alpha=0.7)
    plt.title("Prediction vs Actual")
    plt.legend()
    plt.show()

# Run everything
df = download_data()
df = add_features(df)
model, X_test, y_test, preds = train_model(df)
plot_predictions(y_test, preds)
