# run/live_trade.py

import time
import pandas as pd
from core.model import load_model
from core.features import add_features
from core.strategy import AceStrategy
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Placeholder: Simulated broker portfolio
portfolio = {
    "cash": 10000,
    "position": 0,
    "entry_price": None
}

# Load your model once
model = load_model()

# Define your features list (must match trained model)
features = [
    'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
    'BB_upper', 'BB_lower', 'Momentum', 'Volatility',
    'Volume_MA', 'Price_Range', 'Prev_Direction',
    'Resistance', 'Support'
]

def fetch_latest_candle(symbol='EURUSD=X'):
    """
    Fetch the latest OHLCV candle from Yahoo Finance.
    This simulates a live data feed for testing.
    """
    import yfinance as yf
    df = yf.download(symbol, period="2d", interval="1h")
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                       'Close': 'close', 'Volume': 'volume'}, inplace=True)
    return df

def make_prediction(df):
    df = add_features(df)
    latest = df.iloc[-1]
    row = latest[features].values.reshape(1, -1)
    proba = model.predict_proba(row)[0]
    pred = proba.argmax()
    conf = proba.max()
    return pred, conf, latest['close']

def trade_logic(pred, conf, price, confidence_thresh=0.6, risk_per_trade=0.02):
    """
    Simulated trade logic: logs decisions.
    """
    print(f"{datetime.utcnow()} — Prediction: {'BUY' if pred==1 else 'SELL'}, Conf: {conf:.2f}, Price: {price:.5f}")
    
    if conf < confidence_thresh:
        print("Confidence too low. No trade.")
        return

    if portfolio["position"] == 0:
        portfolio["position"] = 1 if pred == 1 else -1
        portfolio["entry_price"] = price
        print(f"OPENING {'LONG' if pred==1 else 'SHORT'} @ {price:.5f}")
    else:
        print("Already in position. No new trade.")

def run_live_loop(interval_seconds=3600):
    while True:
        try:
            df = fetch_latest_candle()
            pred, conf, price = make_prediction(df)
            trade_logic(pred, conf, price)
        except Exception as e:
            print(f"Error: {e}")

        print("-" * 60)
        time.sleep(interval_seconds)

if __name__ == "__main__":
    run_live_loop()
