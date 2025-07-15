#!/usr/bin/env python3
"""
PROJECT ACE - Lightning Fast Version
Optimized for speed and simplicity
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LightningTrader:
    def __init__(self, initial_capital=10000, risk_per_trade=0.02):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.trades = []
        
    def add_trade(self, date, direction, price, pnl):
        """Add a completed trade"""
        self.capital += pnl
        self.trades.append({
            'date': str(date),
            'direction': direction,
            'price': price,
            'pnl': pnl,
            'capital': self.capital
        })
    
    def get_stats(self):
        """Get quick performance stats"""
        if not self.trades:
            return "No trades executed"
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        win_rate = wins / len(self.trades) * 100
        
        return {
            'total_trades': len(self.trades),
            'win_rate': f"{win_rate:.1f}%",
            'total_return': f"{total_return:.2f}%",
            'final_capital': f"${self.capital:.2f}"
        }

def get_data_fast():
    """Quick data download and preparation"""
    print("📊 Getting EUR/USD data...")
    
    # Download data
    df = yf.download("EURUSD=X", start="2023-01-01", end="2024-06-30", 
                     interval="1d", auto_adjust=True, progress=False)
    df = df.dropna()
    
    # Simple features
    df['return'] = df['Close'].pct_change()
    df['target'] = (df['return'].shift(-1) > 0).astype(int)
    
    # Technical indicators
    df['ma_fast'] = df['Close'].rolling(5).mean()
    df['ma_slow'] = df['Close'].rolling(20).mean()
    df['rsi'] = 100 - (100 / (1 + (df['return'].where(df['return'] > 0, 0).rolling(14).mean() / 
                                   (-df['return'].where(df['return'] < 0, 0)).rolling(14).mean())))
    df['volatility'] = df['return'].rolling(10).std()
    
    df = df.dropna()
    print(f"✅ Data ready: {len(df)} days")
    return df

def train_fast_model(df):
    """Quick model training"""
    print("🧠 Training model...")
    
    features = ['ma_fast', 'ma_slow', 'rsi', 'volatility']
    X = df[features].values
    y = df['target'].values
    
    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Model accuracy: {accuracy:.3f}")
    
    return model, X_test, y_test, predictions, probabilities, df.iloc[split:]

def simulate_trading(test_df, predictions, probabilities):
    """Fast trading simulation"""
    print("💰 Running simulation...")
    
    trader = LightningTrader()
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        if i >= len(predictions):
            break
            
        prediction = predictions[i]
        confidence = probabilities[i]
        
        # Only trade with high confidence
        if confidence > 0.6:                # Simulate trade outcome (simplified)
                price = float(row['Close'])
                
                # Simple PnL calculation based on next day's actual move
                if i < len(test_df) - 1:
                    next_price = float(test_df.iloc[i + 1]['Close'])
                    actual_return = (next_price - price) / price
                
                # Calculate PnL based on prediction accuracy
                if int(prediction) == 1:  # Predicted UP
                    pnl = float(actual_return) * 1000  # $1000 position size
                else:  # Predicted DOWN
                    pnl = -float(actual_return) * 1000
                
                trader.add_trade(idx, prediction, price, pnl)
    
    return trader

def main():
    """Main execution - lightning fast"""
    print("⚡ PROJECT ACE - Lightning Version")
    print("=" * 40)
    
    try:
        # Step 1: Get data
        df = get_data_fast()
        
        # Step 2: Train model
        model, X_test, y_test, predictions, probabilities, test_df = train_fast_model(df)
        
        # Step 3: Simulate trading
        trader = simulate_trading(test_df, predictions, probabilities)
        
        # Step 4: Results
        stats = trader.get_stats()
        print("\n🎯 RESULTS:")
        print("=" * 20)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'stats': {k: str(v) for k, v in stats.items()},  # Convert to strings
            'trades': [
                {
                    'date': t['date'],
                    'direction': int(t['direction']),
                    'price': float(t['price']),
                    'pnl': float(t['pnl']),
                    'capital': float(t['capital'])
                } for t in trader.trades[-10:]  # Last 10 trades only
            ]
        }
        
        with open('lightning_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Complete! {len(trader.trades)} trades executed")
        print("📁 Results saved to lightning_results.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready for Phase 3: News Analysis!")
    else:
        print("\n🔧 Need to debug issues...")
