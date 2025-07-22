# Demo Trading System - Test before real trading
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.model import load_model
from core.features import add_features
from core.data_loader import download_alpha_fx_daily
import warnings
warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/demo_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Demo portfolio
portfolio = {
    "cash": 10000,
    "position": 0,
    "entry_price": None,
    "trades": [],
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0
}

# Load model
logger.info("Loading trading model...")
model = load_model()

# Features list - must match the model training
features = [
    'open','high','low','close','volume',
    'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
    'BB_upper', 'BB_lower', 'Momentum', 'Volatility',
    'Volume_MA', 'Price_Range', 'Prev_Direction',
    'Resistance', 'Support'
]

def load_demo_data():
    """Load historical data for demo trading"""
    logger.info("Loading historical data for demo...")
    daily = download_alpha_fx_daily()
    hourly = daily.resample('1h').ffill()
    df = add_features(hourly)
    return df

def make_prediction(df, current_idx):
    """Make prediction using historical data"""
    try:
        # Get data up to current point
        current_data = df.iloc[:current_idx+1]
        latest = current_data.iloc[-1]
        
        # Prepare features
        row = latest[features].values.reshape(1, -1)
        
        # Get prediction
        proba = model.predict_proba(row)[0]
        pred = proba.argmax()
        conf = proba.max()
        
        return pred, conf, latest['close']
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 0, 0.0, df.iloc[current_idx]['close']

def execute_trade(pred, conf, price, timestamp):
    """Execute demo trades with realistic conditions"""
    signal_type = 'BUY' if pred == 1 else 'SELL'
    
    # Trading parameters
    confidence_threshold = 0.65
    stop_loss_pips = 30
    take_profit_pips = 60
    pip_value = 0.0001  # For EUR/USD
    position_size = 0.1  # 0.1 lot (10,000 units)
    
    logger.info(f"[{timestamp}] Signal: {signal_type}, Confidence: {conf:.3f}, Price: {price:.5f}")
    
    if conf < confidence_threshold:
        logger.info(f"Confidence below threshold {confidence_threshold}. No trade.")
        return
    
    if portfolio["position"] == 0:
        # Open new position
        portfolio["position"] = 1 if pred == 1 else -1
        portfolio["entry_price"] = price
        
        # Calculate stop loss and take profit
        if pred == 1:  # Long
            portfolio["stop_loss"] = price - (stop_loss_pips * pip_value)
            portfolio["take_profit"] = price + (take_profit_pips * pip_value)
        else:  # Short
            portfolio["stop_loss"] = price + (stop_loss_pips * pip_value)
            portfolio["take_profit"] = price - (take_profit_pips * pip_value)
        
        portfolio["total_trades"] += 1
        
        logger.info(f">>> OPENING {'LONG' if pred==1 else 'SHORT'} position @ {price:.5f}")
        logger.info(f"    Stop Loss: {portfolio['stop_loss']:.5f} | Take Profit: {portfolio['take_profit']:.5f}")
        
    else:
        # Check exit conditions
        current_position = portfolio["position"]
        entry_price = portfolio["entry_price"]
        
        should_exit = False
        exit_reason = ""
        
        if current_position == 1:  # Long
            if price <= portfolio["stop_loss"]:
                should_exit = True
                exit_reason = "STOP LOSS HIT"
            elif price >= portfolio["take_profit"]:
                should_exit = True
                exit_reason = "TAKE PROFIT HIT"
        else:  # Short
            if price >= portfolio["stop_loss"]:
                should_exit = True
                exit_reason = "STOP LOSS HIT"
            elif price <= portfolio["take_profit"]:
                should_exit = True
                exit_reason = "TAKE PROFIT HIT"
        
        if should_exit:
            # Calculate P&L
            if current_position == 1:  # Long
                pnl_pips = (price - entry_price) / pip_value
            else:  # Short
                pnl_pips = (entry_price - price) / pip_value
            
            pnl_dollars = pnl_pips * position_size * 10  # $10 per pip for 0.1 lot
            portfolio["cash"] += pnl_dollars
            
            if pnl_dollars > 0:
                portfolio["winning_trades"] += 1
            else:
                portfolio["losing_trades"] += 1
            
            trade_record = {
                "timestamp": timestamp,
                "type": "LONG" if current_position == 1 else "SHORT",
                "entry": entry_price,
                "exit": price,
                "pnl_pips": pnl_pips,
                "pnl_dollars": pnl_dollars,
                "reason": exit_reason
            }
            portfolio["trades"].append(trade_record)
            
            logger.info(f"<<< CLOSING position @ {price:.5f} - {exit_reason}")
            logger.info(f"    P&L: {pnl_pips:.1f} pips (${pnl_dollars:.2f})")
            logger.info(f"    New Balance: ${portfolio['cash']:.2f}")
            
            # Reset position
            portfolio["position"] = 0
            portfolio["entry_price"] = None

def run_demo_trading():
    """Run demo trading simulation"""
    logger.info("="*60)
    logger.info("FOREX TRADING DEMO - Paper Trading Mode")
    logger.info("="*60)
    logger.info(f"Starting Capital: ${portfolio['cash']:.2f}")
    logger.info("Loading data...")
    
    # Load historical data
    df = load_demo_data()
    
    # Start from 1000 bars ago to have enough history
    start_idx = len(df) - 1000
    end_idx = len(df) - 1
    
    logger.info(f"Demo period: {df.index[start_idx]} to {df.index[end_idx]}")
    logger.info("Starting demo trading...\n")
    
    # Simulate trading
    for i in range(start_idx, end_idx):
        # Make prediction
        pred, conf, price = make_prediction(df, i)
        timestamp = df.index[i]
        
        # Execute trade logic
        execute_trade(pred, conf, price, timestamp)
        
        # Simulate delay (in real trading this would be the wait time)
        if i % 100 == 0:  # Show progress every 100 bars
            progress = ((i - start_idx) / (end_idx - start_idx)) * 100
            logger.info(f"Progress: {progress:.1f}% - Current time: {timestamp}")
        
        # Small delay for readability in output
        time.sleep(0.01)
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("DEMO TRADING RESULTS")
    logger.info("="*60)
    logger.info(f"Initial Capital: $10,000.00")
    logger.info(f"Final Balance: ${portfolio['cash']:.2f}")
    logger.info(f"Net Profit/Loss: ${portfolio['cash'] - 10000:.2f}")
    logger.info(f"Return: {((portfolio['cash'] - 10000) / 10000) * 100:.2f}%")
    logger.info(f"\nTotal Trades: {portfolio['total_trades']}")
    logger.info(f"Winning Trades: {portfolio['winning_trades']}")
    logger.info(f"Losing Trades: {portfolio['losing_trades']}")
    
    if portfolio['total_trades'] > 0:
        win_rate = (portfolio['winning_trades'] / portfolio['total_trades']) * 100
        logger.info(f"Win Rate: {win_rate:.1f}%")
    
    # Show last 5 trades
    if portfolio["trades"]:
        logger.info("\nLast 5 Trades:")
        logger.info("-" * 60)
        for trade in portfolio["trades"][-5:]:
            logger.info(f"{trade['timestamp']} | {trade['type']} | "
                       f"Entry: {trade['entry']:.5f} | Exit: {trade['exit']:.5f} | "
                       f"P&L: {trade['pnl_pips']:.1f} pips (${trade['pnl_dollars']:.2f}) | "
                       f"{trade['reason']}")
    
    logger.info("\n>>> Demo completed! Review the results before live trading.")

if __name__ == "__main__":
    run_demo_trading()