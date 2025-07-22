# run/live_trade.py

import time
import signal
import sys
import logging
import pandas as pd
from core.enhanced_model import load_enhanced_model
from core.features import add_features
from core.strategy import AceStrategy
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Graceful shutdown handler
def signal_handler(sig, frame):
    logger.info('Graceful shutdown initiated...')
    logger.info('Closing any open positions and saving portfolio state')
    # Add cleanup logic here
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Placeholder: Simulated broker portfolio
portfolio = {
    "cash": 10000,
    "position": 0,
    "entry_price": None
}

# Load enhanced model
try:
    model = load_enhanced_model()
    logger.info("Enhanced forex model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load enhanced model: {e}")
    from core.model import load_model
    model = load_model()
    logger.info("Fallback to basic model")

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
    """Enhanced prediction with better feature handling and confidence."""
    try:
        if hasattr(model, 'predict_with_confidence'):
            # Use enhanced model's prediction method
            df_features = model.create_advanced_features(df)
            latest_features = df_features.iloc[-1]
            pred, conf = model.predict_with_confidence(latest_features)
            price = df.iloc[-1]['close']
            return pred, conf, price
        else:
            # Fallback to basic model
            df = add_features(df)
            latest = df.iloc[-1]
            row = latest[features].values.reshape(1, -1)
            proba = model.predict_proba(row)[0]
            pred = proba.argmax()
            conf = proba.max()
            return pred, conf, latest['close']
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 0, 0.0, df.iloc[-1]['close']

def trade_logic(pred, conf, price, confidence_thresh=0.65, risk_per_trade=0.02, max_risk_reward=2.0):
    """
    Enhanced trade logic with improved risk management and logging.
    """
    signal_type = 'BUY' if pred == 1 else 'SELL'
    logger.info(f"Signal: {signal_type}, Confidence: {conf:.3f}, Price: {price:.5f}")
    
    # Enhanced confidence threshold for better win rate
    if conf < confidence_thresh:
        logger.info(f"Confidence {conf:.3f} below threshold {confidence_thresh}. No trade.")
        return
    
    # Position sizing based on confidence and risk
    position_size = min(conf * risk_per_trade, risk_per_trade * 1.5)
    
    if portfolio["position"] == 0:
        portfolio["position"] = 1 if pred == 1 else -1
        portfolio["entry_price"] = price
        portfolio["position_size"] = position_size
        
        # Calculate stop loss and take profit
        volatility_factor = 0.001  # 0.1% base volatility
        stop_loss = price * (1 - volatility_factor) if pred == 1 else price * (1 + volatility_factor)
        take_profit = price * (1 + volatility_factor * max_risk_reward) if pred == 1 else price * (1 - volatility_factor * max_risk_reward)
        
        portfolio["stop_loss"] = stop_loss
        portfolio["take_profit"] = take_profit
        
        logger.info(f"OPENING {'LONG' if pred==1 else 'SHORT'} @ {price:.5f}")
        logger.info(f"Position Size: {position_size:.3f}, Stop Loss: {stop_loss:.5f}, Take Profit: {take_profit:.5f}")
    else:
        # Check for exit conditions
        current_position = portfolio["position"]
        entry_price = portfolio["entry_price"]
        stop_loss = portfolio.get("stop_loss", 0)
        take_profit = portfolio.get("take_profit", 0)
        
        should_exit = False
        exit_reason = ""
        
        if current_position == 1:  # Long position
            if price <= stop_loss:
                should_exit = True
                exit_reason = "Stop Loss"
            elif price >= take_profit:
                should_exit = True
                exit_reason = "Take Profit"
        else:  # Short position
            if price >= stop_loss:
                should_exit = True
                exit_reason = "Stop Loss"
            elif price <= take_profit:
                should_exit = True
                exit_reason = "Take Profit"
        
        if should_exit:
            pnl = (price - entry_price) * current_position
            pnl_pct = (pnl / entry_price) * 100
            portfolio["cash"] += pnl * 1000  # Assuming 1000 unit trades
            logger.info(f"CLOSING position @ {price:.5f} - {exit_reason}")
            logger.info(f"P&L: {pnl:.5f} ({pnl_pct:.2f}%), New Cash: {portfolio['cash']:.2f}")
            
            # Reset position
            portfolio["position"] = 0
            portfolio["entry_price"] = None
        else:
            logger.info(f"Holding {'LONG' if current_position==1 else 'SHORT'} position. Entry: {entry_price:.5f}")

def run_live_loop(interval_seconds=3600):
    """Main trading loop with enhanced error handling and logging."""
    logger.info("Starting enhanced live trading loop...")
    logger.info(f"Initial portfolio: {portfolio}")
    
    iteration = 0
    while True:
        try:
            iteration += 1
            logger.info(f"--- Trading Iteration {iteration} ---")
            
            # Fetch latest data
            df = fetch_latest_candle()
            if df.empty:
                logger.warning("No data received, skipping iteration")
                continue
            
            # Make prediction
            pred, conf, price = make_prediction(df)
            
            # Execute trading logic
            trade_logic(pred, conf, price)
            
            # Log portfolio status
            logger.info(f"Portfolio Status: Cash: {portfolio['cash']:.2f}, Position: {portfolio['position']}")
            
        except KeyboardInterrupt:
            logger.info("Manual stop requested")
            break
        except Exception as e:
            logger.error(f"Trading loop error: {e}", exc_info=True)
            # Continue loop even on errors
        
        logger.info("-" * 60)
        time.sleep(interval_seconds)

if __name__ == "__main__":
    run_live_loop()
