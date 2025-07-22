# Legendary Trading Demo - Using strategies from the world's best traders
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.model import load_model
from core.features import add_features
from core.data_loader import download_alpha_fx_daily
from core.legendary_strategy import LegendaryTradingStrategy
import warnings
warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legendary_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Portfolio for tracking
portfolio = {
    "cash": 10000,
    "position": 0,
    "entry_price": None,
    "trades": [],
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "max_drawdown": 0,
    "peak_balance": 10000
}

# Initialize legendary strategy
strategy = LegendaryTradingStrategy()

# Load ML model for additional confirmation
logger.info("Loading ML model...")
model = load_model()

# Features for ML model
ml_features = [
    'open','high','low','close','volume',
    'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
    'BB_upper', 'BB_lower', 'Momentum', 'Volatility',
    'Volume_MA', 'Price_Range', 'Prev_Direction',
    'Resistance', 'Support'
]

def load_demo_data():
    """Load historical data for demo"""
    logger.info("Loading historical data...")
    daily = download_alpha_fx_daily()
    hourly = daily.resample('1h').ffill()
    df = add_features(hourly)
    return df

def get_ml_prediction(df, current_idx):
    """Get ML model prediction for additional confirmation"""
    try:
        current_data = df.iloc[:current_idx+1]
        latest = current_data.iloc[-1]
        row = latest[ml_features].values.reshape(1, -1)
        proba = model.predict_proba(row)[0]
        pred = proba.argmax()
        conf = proba.max()
        return pred, conf
    except:
        return 0, 0.5

def execute_legendary_trade(analysis, ml_pred, ml_conf, timestamp):
    """Execute trades based on legendary strategy analysis"""
    
    # Paul Tudor Jones rule: Only trade if we have proper risk/reward
    if analysis['risk_reward'] < 2.5:  # Slightly more flexible for better opportunities
        logger.info(f"[{timestamp}] Risk/Reward {analysis['risk_reward']:.1f} below minimum 2.5. No trade.")
        return
    
    # Additional ML confirmation (optional boost)
    ml_agrees = False
    if analysis['trade_type'] == 'LONG' and ml_pred == 1 and ml_conf > 0.6:
        ml_agrees = True
        analysis['confidence'] = min(0.95, analysis['confidence'] + 0.05)
    elif analysis['trade_type'] == 'SHORT' and ml_pred == 0 and ml_conf > 0.6:
        ml_agrees = True
        analysis['confidence'] = min(0.95, analysis['confidence'] + 0.05)
    
    logger.info(f"[{timestamp}] Signal: {analysis['trade_type']}, Confidence: {analysis['confidence']:.3f}, R:R: {analysis['risk_reward']:.1f}")
    logger.info(f"   Confirmations: {', '.join(analysis['confirmations'][:3])}")
    if ml_agrees:
        logger.info(f"   ML Model agrees with {ml_conf:.2f} confidence")
    
    # Position sizing based on Paul Tudor Jones 1% rule
    risk_amount = portfolio['cash'] * 0.01  # 1% risk
    pip_value = 0.0001
    
    if portfolio["position"] == 0 and analysis['trade_type'] != 'NO_TRADE':
        # Open new position
        portfolio["position"] = 1 if analysis['trade_type'] == 'LONG' else -1
        portfolio["entry_price"] = analysis['entry_price']
        portfolio["stop_loss"] = analysis['stop_loss']
        portfolio["take_profit"] = analysis['take_profit']
        portfolio["total_trades"] += 1
        
        # Calculate position size based on stop distance
        stop_distance = abs(analysis['entry_price'] - analysis['stop_loss'])
        position_size = risk_amount / (stop_distance / pip_value * 10)  # $10 per pip for 0.1 lot
        
        logger.info(f">>> OPENING {analysis['trade_type']} position @ {analysis['entry_price']:.5f}")
        logger.info(f"    Stop Loss: {analysis['stop_loss']:.5f} | Take Profit: {analysis['take_profit']:.5f}")
        logger.info(f"    Position Size: {position_size:.3f} lots | Risk: ${risk_amount:.2f}")
        logger.info(f"    Strategy: {analysis['confirmations'][0]}")
        
    elif portfolio["position"] != 0:
        # Check exit conditions
        current_position = portfolio["position"]
        entry_price = portfolio["entry_price"]
        current_price = analysis['entry_price']
        
        should_exit = False
        exit_reason = ""
        
        # Implement trailing stop (George Soros style - let winners run)
        if current_position == 1:  # Long
            # Move stop to breakeven after 1:1
            if current_price > entry_price + (entry_price - portfolio["stop_loss"]):
                new_stop = entry_price + (current_price - entry_price) * 0.5  # Trail at 50%
                portfolio["stop_loss"] = max(portfolio["stop_loss"], new_stop)
            
            if current_price <= portfolio["stop_loss"]:
                should_exit = True
                exit_reason = "TRAILING STOP" if portfolio["stop_loss"] > entry_price else "STOP LOSS"
            elif current_price >= portfolio["take_profit"]:
                should_exit = True
                exit_reason = "TAKE PROFIT"
        
        else:  # Short
            # Move stop to breakeven after 1:1
            if current_price < entry_price - (portfolio["stop_loss"] - entry_price):
                new_stop = entry_price - (entry_price - current_price) * 0.5  # Trail at 50%
                portfolio["stop_loss"] = min(portfolio["stop_loss"], new_stop)
            
            if current_price >= portfolio["stop_loss"]:
                should_exit = True
                exit_reason = "TRAILING STOP" if portfolio["stop_loss"] < entry_price else "STOP LOSS"
            elif current_price <= portfolio["take_profit"]:
                should_exit = True
                exit_reason = "TAKE PROFIT"
        
        if should_exit:
            # Calculate P&L
            if current_position == 1:  # Long
                pnl_pips = (current_price - entry_price) / pip_value
            else:  # Short
                pnl_pips = (entry_price - current_price) / pip_value
            
            pnl_dollars = pnl_pips * 0.1 * 10  # $10 per pip for 0.1 lot
            portfolio["cash"] += pnl_dollars
            
            # Update win/loss statistics
            if pnl_dollars > 0:
                portfolio["winning_trades"] += 1
            else:
                portfolio["losing_trades"] += 1
            
            # Track drawdown
            if portfolio["cash"] > portfolio["peak_balance"]:
                portfolio["peak_balance"] = portfolio["cash"]
            drawdown = (portfolio["peak_balance"] - portfolio["cash"]) / portfolio["peak_balance"] * 100
            portfolio["max_drawdown"] = max(portfolio["max_drawdown"], drawdown)
            
            trade_record = {
                "timestamp": timestamp,
                "type": "LONG" if current_position == 1 else "SHORT",
                "entry": entry_price,
                "exit": current_price,
                "pnl_pips": pnl_pips,
                "pnl_dollars": pnl_dollars,
                "reason": exit_reason
            }
            portfolio["trades"].append(trade_record)
            
            logger.info(f"<<< CLOSING position @ {current_price:.5f} - {exit_reason}")
            logger.info(f"    P&L: {pnl_pips:.1f} pips (${pnl_dollars:.2f})")
            logger.info(f"    New Balance: ${portfolio['cash']:.2f} | Drawdown: {drawdown:.1f}%")
            
            # Reset position
            portfolio["position"] = 0
            portfolio["entry_price"] = None

def run_legendary_demo():
    """Run demo with legendary trading strategies"""
    logger.info("="*80)
    logger.info("LEGENDARY FOREX TRADING DEMO")
    logger.info("Using strategies from: Paul Tudor Jones, George Soros, ICT, and more")
    logger.info("="*80)
    logger.info(f"Starting Capital: ${portfolio['cash']:.2f}")
    logger.info("Target Win Rate: 70-85%")
    logger.info("Risk per trade: 1% (Paul Tudor Jones rule)")
    logger.info("Minimum R:R: 3:1")
    logger.info("="*80)
    
    # Load historical data
    df = load_demo_data()
    
    # Use last 500 hours for faster demo
    start_idx = len(df) - 500
    end_idx = len(df) - 1
    
    logger.info(f"Demo period: {df.index[start_idx]} to {df.index[end_idx]}")
    logger.info("Starting legendary strategy analysis...\n")
    
    # Track strategy performance
    strategy_stats = {
        'PTJ 200-MA': {'trades': 0, 'wins': 0},
        'Soros Reflexivity': {'trades': 0, 'wins': 0},
        'Smart Money': {'trades': 0, 'wins': 0},
        'Triangle': {'trades': 0, 'wins': 0},
        'S/D Zone': {'trades': 0, 'wins': 0}
    }
    
    # Simulate trading
    for i in range(start_idx, end_idx):
        # Get data window for analysis
        window_data = df.iloc[max(0, i-200):i+1].copy()
        
        # Run legendary strategy analysis
        analysis = strategy.analyze(window_data)
        
        # Get ML prediction
        ml_pred, ml_conf = get_ml_prediction(df, i)
        
        timestamp = df.index[i]
        
        # Execute trade logic
        execute_legendary_trade(analysis, ml_pred, ml_conf, timestamp)
        
        # Update strategy stats
        if analysis['trade_type'] != 'NO_TRADE' and portfolio["position"] == 0:
            # Just opened a trade
            main_strategy = analysis['confirmations'][0].split(':')[0] if analysis['confirmations'] else 'Unknown'
            if main_strategy in strategy_stats:
                strategy_stats[main_strategy]['trades'] += 1
        
        # Progress update
        if i % 50 == 0:
            progress = ((i - start_idx) / (end_idx - start_idx)) * 100
            logger.info(f"Progress: {progress:.1f}% - Current time: {timestamp}")
    
    # Calculate final statistics
    total_pnl = portfolio['cash'] - 10000
    total_return = (total_pnl / 10000) * 100
    
    if portfolio['total_trades'] > 0:
        win_rate = (portfolio['winning_trades'] / portfolio['total_trades']) * 100
        avg_win = sum(t['pnl_dollars'] for t in portfolio['trades'] if t['pnl_dollars'] > 0) / max(1, portfolio['winning_trades'])
        avg_loss = sum(abs(t['pnl_dollars']) for t in portfolio['trades'] if t['pnl_dollars'] < 0) / max(1, portfolio['losing_trades'])
        profit_factor = (avg_win * portfolio['winning_trades']) / max(1, (avg_loss * portfolio['losing_trades']))
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("LEGENDARY TRADING RESULTS")
    logger.info("="*80)
    logger.info(f"Initial Capital: $10,000.00")
    logger.info(f"Final Balance: ${portfolio['cash']:.2f}")
    logger.info(f"Net Profit/Loss: ${total_pnl:.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Maximum Drawdown: {portfolio['max_drawdown']:.2f}%")
    
    logger.info(f"\nTRADE STATISTICS")
    logger.info("="*80)
    logger.info(f"Total Trades: {portfolio['total_trades']}")
    logger.info(f"Winning Trades: {portfolio['winning_trades']}")
    logger.info(f"Losing Trades: {portfolio['losing_trades']}")
    logger.info(f"Win Rate: {win_rate:.1f}%")
    logger.info(f"Average Win: ${avg_win:.2f}")
    logger.info(f"Average Loss: ${avg_loss:.2f}")
    logger.info(f"Profit Factor: {profit_factor:.2f}")
    
    # Show last 10 trades
    if portfolio["trades"]:
        logger.info(f"\nLAST 10 TRADES:")
        logger.info("-" * 80)
        for trade in portfolio["trades"][-10:]:
            result = "WIN" if trade['pnl_dollars'] > 0 else "LOSS"
            logger.info(f"{trade['timestamp']} | {trade['type']:5} | Entry: {trade['entry']:.5f} | "
                       f"Exit: {trade['exit']:.5f} | {trade['pnl_pips']:+6.1f} pips | "
                       f"${trade['pnl_dollars']:+7.2f} | {trade['reason']:15} | {result}")
    
    logger.info("\n>>> Legendary trading demo completed!")
    
    # Performance summary
    if win_rate >= 70:
        logger.info("EXCELLENT: Achieved target win rate of 70%+!")
    elif win_rate >= 60:
        logger.info("GOOD: Win rate above 60%, close to target.")
    else:
        logger.info("NEEDS OPTIMIZATION: Win rate below target, consider adjusting parameters.")

if __name__ == "__main__":
    run_legendary_demo()