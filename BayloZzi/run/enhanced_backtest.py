# run/enhanced_backtest.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from core.enhanced_model import train_enhanced_forex_model, load_enhanced_model
from core.risk_manager import AdvancedRiskManager
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBacktester:
    """
    Comprehensive backtesting system with advanced metrics and visualization.
    Designed to validate forex trading strategies and optimize win rates.
    """
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.risk_manager = AdvancedRiskManager(initial_balance)
        self.results = {}
        
    def load_and_prepare_data(self, data_path="data/eurusd_daily_alpha.csv"):
        """Load and prepare forex data for backtesting."""
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return None
                
            # Convert Date column
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Add Volume if not present
            if 'Volume' not in df.columns:
                df['Volume'] = 1000000  # Default volume
                
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def run_comprehensive_backtest(self, df, train_ratio=0.7, validation_ratio=0.15):
        """
        Run comprehensive backtest with train/validation/test splits.
        
        Args:
            df: DataFrame with OHLCV data
            train_ratio: Portion of data for training
            validation_ratio: Portion of data for validation
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE BACKTEST")
        logger.info("=" * 60)
        
        # Split data
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + validation_ratio))
        
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:].copy()
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Train enhanced model
        logger.info("Training enhanced forex model...")
        predictor, training_results = train_enhanced_forex_model(train_data)
        
        # Validation phase
        logger.info("Running validation...")
        val_results = self.backtest_period(predictor, val_data, "Validation")
        
        # Test phase (out-of-sample)
        logger.info("Running out-of-sample test...")
        test_results = self.backtest_period(predictor, test_data, "Test")
        
        # Compile comprehensive results
        self.results = {
            'training': training_results,
            'validation': val_results,
            'test': test_results,
            'model': predictor,
            'data_splits': {
                'train': len(train_data),
                'validation': len(val_data),
                'test': len(test_data)
            }
        }
        
        # Generate reports
        self.generate_performance_report()
        self.create_visualizations()
        
        return self.results
    
    def backtest_period(self, predictor, df, period_name):
        """Backtest a specific period with the trained model."""
        logger.info(f"Backtesting {period_name} period...")
        
        # Reset risk manager for this period
        period_risk_manager = AdvancedRiskManager(self.initial_balance)
        
        # Generate features and predictions
        df_enhanced = predictor.create_advanced_features(df)
        df_enhanced = predictor.create_enhanced_labels(df_enhanced)
        
        trades = []
        portfolio_values = [self.initial_balance]
        signals_generated = 0
        trades_taken = 0
        
        for i in range(1, len(df_enhanced) - 1):  # Leave buffer for lookahead
            current_row = df_enhanced.iloc[i]
            current_price = current_row['Close']
            
            try:
                # Generate prediction
                pred, confidence = predictor.predict_with_confidence(current_row)
                signals_generated += 1
                
                # Validate trade with risk manager
                validation = period_risk_manager.validate_trade(pred, confidence, current_price)
                
                if not validation['approved']:
                    continue
                    
                trades_taken += 1
                
                # Calculate position details
                volatility = df_enhanced.iloc[max(0, i-10):i]['Close'].pct_change().std()
                stop_loss, take_profit = period_risk_manager.calculate_stop_loss_take_profit(
                    current_price, pred, volatility
                )
                
                position_size = period_risk_manager.calculate_position_size(
                    current_price, stop_loss, confidence
                )
                
                # Open position
                position_id = period_risk_manager.open_position(
                    "EURUSD", pred, current_price, stop_loss, take_profit, 
                    position_size, confidence
                )
                
                # Look ahead to find exit point (simulating hold until SL/TP)
                exit_found = False
                for j in range(i + 1, min(i + 50, len(df_enhanced))):  # Max 50 periods hold
                    future_price = df_enhanced.iloc[j]['Close']
                    
                    # Check exit conditions
                    closed_trades = period_risk_manager.check_position_exits({'EURUSD': future_price})
                    if closed_trades:
                        trades.extend(closed_trades)
                        exit_found = True
                        break
                
                # Force close if no exit found (end of data)
                if not exit_found and position_id in period_risk_manager.open_positions:
                    final_price = df_enhanced.iloc[-1]['Close']
                    trade_record = period_risk_manager.close_position(position_id, final_price, "End of Data")
                    trades.append(trade_record)
                
                # Update portfolio value
                portfolio_values.append(period_risk_manager.current_balance)
                
            except Exception as e:
                logger.warning(f"Error processing row {i}: {e}")
                continue
        
        # Calculate period statistics
        performance = period_risk_manager.get_performance_summary()
        
        # Additional metrics
        if trades:
            wins = sum(1 for trade in trades if trade['pnl'] > 0)
            losses = len(trades) - wins
            win_rate = wins / len(trades) if trades else 0
            
            winning_trades = [trade['pnl'] for trade in trades if trade['pnl'] > 0]
            losing_trades = [trade['pnl'] for trade in trades if trade['pnl'] < 0]
            
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
            profit_factor = (avg_win * wins) / (avg_loss * losses) if losses > 0 else float('inf')
            
            # Sharpe and Sortino ratios
            returns = [trade['pnl_pct'] for trade in trades]
            sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
            
            negative_returns = [r for r in returns if r < 0]
            sortino = np.mean(returns) / np.std(negative_returns) if negative_returns and np.std(negative_returns) > 0 else 0
            
        else:
            win_rate = 0
            avg_win = avg_loss = profit_factor = sharpe = sortino = 0
        
        period_results = {
            'period': period_name,
            'total_trades': len(trades),
            'signals_generated': signals_generated,
            'signal_to_trade_ratio': trades_taken / signals_generated if signals_generated > 0 else 0,
            'win_rate': win_rate,
            'total_return': performance['total_return'],
            'final_balance': performance['current_balance'],
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': performance['max_drawdown'],
            'trades': trades,
            'portfolio_values': portfolio_values
        }
        
        logger.info(f"{period_name} Results:")
        logger.info(f"  Win Rate: {win_rate:.2%}")
        logger.info(f"  Total Return: {performance['total_return']:.2f}%")
        logger.info(f"  Total Trades: {len(trades)}")
        logger.info(f"  Profit Factor: {profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        
        return period_results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        logger.info("Generating performance report...")
        
        report = f"""
# ENHANCED FOREX TRADING SYSTEM - PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## MODEL PERFORMANCE
Training Win Rate: {self.results['training']['win_rate']:.2%}
Validation Win Rate: {self.results['validation']['win_rate']:.2%}
Test Win Rate: {self.results['test']['win_rate']:.2%}

## TRADING PERFORMANCE

### Validation Period
- Total Return: {self.results['validation']['total_return']:.2f}%
- Win Rate: {self.results['validation']['win_rate']:.2%}
- Total Trades: {self.results['validation']['total_trades']}
- Profit Factor: {self.results['validation']['profit_factor']:.2f}
- Sharpe Ratio: {self.results['validation']['sharpe_ratio']:.2f}
- Max Drawdown: {self.results['validation']['max_drawdown']:.2f}%

### Out-of-Sample Test Period
- Total Return: {self.results['test']['total_return']:.2f}%
- Win Rate: {self.results['test']['win_rate']:.2%}
- Total Trades: {self.results['test']['total_trades']}
- Profit Factor: {self.results['test']['profit_factor']:.2f}
- Sharpe Ratio: {self.results['test']['sharpe_ratio']:.2f}
- Max Drawdown: {self.results['test']['max_drawdown']:.2f}%

## RISK METRICS
Signal-to-Trade Ratio: {self.results['test']['signal_to_trade_ratio']:.2%}
Average Win: ${self.results['test']['avg_win']:.2f}
Average Loss: ${self.results['test']['avg_loss']:.2f}
Sortino Ratio: {self.results['test']['sortino_ratio']:.2f}

## RECOMMENDATIONS
"""
        
        # Add recommendations based on performance
        test_win_rate = self.results['test']['win_rate']
        if test_win_rate >= 0.65:
            report += "✅ EXCELLENT: Win rate above 65% - System ready for live trading\n"
        elif test_win_rate >= 0.55:
            report += "⚡ GOOD: Win rate above 55% - Consider optimization before live trading\n"
        else:
            report += "❌ NEEDS WORK: Win rate below 55% - Significant improvements needed\n"
            
        if self.results['test']['sharpe_ratio'] >= 1.5:
            report += "✅ Strong risk-adjusted returns (Sharpe > 1.5)\n"
        elif self.results['test']['sharpe_ratio'] >= 1.0:
            report += "⚡ Decent risk-adjusted returns (Sharpe > 1.0)\n"
        else:
            report += "❌ Poor risk-adjusted returns - Review risk management\n"
        
        # Save report
        with open('logs/performance_report.md', 'w') as f:
            f.write(report)
        
        logger.info("Performance report saved to logs/performance_report.md")
        print(report)
    
    def create_visualizations(self):
        """Create performance visualizations."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Portfolio value over time
            axes[0, 0].plot(self.results['test']['portfolio_values'])
            axes[0, 0].set_title('Portfolio Value Over Time (Test Period)')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True)
            
            # Win rate comparison
            periods = ['Training', 'Validation', 'Test']
            win_rates = [
                self.results['training']['win_rate'],
                self.results['validation']['win_rate'],
                self.results['test']['win_rate']
            ]
            axes[0, 1].bar(periods, win_rates)
            axes[0, 1].set_title('Win Rate by Period')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].set_ylim(0, 1)
            
            # Trade P&L distribution
            test_trades = self.results['test']['trades']
            if test_trades:
                pnl_values = [trade['pnl'] for trade in test_trades]
                axes[1, 0].hist(pnl_values, bins=20, alpha=0.7)
                axes[1, 0].axvline(x=0, color='red', linestyle='--')
                axes[1, 0].set_title('Trade P&L Distribution')
                axes[1, 0].set_xlabel('P&L ($)')
            
            # Returns comparison
            returns = [
                self.results['validation']['total_return'],
                self.results['test']['total_return']
            ]
            period_names = ['Validation', 'Test']
            axes[1, 1].bar(period_names, returns)
            axes[1, 1].set_title('Total Returns by Period')
            axes[1, 1].set_ylabel('Return (%)')
            
            plt.tight_layout()
            plt.savefig('logs/performance_charts.png', dpi=300, bbox_inches='tight')
            logger.info("Performance charts saved to logs/performance_charts.png")
            
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")


def main():
    """Main function to run enhanced backtesting."""
    logger.info("Starting Enhanced Forex Backtesting System")
    
    # Initialize backtester
    backtester = EnhancedBacktester(initial_balance=10000)
    
    # Load data
    df = backtester.load_and_prepare_data("data/eurusd_daily_alpha.csv")
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Run comprehensive backtest
    results = backtester.run_comprehensive_backtest(df)
    
    # Summary
    test_results = results['test']
    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Final Win Rate: {test_results['win_rate']:.2%}")
    logger.info(f"Total Return: {test_results['total_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {test_results['max_drawdown']:.2f}%")
    logger.info("=" * 60)
    
    if test_results['win_rate'] >= 0.60:
        logger.info("🎯 SUCCESS: System achieved target win rate!")
    else:
        logger.info("📈 Continue optimizing to reach 60%+ win rate")


if __name__ == "__main__":
    main()