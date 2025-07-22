# run/advanced_backtest.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from core.enhanced_model_v2 import train_enhanced_forex_model_v2, EnhancedForexPredictorV2
from core.advanced_strategy import AdvancedForexStrategy
from core.risk_manager import AdvancedRiskManager
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedBacktester:
    """
    Advanced backtesting system incorporating:
    1. Smart Money Concepts (ICT)
    2. Harmonic Patterns
    3. Supply/Demand Analysis
    4. Enhanced Risk Management
    5. Multiple confirmation signals
    """
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.risk_manager = AdvancedRiskManager(initial_balance)
        self.strategy = AdvancedForexStrategy()
        self.results = {}
        
    def load_and_prepare_data(self, data_path="data/eurusd_daily_alpha.csv"):
        """Load and prepare forex data for backtesting."""
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Handle column names (case insensitive)
            df.columns = df.columns.str.lower()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return None
            
            # Handle date column
            date_cols = [col for col in df.columns if 'date' in col.lower() or col == 'unnamed: 0']
            if date_cols:
                df['Date'] = pd.to_datetime(df[date_cols[0]])
                df.set_index('Date', inplace=True)
            else:
                # Create synthetic dates if no date column
                df['Date'] = pd.date_range(start='2010-01-01', periods=len(df), freq='D')
                df.set_index('Date', inplace=True)
            
            # Add volume if not present
            if 'volume' not in df.columns:
                df['volume'] = 1000000  # Default volume
                
            # Remove any unnamed columns
            df = df[[col for col in df.columns if not col.startswith('unnamed')]]
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def run_advanced_backtest(self, df, train_ratio=0.7, validation_ratio=0.15,
                            min_confidence=0.65, use_multiple_confirmations=True):
        """
        Run advanced backtest with Smart Money concepts and multiple confirmations.
        """
        logger.info("=" * 60)
        logger.info("STARTING ADVANCED BACKTEST WITH SMART MONEY CONCEPTS")
        logger.info("=" * 60)
        
        # Split data
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + validation_ratio))
        
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:].copy()
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Train enhanced model V2
        logger.info("Training enhanced forex model V2 with Smart Money concepts...")
        predictor, training_results = train_enhanced_forex_model_v2(train_data)
        
        # Validation phase
        logger.info("Running validation with advanced strategies...")
        val_results = self.backtest_period_advanced(
            predictor, val_data, "Validation", min_confidence, use_multiple_confirmations
        )
        
        # Test phase (out-of-sample)
        logger.info("Running out-of-sample test with advanced strategies...")
        test_results = self.backtest_period_advanced(
            predictor, test_data, "Test", min_confidence, use_multiple_confirmations
        )
        
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
        
        # Generate advanced reports
        self.generate_advanced_performance_report()
        self.create_advanced_visualizations()
        self.analyze_trading_patterns()
        
        return self.results
    
    def backtest_period_advanced(self, predictor, df, period_name, 
                               min_confidence=0.65, use_multiple_confirmations=True):
        """
        Backtest with advanced strategy confirmations.
        """
        logger.info(f"Backtesting {period_name} period with advanced strategies...")
        
        # Reset risk manager for this period
        period_risk_manager = AdvancedRiskManager(self.initial_balance)
        
        # Generate features and predictions
        df_enhanced = predictor.create_advanced_features(df)
        df_enhanced = predictor.create_smart_money_features(df_enhanced)
        df_enhanced = predictor.create_enhanced_labels(df_enhanced)
        
        trades = []
        portfolio_values = [self.initial_balance]
        signals_generated = 0
        trades_taken = 0
        strategy_confirmations = []
        
        for i in range(50, len(df_enhanced) - 10):  # Need history for indicators
            current_row = df_enhanced.iloc[i]
            current_price = current_row['close']
            
            try:
                # Generate ML prediction
                pred, ml_confidence = predictor.predict_with_confidence(current_row)
                
                # Get advanced strategy signals
                strategy_signals = self.strategy.generate_advanced_signals(
                    df_enhanced.iloc[:i+1], i
                )
                
                # Combine ML and strategy signals
                if use_multiple_confirmations:
                    # Require both ML and strategy agreement
                    if strategy_signals['signal'] != 0 and pred == 1:
                        final_signal = strategy_signals['signal']
                        # Boost confidence if both agree
                        final_confidence = min(0.95, 
                            ml_confidence * 0.6 + strategy_signals['confidence'] * 0.4
                        )
                        confirmation_type = "ML + Strategy"
                    else:
                        # Skip if no agreement
                        continue
                else:
                    # Use ML prediction with strategy as filter
                    if ml_confidence >= min_confidence and strategy_signals['signal'] != -pred:
                        final_signal = 1 if pred == 1 else -1
                        final_confidence = ml_confidence
                        confirmation_type = "ML Primary"
                    else:
                        continue
                
                signals_generated += 1
                
                # Validate trade with risk manager
                validation = period_risk_manager.validate_trade(
                    final_signal, final_confidence, current_price
                )
                
                if not validation['approved']:
                    continue
                
                trades_taken += 1
                
                # Calculate dynamic stop loss and take profit
                stop_loss, take_profit = self.strategy.calculate_dynamic_sl_tp(
                    df_enhanced.iloc[:i+1], final_signal, current_price, i
                )
                
                # Calculate position size
                position_size = period_risk_manager.calculate_position_size(
                    current_price, stop_loss, final_confidence
                )
                
                # Open position
                position_id = period_risk_manager.open_position(
                    "EURUSD", final_signal, current_price, stop_loss, take_profit,
                    position_size, final_confidence
                )
                
                # Record confirmation details
                strategy_confirmations.append({
                    'position_id': position_id,
                    'confirmation_type': confirmation_type,
                    'ml_confidence': ml_confidence,
                    'strategy_confidence': strategy_signals['confidence'],
                    'final_confidence': final_confidence,
                    'reasons': strategy_signals['reasons']
                })
                
                # Simulate trade execution
                exit_found = False
                for j in range(i + 1, min(i + 100, len(df_enhanced))):  # Max 100 periods hold
                    future_price = df_enhanced.iloc[j]['close']
                    
                    # Check exit conditions
                    closed_trades = period_risk_manager.check_position_exits({'EURUSD': future_price})
                    if closed_trades:
                        trades.extend(closed_trades)
                        exit_found = True
                        break
                    
                    # Check for trailing stop or partial profit taking
                    if j - i > 10 and position_id in period_risk_manager.open_positions:
                        position = period_risk_manager.open_positions[position_id]
                        current_pnl = (future_price - position['entry_price']) / position['entry_price']
                        
                        # Trailing stop if in profit
                        if final_signal == 1 and current_pnl > 0.005:  # 0.5% profit
                            new_stop = future_price * 0.995  # Trail to 0.5% below
                            if new_stop > position['stop_loss']:
                                position['stop_loss'] = new_stop
                        elif final_signal == -1 and current_pnl > 0.005:
                            new_stop = future_price * 1.005
                            if new_stop < position['stop_loss']:
                                position['stop_loss'] = new_stop
                
                # Force close if no exit found
                if not exit_found and position_id in period_risk_manager.open_positions:
                    final_price = df_enhanced.iloc[-1]['close']
                    trade_record = period_risk_manager.close_position(
                        position_id, final_price, "End of Data"
                    )
                    trades.append(trade_record)
                
                # Update portfolio value
                portfolio_values.append(period_risk_manager.current_balance)
                
            except Exception as e:
                logger.warning(f"Error processing row {i}: {e}")
                continue
        
        # Calculate advanced statistics
        performance = period_risk_manager.get_performance_summary()
        
        # Additional metrics
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] < 0]
            
            win_rate = len(wins) / len(trades) if trades else 0
            
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0
            
            # Risk-reward ratio
            avg_win_pct = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
            avg_loss_pct = abs(np.mean([t['pnl_pct'] for t in losses])) if losses else 0
            risk_reward = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 0
            
            profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if losses else float('inf')
            
            # Advanced metrics
            consecutive_wins = self.calculate_consecutive_wins(trades)
            consecutive_losses = self.calculate_consecutive_losses(trades)
            
            # Sharpe and Sortino ratios
            returns = [t['pnl_pct'] for t in trades]
            sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
            
            negative_returns = [r for r in returns if r < 0]
            sortino = np.mean(returns) / np.std(negative_returns) if negative_returns and np.std(negative_returns) > 0 else 0
            
            # Calmar ratio (return / max drawdown)
            max_dd = performance['max_drawdown']
            calmar = performance['total_return'] / max_dd if max_dd > 0 else 0
            
        else:
            win_rate = avg_win = avg_loss = profit_factor = sharpe = sortino = calmar = 0
            risk_reward = consecutive_wins = consecutive_losses = 0
        
        # Strategy analysis
        confirmation_analysis = self.analyze_confirmations(strategy_confirmations, trades)
        
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
            'risk_reward_ratio': risk_reward,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': performance['max_drawdown'],
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'trades': trades,
            'portfolio_values': portfolio_values,
            'confirmation_analysis': confirmation_analysis
        }
        
        logger.info(f"{period_name} Results:")
        logger.info(f"  Win Rate: {win_rate:.2%}")
        logger.info(f"  Total Return: {performance['total_return']:.2f}%")
        logger.info(f"  Risk-Reward Ratio: {risk_reward:.2f}")
        logger.info(f"  Profit Factor: {profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Calmar Ratio: {calmar:.2f}")
        
        return period_results
    
    def calculate_consecutive_wins(self, trades):
        """Calculate maximum consecutive winning trades."""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade['pnl'] > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def calculate_consecutive_losses(self, trades):
        """Calculate maximum consecutive losing trades."""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def analyze_confirmations(self, confirmations, trades):
        """Analyze which confirmation types work best."""
        if not confirmations or not trades:
            return {}
        
        # Map trades to confirmations
        confirmation_performance = {}
        
        for conf in confirmations:
            position_id = conf['position_id']
            # Find corresponding trade
            trade = next((t for t in trades if t['position_id'] == position_id), None)
            
            if trade:
                conf_type = conf['confirmation_type']
                if conf_type not in confirmation_performance:
                    confirmation_performance[conf_type] = {
                        'trades': 0,
                        'wins': 0,
                        'total_pnl': 0,
                        'avg_confidence': 0
                    }
                
                confirmation_performance[conf_type]['trades'] += 1
                if trade['pnl'] > 0:
                    confirmation_performance[conf_type]['wins'] += 1
                confirmation_performance[conf_type]['total_pnl'] += trade['pnl']
                confirmation_performance[conf_type]['avg_confidence'] += conf['final_confidence']
        
        # Calculate averages
        for conf_type in confirmation_performance:
            perf = confirmation_performance[conf_type]
            perf['win_rate'] = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
            perf['avg_pnl'] = perf['total_pnl'] / perf['trades'] if perf['trades'] > 0 else 0
            perf['avg_confidence'] = perf['avg_confidence'] / perf['trades'] if perf['trades'] > 0 else 0
        
        return confirmation_performance
    
    def analyze_trading_patterns(self):
        """Analyze successful trading patterns from results."""
        test_trades = self.results['test']['trades']
        if not test_trades:
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("TRADING PATTERN ANALYSIS")
        logger.info("=" * 60)
        
        # Analyze by time held
        winning_trades = [t for t in test_trades if t['pnl'] > 0]
        losing_trades = [t for t in test_trades if t['pnl'] < 0]
        
        if winning_trades:
            avg_win_duration = np.mean([
                (t['exit_time'] - t['entry_time']).total_seconds() / 3600 
                for t in winning_trades
            ])
            logger.info(f"Average winning trade duration: {avg_win_duration:.1f} hours")
        
        if losing_trades:
            avg_loss_duration = np.mean([
                (t['exit_time'] - t['entry_time']).total_seconds() / 3600 
                for t in losing_trades
            ])
            logger.info(f"Average losing trade duration: {avg_loss_duration:.1f} hours")
        
        # Analyze by exit reason
        exit_reasons = {}
        for trade in test_trades:
            reason = trade['reason']
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            
            exit_reasons[reason]['count'] += 1
            if trade['pnl'] > 0:
                exit_reasons[reason]['wins'] += 1
            exit_reasons[reason]['total_pnl'] += trade['pnl']
        
        logger.info("\nExit Reason Analysis:")
        for reason, stats in exit_reasons.items():
            win_rate = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
            avg_pnl = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0
            logger.info(f"  {reason}: {stats['count']} trades, "
                       f"{win_rate:.1%} win rate, ${avg_pnl:.2f} avg P&L")
        
        # Confirmation analysis
        conf_analysis = self.results['test']['confirmation_analysis']
        if conf_analysis:
            logger.info("\nConfirmation Type Performance:")
            for conf_type, stats in conf_analysis.items():
                logger.info(f"  {conf_type}:")
                logger.info(f"    Trades: {stats['trades']}")
                logger.info(f"    Win Rate: {stats['win_rate']:.1%}")
                logger.info(f"    Avg P&L: ${stats['avg_pnl']:.2f}")
                logger.info(f"    Avg Confidence: {stats['avg_confidence']:.3f}")
    
    def generate_advanced_performance_report(self):
        """Generate comprehensive performance report with advanced metrics."""
        logger.info("Generating advanced performance report...")
        
        test_results = self.results['test']
        val_results = self.results['validation']
        
        report = f"""
# ADVANCED FOREX TRADING SYSTEM - PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## STRATEGY OVERVIEW
- Smart Money Concepts (ICT) Integration
- Harmonic Pattern Recognition
- Supply/Demand Zone Analysis
- Multi-Confirmation Trading
- Advanced Risk Management

## MODEL PERFORMANCE
Training Win Rate: {self.results['training']['win_rate']:.2%}
Validation Win Rate: {val_results['win_rate']:.2%}
Test Win Rate: {test_results['win_rate']:.2%}

## TRADING PERFORMANCE - OUT OF SAMPLE TEST

### Returns & Risk Metrics
- Total Return: {test_results['total_return']:.2f}%
- Win Rate: {test_results['win_rate']:.2%}
- Risk-Reward Ratio: {test_results['risk_reward_ratio']:.2f}
- Profit Factor: {test_results['profit_factor']:.2f}
- Sharpe Ratio: {test_results['sharpe_ratio']:.2f}
- Sortino Ratio: {test_results['sortino_ratio']:.2f}
- Calmar Ratio: {test_results['calmar_ratio']:.2f}
- Max Drawdown: {test_results['max_drawdown']:.2f}%

### Trade Statistics
- Total Trades: {test_results['total_trades']}
- Average Win: ${test_results['avg_win']:.2f}
- Average Loss: ${test_results['avg_loss']:.2f}
- Max Consecutive Wins: {test_results['consecutive_wins']}
- Max Consecutive Losses: {test_results['consecutive_losses']}
- Signal-to-Trade Ratio: {test_results['signal_to_trade_ratio']:.2%}

## VALIDATION PERIOD COMPARISON
- Validation Return: {val_results['total_return']:.2f}%
- Validation Win Rate: {val_results['win_rate']:.2%}
- Validation Sharpe: {val_results['sharpe_ratio']:.2f}

## KEY INSIGHTS & RECOMMENDATIONS
"""
        
        # Add performance-based recommendations
        win_rate = test_results['win_rate']
        sharpe = test_results['sharpe_ratio']
        profit_factor = test_results['profit_factor']
        
        if win_rate >= 0.70:
            report += "✅ EXCELLENT: Win rate exceeds 70% - System demonstrates high accuracy\n"
        elif win_rate >= 0.60:
            report += "✅ STRONG: Win rate above 60% - System ready for live trading with monitoring\n"
        elif win_rate >= 0.55:
            report += "⚡ GOOD: Win rate above 55% - Consider fine-tuning confidence thresholds\n"
        else:
            report += "❌ NEEDS IMPROVEMENT: Win rate below 55% - Review strategy parameters\n"
        
        if sharpe >= 2.0:
            report += "✅ EXCELLENT: Sharpe ratio > 2.0 - Outstanding risk-adjusted returns\n"
        elif sharpe >= 1.5:
            report += "✅ STRONG: Sharpe ratio > 1.5 - Very good risk-adjusted returns\n"
        elif sharpe >= 1.0:
            report += "⚡ GOOD: Sharpe ratio > 1.0 - Positive risk-adjusted returns\n"
        else:
            report += "❌ CAUTION: Sharpe ratio < 1.0 - Risk-adjusted returns need improvement\n"
        
        if profit_factor >= 2.0:
            report += "✅ EXCELLENT: Profit factor > 2.0 - Strong profitability\n"
        elif profit_factor >= 1.5:
            report += "✅ STRONG: Profit factor > 1.5 - Good profit/loss ratio\n"
        else:
            report += "⚡ ADEQUATE: Profit factor < 1.5 - Consider improving exit strategies\n"
        
        # Strategy-specific insights
        report += "\n## ADVANCED STRATEGY PERFORMANCE\n"
        
        if 'confirmation_analysis' in test_results and test_results['confirmation_analysis']:
            report += "\n### Confirmation Type Analysis\n"
            for conf_type, stats in test_results['confirmation_analysis'].items():
                report += f"- {conf_type}: {stats['trades']} trades, "
                report += f"{stats['win_rate']:.1%} win rate, "
                report += f"${stats['avg_pnl']:.2f} avg P&L\n"
        
        report += """
## NEXT STEPS FOR IMPROVEMENT

1. **Fine-tune Confidence Thresholds**
   - Current minimum: 0.65
   - Consider dynamic thresholds based on market volatility
   
2. **Optimize Risk Management**
   - Implement partial profit taking at 1.5x risk
   - Use trailing stops after 2x risk achieved
   
3. **Enhanced Market Filtering**
   - Avoid trading during major news events
   - Filter by market session (focus on London/NY overlap)
   
4. **Position Sizing Optimization**
   - Scale position size based on confidence level
   - Reduce size during high volatility periods

5. **Live Trading Preparation**
   - Start with micro lots to validate performance
   - Monitor slippage and execution quality
   - Implement real-time performance tracking
"""
        
        # Save report
        with open('logs/advanced_performance_report.md', 'w') as f:
            f.write(report)
        
        logger.info("Advanced performance report saved to logs/advanced_performance_report.md")
        print(report)
    
    def create_advanced_visualizations(self):
        """Create advanced performance visualizations."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('seaborn-v0_8-darkgrid')
            fig = plt.figure(figsize=(20, 12))
            
            # Create a 3x3 grid of subplots
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Portfolio value over time
            ax1 = fig.add_subplot(gs[0, :2])
            test_portfolio = self.results['test']['portfolio_values']
            ax1.plot(test_portfolio, linewidth=2, color='#2E86AB')
            ax1.fill_between(range(len(test_portfolio)), test_portfolio, 
                           alpha=0.3, color='#2E86AB')
            ax1.set_title('Portfolio Value Over Time (Test Period)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Add drawdown visualization
            portfolio_series = pd.Series(test_portfolio)
            rolling_max = portfolio_series.expanding().max()
            drawdown = (portfolio_series - rolling_max) / rolling_max * 100
            
            ax1_twin = ax1.twinx()
            ax1_twin.fill_between(range(len(drawdown)), drawdown, 0, 
                                alpha=0.3, color='red', label='Drawdown')
            ax1_twin.set_ylabel('Drawdown (%)', fontsize=12)
            ax1_twin.set_ylim(-20, 5)
            
            # 2. Win rate comparison across periods
            ax2 = fig.add_subplot(gs[0, 2])
            periods = ['Training', 'Validation', 'Test']
            win_rates = [
                self.results['training']['win_rate'],
                self.results['validation']['win_rate'],
                self.results['test']['win_rate']
            ]
            colors = ['#A23B72', '#F18F01', '#C73E1D']
            bars = ax2.bar(periods, win_rates, color=colors, alpha=0.8)
            ax2.set_title('Win Rate by Period', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Win Rate', fontsize=12)
            ax2.set_ylim(0, 1)
            ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Target (60%)')
            
            # Add value labels on bars
            for bar, rate in zip(bars, win_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')
            
            # 3. Trade P&L distribution
            ax3 = fig.add_subplot(gs[1, 0])
            test_trades = self.results['test']['trades']
            if test_trades:
                pnl_values = [trade['pnl'] for trade in test_trades]
                ax3.hist(pnl_values, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax3.axvline(x=np.mean(pnl_values), color='green', linestyle='-', 
                          linewidth=2, label=f'Mean: ${np.mean(pnl_values):.2f}')
                ax3.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
                ax3.set_xlabel('P&L ($)', fontsize=12)
                ax3.set_ylabel('Frequency', fontsize=12)
                ax3.legend()
            
            # 4. Risk metrics comparison
            ax4 = fig.add_subplot(gs[1, 1])
            metrics = ['Sharpe', 'Sortino', 'Calmar']
            test_values = [
                self.results['test']['sharpe_ratio'],
                self.results['test']['sortino_ratio'],
                self.results['test']['calmar_ratio']
            ]
            val_values = [
                self.results['validation']['sharpe_ratio'],
                self.results['validation']['sortino_ratio'],
                self.results['validation']['calmar_ratio']
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            ax4.bar(x - width/2, val_values, width, label='Validation', color='#F18F01', alpha=0.8)
            ax4.bar(x + width/2, test_values, width, label='Test', color='#C73E1D', alpha=0.8)
            ax4.set_xlabel('Risk Metrics', fontsize=12)
            ax4.set_title('Risk-Adjusted Performance Metrics', fontsize=14, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Monthly returns heatmap
            ax5 = fig.add_subplot(gs[1, 2])
            if test_trades:
                # Create monthly returns
                monthly_returns = {}
                for trade in test_trades:
                    month_key = trade['exit_time'].strftime('%Y-%m')
                    if month_key not in monthly_returns:
                        monthly_returns[month_key] = 0
                    monthly_returns[month_key] += trade['pnl_pct']
                
                if monthly_returns:
                    months = sorted(monthly_returns.keys())
                    returns = [monthly_returns[m] for m in months]
                    
                    # Create a simple bar chart for monthly returns
                    ax5.bar(range(len(months)), returns, 
                           color=['green' if r > 0 else 'red' for r in returns], alpha=0.7)
                    ax5.set_title('Monthly Returns', fontsize=14, fontweight='bold')
                    ax5.set_xlabel('Month', fontsize=12)
                    ax5.set_ylabel('Return (%)', fontsize=12)
                    ax5.set_xticks(range(len(months)))
                    ax5.set_xticklabels([m[-2:] for m in months], rotation=45)
                    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 6. Win/Loss streaks
            ax6 = fig.add_subplot(gs[2, 0])
            if test_trades:
                results = [1 if t['pnl'] > 0 else -1 for t in test_trades]
                cumsum = np.cumsum(results)
                ax6.plot(cumsum, linewidth=2, color='#2E86AB')
                ax6.fill_between(range(len(cumsum)), cumsum, 0, 
                               where=(cumsum >= 0), alpha=0.3, color='green', label='Winning')
                ax6.fill_between(range(len(cumsum)), cumsum, 0, 
                               where=(cumsum < 0), alpha=0.3, color='red', label='Losing')
                ax6.set_title('Cumulative Win/Loss Performance', fontsize=14, fontweight='bold')
                ax6.set_xlabel('Trade Number', fontsize=12)
                ax6.set_ylabel('Cumulative Wins - Losses', fontsize=12)
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            
            # 7. Confirmation type performance
            ax7 = fig.add_subplot(gs[2, 1])
            conf_analysis = self.results['test'].get('confirmation_analysis', {})
            if conf_analysis:
                conf_types = list(conf_analysis.keys())
                conf_win_rates = [conf_analysis[ct]['win_rate'] for ct in conf_types]
                conf_trades = [conf_analysis[ct]['trades'] for ct in conf_types]
                
                # Create bubble chart
                for i, (ct, wr, trades) in enumerate(zip(conf_types, conf_win_rates, conf_trades)):
                    ax7.scatter(i, wr, s=trades*20, alpha=0.6, label=f'{ct} ({trades} trades)')
                
                ax7.set_xticks(range(len(conf_types)))
                ax7.set_xticklabels(conf_types, rotation=45)
                ax7.set_ylabel('Win Rate', fontsize=12)
                ax7.set_title('Confirmation Type Performance', fontsize=14, fontweight='bold')
                ax7.set_ylim(0, 1)
                ax7.axhline(y=0.6, color='green', linestyle='--', alpha=0.5)
                ax7.legend()
                ax7.grid(True, alpha=0.3)
            
            # 8. Exit reason analysis
            ax8 = fig.add_subplot(gs[2, 2])
            if test_trades:
                exit_reasons = {}
                for trade in test_trades:
                    reason = trade['reason']
                    if reason not in exit_reasons:
                        exit_reasons[reason] = {'wins': 0, 'losses': 0}
                    if trade['pnl'] > 0:
                        exit_reasons[reason]['wins'] += 1
                    else:
                        exit_reasons[reason]['losses'] += 1
                
                reasons = list(exit_reasons.keys())
                wins = [exit_reasons[r]['wins'] for r in reasons]
                losses = [exit_reasons[r]['losses'] for r in reasons]
                
                x = np.arange(len(reasons))
                width = 0.35
                ax8.bar(x - width/2, wins, width, label='Wins', color='green', alpha=0.7)
                ax8.bar(x + width/2, losses, width, label='Losses', color='red', alpha=0.7)
                ax8.set_xlabel('Exit Reason', fontsize=12)
                ax8.set_ylabel('Count', fontsize=12)
                ax8.set_title('Trade Exits by Reason', fontsize=14, fontweight='bold')
                ax8.set_xticks(x)
                ax8.set_xticklabels(reasons, rotation=45)
                ax8.legend()
            
            plt.suptitle('Advanced Forex Trading System - Performance Analysis', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('logs/advanced_performance_charts.png', dpi=300, bbox_inches='tight')
            logger.info("Advanced performance charts saved to logs/advanced_performance_charts.png")
            
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")


def main():
    """Main function to run advanced backtesting."""
    logger.info("Starting Advanced Forex Backtesting System with Smart Money Concepts")
    
    # Initialize backtester
    backtester = AdvancedBacktester(initial_balance=10000)
    
    # Load data
    df = backtester.load_and_prepare_data("data/eurusd_daily_alpha.csv")
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Run comprehensive backtest with advanced strategies
    results = backtester.run_advanced_backtest(
        df, 
        min_confidence=0.65,
        use_multiple_confirmations=True
    )
    
    # Summary
    test_results = results['test']
    logger.info("=" * 60)
    logger.info("ADVANCED BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Final Win Rate: {test_results['win_rate']:.2%}")
    logger.info(f"Total Return: {test_results['total_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
    logger.info(f"Profit Factor: {test_results['profit_factor']:.2f}")
    logger.info(f"Max Drawdown: {test_results['max_drawdown']:.2f}%")
    logger.info("=" * 60)
    
    if test_results['win_rate'] >= 0.65:
        logger.info("🎯 EXCELLENT: System achieved 65%+ win rate with advanced strategies!")
    elif test_results['win_rate'] >= 0.60:
        logger.info("✅ SUCCESS: System achieved target 60%+ win rate!")
    else:
        logger.info("📈 Continue optimizing to reach target win rate")


if __name__ == "__main__":
    main()