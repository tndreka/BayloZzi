# core/risk_manager.py
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class AdvancedRiskManager:
    """
    Sophisticated risk management system for forex trading.
    Implements multiple risk control mechanisms to maximize win rate and minimize losses.
    """
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_drawdown = 0.15  # 15% max drawdown
        
        # Position tracking
        self.open_positions = {}
        self.trade_history = []
        self.daily_pnl = {}
        
        # Performance metrics
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.sharpe_ratio = 0.0
        
        # Risk controls
        self.is_trading_enabled = True
        self.daily_loss_reached = False
        self.max_concurrent_positions = 3
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              confidence: float, balance: float = None) -> float:
        """
        Calculate optimal position size based on risk parameters and confidence.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            confidence: Model confidence (0-1)
            balance: Current account balance
            
        Returns:
            Position size as a fraction of balance
        """
        if balance is None:
            balance = self.current_balance
            
        # Risk per trade adjusted by confidence
        base_risk = self.max_risk_per_trade
        confidence_multiplier = max(0.5, min(1.5, confidence * 1.5))
        adjusted_risk = base_risk * confidence_multiplier
        
        # Calculate position size based on stop loss distance
        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0:
            return 0
            
        # Position size = (Risk Amount) / (Price Difference)
        risk_amount = balance * adjusted_risk
        position_size = risk_amount / price_diff
        
        # Maximum position size constraint (10% of balance)
        max_position = balance * 0.1
        position_size = min(position_size, max_position)
        
        logger.info(f"Position sizing: Risk={adjusted_risk:.3f}, Size={position_size:.2f}")
        return position_size
    
    def validate_trade(self, signal: int, confidence: float, current_price: float) -> Dict:
        """
        Comprehensive trade validation with multiple risk checks.
        
        Returns:
            Dict with validation result and reasoning
        """
        validation = {
            'approved': False,
            'reasons': [],
            'risk_level': 'high'
        }
        
        # Check if trading is enabled
        if not self.is_trading_enabled:
            validation['reasons'].append("Trading disabled due to risk limits")
            return validation
            
        # Check daily loss limit
        today = datetime.now().date()
        daily_loss = self.daily_pnl.get(today, 0)
        max_daily_loss_amount = self.initial_balance * self.max_daily_loss
        
        if daily_loss <= -max_daily_loss_amount:
            validation['reasons'].append(f"Daily loss limit reached: {daily_loss:.2f}")
            self.daily_loss_reached = True
            return validation
            
        # Check drawdown
        current_drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
        if current_drawdown >= self.max_drawdown:
            validation['reasons'].append(f"Maximum drawdown exceeded: {current_drawdown:.2%}")
            self.is_trading_enabled = False
            return validation
            
        # Check confidence threshold
        min_confidence = self.get_dynamic_confidence_threshold()
        if confidence < min_confidence:
            validation['reasons'].append(f"Confidence {confidence:.3f} below threshold {min_confidence:.3f}")
            return validation
            
        # Check maximum concurrent positions
        if len(self.open_positions) >= self.max_concurrent_positions:
            validation['reasons'].append(f"Maximum positions limit reached: {len(self.open_positions)}")
            return validation
            
        # Market condition checks
        if not self.check_market_conditions():
            validation['reasons'].append("Unfavorable market conditions detected")
            return validation
            
        # All checks passed
        validation['approved'] = True
        validation['risk_level'] = self.assess_risk_level(confidence, current_price)
        validation['reasons'].append("All risk checks passed")
        
        return validation
    
    def get_dynamic_confidence_threshold(self) -> float:
        """
        Calculate dynamic confidence threshold based on recent performance.
        Lower threshold when performing well, higher when struggling.
        """
        base_threshold = 0.65
        
        # Adjust based on recent win rate
        if len(self.trade_history) >= 10:
            recent_trades = self.trade_history[-10:]
            recent_wins = sum(1 for trade in recent_trades if trade['pnl'] > 0)
            recent_win_rate = recent_wins / len(recent_trades)
            
            if recent_win_rate > 0.7:
                # Performing well, can be more aggressive
                return max(0.55, base_threshold - 0.1)
            elif recent_win_rate < 0.4:
                # Struggling, be more conservative
                return min(0.8, base_threshold + 0.15)
                
        return base_threshold
    
    def check_market_conditions(self) -> bool:
        """
        Assess current market conditions for trading suitability.
        Returns True if conditions are favorable.
        """
        # In a real implementation, this would check:
        # - Market volatility
        # - Economic calendar events
        # - Market session times
        # - Weekend/holiday status
        
        current_hour = datetime.now().hour
        
        # Avoid trading during low liquidity hours (typically 22:00-03:00 UTC)
        if current_hour >= 22 or current_hour <= 3:
            return False
            
        # Check if it's weekend
        if datetime.now().weekday() >= 5:  # Saturday or Sunday
            return False
            
        return True
    
    def assess_risk_level(self, confidence: float, price: float) -> str:
        """Assess the risk level of the potential trade."""
        if confidence >= 0.8:
            return 'low'
        elif confidence >= 0.7:
            return 'medium'
        else:
            return 'high'
    
    def calculate_stop_loss_take_profit(self, entry_price: float, signal: int, 
                                      volatility: float = 0.001) -> Tuple[float, float]:
        """
        Calculate optimal stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            signal: 1 for buy, -1 for sell
            volatility: Estimated price volatility
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        # Dynamic risk-reward ratio based on market conditions
        base_risk_reward = 2.0
        
        # Adjust for volatility
        volatility_multiplier = max(1.0, volatility / 0.001)
        stop_distance = volatility * volatility_multiplier * 2
        
        if signal == 1:  # Long position
            stop_loss = entry_price * (1 - stop_distance)
            take_profit = entry_price * (1 + stop_distance * base_risk_reward)
        else:  # Short position
            stop_loss = entry_price * (1 + stop_distance)
            take_profit = entry_price * (1 - stop_distance * base_risk_reward)
            
        return stop_loss, take_profit
    
    def open_position(self, symbol: str, signal: int, entry_price: float, 
                     stop_loss: float, take_profit: float, position_size: float,
                     confidence: float) -> str:
        """
        Open a new position and track it.
        
        Returns:
            Position ID
        """
        position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        position = {
            'id': position_id,
            'symbol': symbol,
            'signal': signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'confidence': confidence,
            'entry_time': datetime.now(),
            'status': 'open'
        }
        
        self.open_positions[position_id] = position
        
        logger.info(f"Position opened: {position_id}")
        logger.info(f"Entry: {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
        
        return position_id
    
    def close_position(self, position_id: str, exit_price: float, reason: str) -> Dict:
        """
        Close an existing position and calculate P&L.
        
        Returns:
            Dictionary with trade results
        """
        if position_id not in self.open_positions:
            logger.error(f"Position {position_id} not found")
            return {}
            
        position = self.open_positions[position_id]
        
        # Calculate P&L
        price_diff = exit_price - position['entry_price']
        if position['signal'] == -1:  # Short position
            price_diff = -price_diff
            
        pnl = price_diff * position['position_size']
        pnl_pct = (price_diff / position['entry_price']) * 100
        
        # Update balance
        self.current_balance += pnl
        
        # Create trade record
        trade_record = {
            'position_id': position_id,
            'symbol': position['symbol'],
            'signal': position['signal'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'confidence': position['confidence'],
            'reason': reason,
            'position_size': position['position_size']
        }
        
        self.trade_history.append(trade_record)
        
        # Update daily P&L
        today = datetime.now().date()
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0
        self.daily_pnl[today] += pnl
        
        # Remove from open positions
        del self.open_positions[position_id]
        
        # Update performance metrics
        self.update_performance_metrics()
        
        logger.info(f"Position closed: {position_id} - {reason}")
        logger.info(f"P&L: {pnl:.2f} ({pnl_pct:.2f}%) - Balance: {self.current_balance:.2f}")
        
        return trade_record
    
    def check_position_exits(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check all open positions for exit conditions.
        
        Args:
            current_prices: Dict of symbol -> current_price
            
        Returns:
            List of closed trades
        """
        closed_trades = []
        positions_to_close = []
        
        for position_id, position in self.open_positions.items():
            symbol = position['symbol']
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Check stop loss
            if position['signal'] == 1:  # Long
                if current_price <= position['stop_loss']:
                    positions_to_close.append((position_id, current_price, 'Stop Loss'))
                elif current_price >= position['take_profit']:
                    positions_to_close.append((position_id, current_price, 'Take Profit'))
            else:  # Short
                if current_price >= position['stop_loss']:
                    positions_to_close.append((position_id, current_price, 'Stop Loss'))
                elif current_price <= position['take_profit']:
                    positions_to_close.append((position_id, current_price, 'Take Profit'))
        
        # Close positions
        for position_id, exit_price, reason in positions_to_close:
            trade_record = self.close_position(position_id, exit_price, reason)
            closed_trades.append(trade_record)
            
        return closed_trades
    
    def update_performance_metrics(self):
        """Update performance statistics."""
        if not self.trade_history:
            return
            
        # Win rate
        wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        self.win_rate = wins / len(self.trade_history)
        
        # Average win/loss
        winning_trades = [trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0]
        losing_trades = [trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0]
        
        self.avg_win = np.mean(winning_trades) if winning_trades else 0
        self.avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Sharpe ratio (simplified)
        returns = [trade['pnl_pct'] for trade in self.trade_history]
        if len(returns) > 1:
            self.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_return': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            'win_rate': self.win_rate * 100,
            'total_trades': len(self.trade_history),
            'open_positions': len(self.open_positions),
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': ((self.initial_balance - min(self.current_balance, self.initial_balance)) / self.initial_balance) * 100,
            'is_trading_enabled': self.is_trading_enabled
        }