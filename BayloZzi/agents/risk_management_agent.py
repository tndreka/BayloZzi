# agents/risk_management_agent.py

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import redis
from ..core.risk_manager import AdvancedRiskManager

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    message_type: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    risk_level: RiskLevel = RiskLevel.MEDIUM

@dataclass
class MarketCondition:
    volatility: float
    liquidity: str
    news_impact: str
    session: str  # asian, london, new_york, overlap
    economic_events: List[str]

class MultiAgentRiskManager(AdvancedRiskManager):
    """
    Enhanced Risk Management Agent for Multi-Agent Forex Trading System.
    Integrates with other agents to provide comprehensive risk assessment.
    """
    
    def __init__(self, initial_balance: float = 10000, agent_id: str = "risk_manager"):
        super().__init__(initial_balance)
        self.agent_id = agent_id
        
        # Redis connection for inter-agent communication
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Enhanced risk parameters
        self.agent_risk_limits = {
            'chart_analysis': {'max_positions': 2, 'confidence_threshold': 0.70, 'max_risk': 0.015},
            'trend_identification': {'max_positions': 3, 'confidence_threshold': 0.65, 'max_risk': 0.020},
            'news_sentiment': {'max_positions': 1, 'confidence_threshold': 0.80, 'max_risk': 0.010},
            'economic_factors': {'max_positions': 1, 'confidence_threshold': 0.75, 'max_risk': 0.012}
        }
        
        # Market condition monitoring
        self.current_market_condition = MarketCondition(
            volatility=0.0, liquidity="normal", news_impact="low", 
            session="london", economic_events=[]
        )
        
        # Agent performance tracking
        self.agent_performance = {
            'chart_analysis': {'accuracy': 0.0, 'sharpe': 0.0, 'trades': 0},
            'trend_identification': {'accuracy': 0.0, 'sharpe': 0.0, 'trades': 0},
            'news_sentiment': {'accuracy': 0.0, 'sharpe': 0.0, 'trades': 0},
            'economic_factors': {'accuracy': 0.0, 'sharpe': 0.0, 'trades': 0}
        }
        
        # Risk alerts and monitoring
        self.risk_alerts = []
        self.emergency_stop_triggered = False
        
        # Correlation matrix for currency pairs
        self.correlation_matrix = self._initialize_correlation_matrix()
        
        # Real-time risk monitoring
        self.risk_metrics = {
            'current_var': 0.0,  # Value at Risk
            'portfolio_beta': 0.0,
            'concentration_risk': 0.0,
            'liquidity_risk': 0.0
        }
    
    def _initialize_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize currency pair correlation matrix."""
        return {
            'EURUSD': {'GBPUSD': 0.7, 'USDJPY': -0.6, 'USDCHF': -0.8, 'AUDUSD': 0.6},
            'GBPUSD': {'EURUSD': 0.7, 'USDJPY': -0.5, 'USDCHF': -0.7, 'AUDUSD': 0.5},
            'USDJPY': {'EURUSD': -0.6, 'GBPUSD': -0.5, 'USDCHF': 0.7, 'AUDUSD': -0.4},
            'USDCHF': {'EURUSD': -0.8, 'GBPUSD': -0.7, 'USDJPY': 0.7, 'AUDUSD': -0.5},
            'AUDUSD': {'EURUSD': 0.6, 'GBPUSD': 0.5, 'USDJPY': -0.4, 'USDCHF': -0.5}
        }
    
    async def process_agent_signal(self, message: AgentMessage) -> Dict[str, Any]:
        """
        Process trading signal from another agent with comprehensive risk assessment.
        
        Args:
            message: AgentMessage containing trading signal
            
        Returns:
            Risk assessment and position sizing recommendation
        """
        try:
            signal_data = message.data
            agent_name = message.sender
            confidence = message.confidence
            
            # Extract signal information
            symbol = signal_data.get('symbol', 'EURUSD')
            signal_type = signal_data.get('signal', 0)  # 1 for buy, -1 for sell
            entry_price = signal_data.get('entry_price', 0.0)
            
            # Comprehensive risk assessment
            risk_assessment = await self._comprehensive_risk_assessment(
                agent_name, symbol, signal_type, entry_price, confidence, signal_data
            )
            
            # Log the assessment
            logger.info(f"Risk assessment for {agent_name} signal on {symbol}: {risk_assessment}")
            
            # Send response back to requesting agent
            await self._send_agent_message(
                receiver=agent_name,
                message_type="risk_assessment",
                data=risk_assessment,
                confidence=risk_assessment['confidence']
            )
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error processing agent signal: {str(e)}")
            return {'approved': False, 'error': str(e)}
    
    async def _comprehensive_risk_assessment(self, agent_name: str, symbol: str, 
                                           signal_type: int, entry_price: float, 
                                           confidence: float, signal_data: Dict) -> Dict[str, Any]:
        """Perform comprehensive multi-dimensional risk assessment."""
        
        assessment = {
            'approved': False,
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'risk_level': RiskLevel.HIGH.value,
            'confidence': 0.0,
            'reasons': [],
            'agent_allocation': 0.0
        }
        
        try:
            # 1. Agent-specific risk limits
            agent_limits = self.agent_risk_limits.get(agent_name, {})
            if not self._check_agent_limits(agent_name, confidence, agent_limits):
                assessment['reasons'].append(f"Agent {agent_name} limits exceeded")
                return assessment
            
            # 2. Portfolio-level risk checks
            if not self._check_portfolio_risk(symbol):
                assessment['reasons'].append("Portfolio risk limits exceeded")
                return assessment
            
            # 3. Market condition assessment
            market_risk = await self._assess_market_conditions(symbol, signal_data)
            if market_risk['risk_level'] == RiskLevel.CRITICAL.value:
                assessment['reasons'].append("Critical market conditions detected")
                return assessment
            
            # 4. Correlation and concentration risk
            correlation_risk = self._assess_correlation_risk(symbol, signal_type)
            if correlation_risk > 0.8:
                assessment['reasons'].append("High correlation risk detected")
                return assessment
            
            # 5. Calculate optimal position size
            position_size = self._calculate_multi_agent_position_size(
                agent_name, symbol, entry_price, confidence, signal_data
            )
            
            if position_size <= 0:
                assessment['reasons'].append("Position size calculation failed")
                return assessment
            
            # 6. Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_enhanced_exits(
                entry_price, signal_type, symbol, market_risk
            )
            
            # 7. Final validation
            if self._final_risk_validation(position_size, stop_loss, take_profit, entry_price):
                assessment.update({
                    'approved': True,
                    'position_size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_level': market_risk['risk_level'],
                    'confidence': min(confidence * 0.9, 0.95),  # Slight confidence reduction for safety
                    'reasons': ['All risk checks passed'],
                    'agent_allocation': agent_limits.get('max_risk', 0.02)
                })
            else:
                assessment['reasons'].append("Final risk validation failed")
                
        except Exception as e:
            logger.error(f"Error in comprehensive risk assessment: {str(e)}")
            assessment['reasons'].append(f"Assessment error: {str(e)}")
        
        return assessment
    
    def _check_agent_limits(self, agent_name: str, confidence: float, limits: Dict) -> bool:
        """Check agent-specific risk limits."""
        if confidence < limits.get('confidence_threshold', 0.7):
            return False
        
        # Check if agent has reached max positions
        agent_positions = [pos for pos in self.open_positions.values() 
                          if pos.get('agent_source') == agent_name]
        if len(agent_positions) >= limits.get('max_positions', 2):
            return False
        
        return True
    
    def _check_portfolio_risk(self, symbol: str) -> bool:
        """Check portfolio-level risk constraints."""
        # Check total open positions
        if len(self.open_positions) >= self.max_concurrent_positions:
            return False
        
        # Check daily loss limit
        today = datetime.now().date()
        daily_loss = self.daily_pnl.get(today, 0)
        if daily_loss <= -self.initial_balance * self.max_daily_loss:
            return False
        
        # Check overall drawdown
        current_drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
        if current_drawdown >= self.max_drawdown:
            return False
        
        return True
    
    async def _assess_market_conditions(self, symbol: str, signal_data: Dict) -> Dict[str, Any]:
        """Assess current market conditions for risk evaluation."""
        try:
            # Get market condition data from other agents
            news_impact = await self._get_news_impact(symbol)
            volatility = signal_data.get('volatility', 0.001)
            economic_events = await self._get_economic_events()
            
            # Determine risk level based on conditions
            risk_level = RiskLevel.LOW
            
            if volatility > 0.005:  # High volatility
                risk_level = RiskLevel.HIGH
            elif news_impact == "high" or len(economic_events) > 2:
                risk_level = RiskLevel.MEDIUM
            
            # Check trading session
            current_hour = datetime.now().hour
            if 22 <= current_hour or current_hour <= 3:  # Low liquidity hours
                risk_level = RiskLevel.HIGH
            
            return {
                'risk_level': risk_level.value,
                'volatility': volatility,
                'news_impact': news_impact,
                'economic_events': economic_events,
                'liquidity_score': self._calculate_liquidity_score(current_hour)
            }
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {str(e)}")
            return {'risk_level': RiskLevel.HIGH.value}
    
    def _assess_correlation_risk(self, symbol: str, signal_type: int) -> float:
        """Assess correlation risk with existing positions."""
        if not self.open_positions:
            return 0.0
        
        total_correlation = 0.0
        for position in self.open_positions.values():
            existing_symbol = position['symbol']
            existing_signal = position['signal']
            
            # Get correlation coefficient
            correlation = self.correlation_matrix.get(symbol, {}).get(existing_symbol, 0.0)
            
            # If signals are in same direction and pairs are correlated, increase risk
            if signal_type == existing_signal and correlation > 0.5:
                total_correlation += correlation
            # If signals are opposite and pairs are negatively correlated, increase risk
            elif signal_type != existing_signal and correlation < -0.5:
                total_correlation += abs(correlation)
        
        return min(total_correlation, 1.0)
    
    def _calculate_multi_agent_position_size(self, agent_name: str, symbol: str, 
                                           entry_price: float, confidence: float, 
                                           signal_data: Dict) -> float:
        """Calculate position size considering multi-agent factors."""
        try:
            # Base position size calculation
            base_size = super().calculate_position_size(
                entry_price, 
                signal_data.get('stop_loss', entry_price * 0.99), 
                confidence, 
                self.current_balance
            )
            
            # Agent-specific adjustments
            agent_limits = self.agent_risk_limits.get(agent_name, {})
            agent_max_risk = agent_limits.get('max_risk', 0.02)
            
            # Adjust based on agent performance
            agent_perf = self.agent_performance.get(agent_name, {})
            performance_multiplier = max(0.5, min(1.5, agent_perf.get('accuracy', 0.5) * 2))
            
            # Market condition adjustment
            volatility = signal_data.get('volatility', 0.001)
            volatility_multiplier = max(0.5, min(1.2, 1.0 / (volatility * 1000 + 1)))
            
            # Calculate final position size
            adjusted_size = base_size * performance_multiplier * volatility_multiplier
            
            # Apply agent-specific maximum
            max_agent_size = self.current_balance * agent_max_risk / abs(entry_price - signal_data.get('stop_loss', entry_price * 0.99))
            
            return min(adjusted_size, max_agent_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def _calculate_enhanced_exits(self, entry_price: float, signal_type: int, 
                                symbol: str, market_risk: Dict) -> Tuple[float, float]:
        """Calculate enhanced stop loss and take profit levels."""
        try:
            volatility = market_risk.get('volatility', 0.001)
            risk_level = market_risk.get('risk_level', RiskLevel.MEDIUM.value)
            
            # Adjust stop distance based on risk level
            base_stop_distance = volatility * 2
            
            if risk_level == RiskLevel.HIGH.value:
                stop_distance = base_stop_distance * 1.5
            elif risk_level == RiskLevel.LOW.value:
                stop_distance = base_stop_distance * 0.8
            else:
                stop_distance = base_stop_distance
            
            # Dynamic risk-reward ratio
            if risk_level == RiskLevel.HIGH.value:
                risk_reward = 2.5  # Higher reward for higher risk
            else:
                risk_reward = 2.0
            
            if signal_type == 1:  # Long position
                stop_loss = entry_price * (1 - stop_distance)
                take_profit = entry_price * (1 + stop_distance * risk_reward)
            else:  # Short position
                stop_loss = entry_price * (1 + stop_distance)
                take_profit = entry_price * (1 - stop_distance * risk_reward)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating exits: {str(e)}")
            return entry_price * 0.99, entry_price * 1.01
    
    def _final_risk_validation(self, position_size: float, stop_loss: float, 
                             take_profit: float, entry_price: float) -> bool:
        """Perform final risk validation checks."""
        try:
            # Check if position size is reasonable
            if position_size <= 0 or position_size > self.current_balance * 0.1:
                return False
            
            # Check if stop loss and take profit are reasonable
            stop_distance = abs(entry_price - stop_loss) / entry_price
            if stop_distance > 0.05:  # More than 5% stop distance is too risky
                return False
            
            # Check risk-reward ratio
            profit_distance = abs(take_profit - entry_price)
            loss_distance = abs(entry_price - stop_loss)
            if profit_distance / loss_distance < 1.5:  # Minimum 1.5:1 risk-reward
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in final validation: {str(e)}")
            return False
    
    async def _send_agent_message(self, receiver: str, message_type: str, 
                                data: Dict, confidence: float):
        """Send message to another agent via Redis."""
        try:
            message = AgentMessage(
                sender=self.agent_id,
                receiver=receiver,
                message_type=message_type,
                data=data,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Publish to Redis channel
            channel = f"agent_messages_{receiver}"
            message_json = json.dumps(message.__dict__, default=str)
            self.redis_client.publish(channel, message_json)
            
        except Exception as e:
            logger.error(f"Error sending agent message: {str(e)}")
    
    async def _get_news_impact(self, symbol: str) -> str:
        """Get news impact assessment from news sentiment agent."""
        try:
            # Request news impact from news agent
            request_data = {'symbol': symbol, 'timeframe': '24h'}
            await self._send_agent_message(
                receiver="news_sentiment",
                message_type="news_impact_request",
                data=request_data,
                confidence=1.0
            )
            
            # Wait for response (simplified - in production, use proper async handling)
            await asyncio.sleep(0.1)
            return "medium"  # Default value
            
        except Exception as e:
            logger.error(f"Error getting news impact: {str(e)}")
            return "unknown"
    
    async def _get_economic_events(self) -> List[str]:
        """Get upcoming economic events from economic factors agent."""
        try:
            # Request economic events
            await self._send_agent_message(
                receiver="economic_factors",
                message_type="economic_events_request",
                data={'timeframe': '24h'},
                confidence=1.0
            )
            
            # Wait for response (simplified)
            await asyncio.sleep(0.1)
            return []  # Default empty list
            
        except Exception as e:
            logger.error(f"Error getting economic events: {str(e)}")
            return []
    
    def _calculate_liquidity_score(self, current_hour: int) -> float:
        """Calculate liquidity score based on trading session."""
        # London session (8-17 GMT): High liquidity
        if 8 <= current_hour <= 17:
            return 1.0
        # New York session (13-22 GMT): High liquidity
        elif 13 <= current_hour <= 22:
            return 1.0
        # Asian session (23-8 GMT): Medium liquidity
        elif current_hour >= 23 or current_hour <= 8:
            return 0.7
        # Low liquidity hours
        else:
            return 0.3
    
    def update_agent_performance(self, agent_name: str, trade_result: Dict):
        """Update performance metrics for specific agent."""
        try:
            if agent_name not in self.agent_performance:
                self.agent_performance[agent_name] = {'accuracy': 0.0, 'sharpe': 0.0, 'trades': 0}
            
            perf = self.agent_performance[agent_name]
            perf['trades'] += 1
            
            # Update accuracy (simplified)
            if trade_result.get('pnl', 0) > 0:
                perf['accuracy'] = (perf['accuracy'] * (perf['trades'] - 1) + 1) / perf['trades']
            else:
                perf['accuracy'] = (perf['accuracy'] * (perf['trades'] - 1)) / perf['trades']
            
            logger.info(f"Updated performance for {agent_name}: {perf}")
            
        except Exception as e:
            logger.error(f"Error updating agent performance: {str(e)}")
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data."""
        return {
            'account_status': {
                'balance': self.current_balance,
                'equity': self.current_balance,  # Simplified
                'margin_used': sum(pos['position_size'] for pos in self.open_positions.values()),
                'free_margin': self.current_balance * 0.9  # Simplified
            },
            'risk_metrics': {
                'current_drawdown': (self.initial_balance - self.current_balance) / self.initial_balance,
                'daily_pnl': self.daily_pnl.get(datetime.now().date(), 0),
                'open_positions': len(self.open_positions),
                'risk_per_trade': self.max_risk_per_trade
            },
            'agent_performance': self.agent_performance,
            'market_conditions': {
                'volatility': self.current_market_condition.volatility,
                'liquidity': self.current_market_condition.liquidity,
                'session': self.current_market_condition.session
            },
            'alerts': self.risk_alerts[-10:],  # Last 10 alerts
            'trading_enabled': self.is_trading_enabled
        }

# Emergency procedures
class EmergencyRiskManager:
    """Emergency risk management procedures."""
    
    @staticmethod
    def emergency_close_all(risk_manager: MultiAgentRiskManager, reason: str):
        """Emergency procedure to close all positions."""
        logger.critical(f"EMERGENCY CLOSE ALL TRIGGERED: {reason}")
        
        for position_id in list(risk_manager.open_positions.keys()):
            # In production, this would close positions via broker API
            logger.critical(f"Emergency closing position: {position_id}")
        
        risk_manager.is_trading_enabled = False
        risk_manager.emergency_stop_triggered = True
    
    @staticmethod
    def circuit_breaker(risk_manager: MultiAgentRiskManager, trigger_value: float):
        """Circuit breaker to halt trading on excessive losses."""
        if trigger_value <= -risk_manager.initial_balance * 0.1:  # 10% loss
            EmergencyRiskManager.emergency_close_all(
                risk_manager, f"Circuit breaker triggered at {trigger_value:.2f}"
            )