# agents/weekly_analysis_engine.py

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
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketDirection(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    SLIGHTLY_BULLISH = "slightly_bullish"
    NEUTRAL = "neutral"
    SLIGHTLY_BEARISH = "slightly_bearish"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

class PredictionConfidence(Enum):
    VERY_HIGH = "very_high"  # 90%+
    HIGH = "high"           # 80-90%
    MEDIUM = "medium"       # 60-80%
    LOW = "low"            # 40-60%
    VERY_LOW = "very_low"  # <40%

@dataclass
class WeeklyAnalysisResult:
    symbol: str
    analysis_date: datetime
    past_week_performance: Dict[str, float]
    key_events_impact: List[Dict[str, Any]]
    technical_summary: Dict[str, Any]
    fundamental_summary: Dict[str, Any]
    sentiment_summary: Dict[str, Any]
    economic_summary: Dict[str, Any]
    market_regime: str  # trending, ranging, volatile, calm
    volatility_forecast: float
    direction_prediction: MarketDirection
    target_levels: Dict[str, float]
    risk_levels: Dict[str, float]
    confidence: PredictionConfidence
    probability_distribution: Dict[MarketDirection, float]
    time_horizon: str  # week
    key_factors: List[str]
    risk_warnings: List[str]

@dataclass
class MultiAgentConsensus:
    chart_analysis_weight: float
    trend_analysis_weight: float
    news_sentiment_weight: float
    economic_factors_weight: float
    final_prediction: MarketDirection
    consensus_strength: float
    conflicting_signals: List[str]
    agreement_score: float

class WeeklyAnalysisEngine:
    """
    Advanced Weekly Analysis Engine for Multi-Agent Forex Trading System.
    Performs comprehensive Sunday analysis to predict upcoming week's market behavior.
    """
    
    def __init__(self, agent_id: str = "weekly_analyzer"):
        self.agent_id = agent_id
        
        # Redis connection for inter-agent communication
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Data storage paths
        self.data_path = "data/weekly_analysis/"
        self.models_path = "models/weekly_analysis/"
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        
        # Analysis history
        self.weekly_analyses = []
        self.prediction_accuracy = []
        
        # Machine learning models for prediction
        self.prediction_models = {}
        self.feature_scalers = {}
        
        # Market regime detection
        self.regime_history = {}
        
        # Agent weights for consensus building
        self.default_agent_weights = {
            'chart_analysis': 0.30,
            'trend_identification': 0.25,
            'news_sentiment': 0.25,
            'economic_factors': 0.20
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_direction': 0,
            'direction_accuracy': 0.0,
            'avg_confidence': 0.0,
            'target_hit_rate': 0.0,
            'weekly_returns_correlation': 0.0
        }
        
        # Market analysis parameters
        self.analysis_params = {
            'lookback_weeks': 12,
            'min_confidence_threshold': 0.6,
            'volatility_threshold': 0.02,
            'trend_threshold': 0.005,
            'news_impact_threshold': 0.7
        }
        
        # Initialize models
        self._initialize_prediction_models()
        
        # Sunday schedule
        self.last_sunday_analysis = None
        
    def _initialize_prediction_models(self):
        """Initialize machine learning models for weekly predictions."""
        try:
            # Direction prediction model (classification)
            self.direction_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Volatility prediction model
            self.volatility_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            # Target level prediction model
            self.target_model = RandomForestRegressor(
                n_estimators=80,
                max_depth=8,
                random_state=42
            )
            
            # Feature scaler
            self.scaler = StandardScaler()
            
            logger.info("Prediction models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing prediction models: {str(e)}")
    
    async def perform_sunday_analysis(self, symbols: List[str] = None) -> Dict[str, WeeklyAnalysisResult]:
        """
        Perform comprehensive Sunday analysis for specified currency pairs.
        This is the main function that coordinates all analysis components.
        
        Args:
            symbols: List of currency pairs to analyze (default: major pairs)
            
        Returns:
            Dictionary of symbol -> WeeklyAnalysisResult
        """
        try:
            if symbols is None:
                symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
            
            logger.info(f"Starting Sunday analysis for {len(symbols)} pairs")
            
            results = {}
            
            for symbol in symbols:
                logger.info(f"Analyzing {symbol}...")
                
                # Gather data from all agents
                agent_data = await self._gather_agent_data(symbol)
                
                # Analyze past week performance
                past_week_data = await self._analyze_past_week(symbol)
                
                # Extract key events and their impacts
                key_events = await self._extract_key_events(symbol, agent_data)
                
                # Generate technical summary
                technical_summary = self._generate_technical_summary(agent_data.get('chart_analysis', {}))
                
                # Generate fundamental summary
                fundamental_summary = self._generate_fundamental_summary(
                    agent_data.get('economic_factors', {}),
                    agent_data.get('news_sentiment', {})
                )
                
                # Build multi-agent consensus
                consensus = await self._build_multi_agent_consensus(symbol, agent_data)
                
                # Detect market regime
                market_regime = self._detect_market_regime(symbol, past_week_data, agent_data)
                
                # Predict volatility
                volatility_forecast = await self._predict_volatility(symbol, past_week_data, agent_data)
                
                # Generate main prediction
                direction_prediction = await self._generate_direction_prediction(
                    symbol, consensus, past_week_data, agent_data
                )
                
                # Calculate target and risk levels
                target_levels = self._calculate_target_levels(
                    symbol, direction_prediction, volatility_forecast, past_week_data
                )
                risk_levels = self._calculate_risk_levels(
                    symbol, direction_prediction, volatility_forecast
                )
                
                # Calculate confidence
                confidence = self._calculate_prediction_confidence(
                    consensus, volatility_forecast, agent_data
                )
                
                # Generate probability distribution
                prob_distribution = self._generate_probability_distribution(
                    direction_prediction, confidence, consensus
                )
                
                # Identify key factors and risks
                key_factors = self._identify_key_factors(agent_data, consensus)
                risk_warnings = self._identify_risk_warnings(agent_data, volatility_forecast)
                
                # Create analysis result
                analysis = WeeklyAnalysisResult(
                    symbol=symbol,
                    analysis_date=datetime.now(),
                    past_week_performance=past_week_data,
                    key_events_impact=key_events,
                    technical_summary=technical_summary,
                    fundamental_summary=fundamental_summary,
                    sentiment_summary=agent_data.get('news_sentiment', {}),
                    economic_summary=agent_data.get('economic_factors', {}),
                    market_regime=market_regime,
                    volatility_forecast=volatility_forecast,
                    direction_prediction=direction_prediction,
                    target_levels=target_levels,
                    risk_levels=risk_levels,
                    confidence=confidence,
                    probability_distribution=prob_distribution,
                    time_horizon="week",
                    key_factors=key_factors,
                    risk_warnings=risk_warnings
                )
                
                results[symbol] = analysis
                
                # Store analysis
                self.weekly_analyses.append({
                    'symbol': symbol,
                    'date': datetime.now(),
                    'analysis': analysis
                })
                
                logger.info(f"Completed analysis for {symbol}: {direction_prediction.value} "
                          f"(confidence: {confidence.value})")
            
            # Generate cross-pair analysis
            cross_pair_insights = self._generate_cross_pair_insights(results)
            
            # Update models with latest data
            await self._update_prediction_models(results)
            
            # Save analysis results
            self._save_analysis_results(results)
            
            # Send alerts for high-confidence predictions
            await self._send_weekly_alerts(results)
            
            self.last_sunday_analysis = datetime.now()
            
            logger.info("Sunday analysis completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Sunday analysis: {str(e)}")
            return {}
    
    async def _gather_agent_data(self, symbol: str) -> Dict[str, Any]:
        """Gather the latest analysis data from all agents."""
        try:
            agent_data = {}
            
            # Request data from each agent
            agents = ['chart_analysis', 'trend_identification', 'news_sentiment', 'economic_factors']
            
            for agent in agents:
                try:
                    # Request latest analysis from agent
                    request_message = {
                        'sender': self.agent_id,
                        'receiver': agent,
                        'message_type': 'data_request',
                        'data': {'symbol': symbol, 'timeframe': 'week'},
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Publish request
                    channel = f"agent_messages_{agent}"
                    self.redis_client.publish(channel, json.dumps(request_message))
                    
                    # Wait for response (simplified - in production use proper async handling)
                    await asyncio.sleep(0.5)
                    
                    # For demo, create mock agent data
                    agent_data[agent] = self._create_mock_agent_data(agent, symbol)
                    
                except Exception as e:
                    logger.warning(f"Could not get data from {agent}: {str(e)}")
                    agent_data[agent] = {}
            
            return agent_data
            
        except Exception as e:
            logger.error(f"Error gathering agent data: {str(e)}")
            return {}
    
    def _create_mock_agent_data(self, agent: str, symbol: str) -> Dict[str, Any]:
        """Create mock agent data for demonstration."""
        base_price = 1.1000 if 'EUR' in symbol else 1.3000 if 'GBP' in symbol else 150.0 if 'JPY' in symbol else 1.0000
        
        if agent == 'chart_analysis':
            return {
                'overall_signal': np.random.choice(['buy', 'sell', 'neutral'], p=[0.3, 0.3, 0.4]),
                'confidence': np.random.uniform(0.5, 0.9),
                'support_levels': [base_price * 0.995, base_price * 0.99],
                'resistance_levels': [base_price * 1.005, base_price * 1.01],
                'trend_direction': np.random.choice(['bullish', 'bearish', 'sideways']),
                'volatility': np.random.uniform(0.008, 0.025),
                'key_indicators': {
                    'rsi': np.random.uniform(30, 70),
                    'macd': np.random.uniform(-0.002, 0.002),
                    'bollinger_position': np.random.uniform(0.2, 0.8)
                }
            }
        
        elif agent == 'trend_identification':
            return {
                'primary_trend': np.random.choice(['bullish', 'bearish', 'sideways']),
                'secondary_trend': np.random.choice(['bullish', 'bearish', 'sideways']),
                'confluence_score': np.random.uniform(0.4, 0.9),
                'trend_strength': np.random.uniform(0.3, 0.8),
                'trend_change_probability': np.random.uniform(0.1, 0.7)
            }
        
        elif agent == 'news_sentiment':
            return {
                'overall_sentiment': np.random.choice(['bullish', 'bearish', 'neutral']),
                'confidence': np.random.uniform(0.4, 0.8),
                'market_risk_level': np.random.choice(['low', 'medium', 'high']),
                'key_events_count': np.random.randint(1, 5),
                'time_sensitivity': np.random.choice(['immediate', 'hours', 'days'])
            }
        
        elif agent == 'economic_factors':
            return {
                'overall_health': np.random.choice(['positive', 'negative', 'neutral']),
                'monetary_policy_bias': np.random.choice(['hawkish', 'dovish', 'neutral']),
                'currency_outlook': {
                    '1m': np.random.uniform(-0.5, 0.5),
                    '3m': np.random.uniform(-0.3, 0.3)
                },
                'central_bank_stance': np.random.choice(['hawkish', 'dovish', 'neutral'])
            }
        
        return {}
    
    async def _analyze_past_week(self, symbol: str) -> Dict[str, float]:
        """Analyze the past week's market performance."""
        try:
            # In production, this would fetch actual market data
            # For demo, create realistic mock data
            
            start_of_week = datetime.now() - timedelta(days=7)
            
            # Mock weekly performance data
            performance = {
                'weekly_return': np.random.uniform(-0.03, 0.03),
                'weekly_volatility': np.random.uniform(0.008, 0.025),
                'max_drawdown': np.random.uniform(-0.015, -0.005),
                'max_gain': np.random.uniform(0.005, 0.02),
                'trading_days': 5,
                'avg_daily_volume': np.random.uniform(1000000, 5000000),
                'price_range': np.random.uniform(0.01, 0.03),
                'closing_momentum': np.random.uniform(-0.01, 0.01)
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error analyzing past week for {symbol}: {str(e)}")
            return {}
    
    async def _extract_key_events(self, symbol: str, agent_data: Dict) -> List[Dict[str, Any]]:
        """Extract key events that impacted the market in the past week."""
        try:
            events = []
            
            # Extract from news sentiment data
            news_data = agent_data.get('news_sentiment', {})
            if news_data.get('key_events_count', 0) > 0:
                events.append({
                    'type': 'news',
                    'impact': news_data.get('overall_sentiment', 'neutral'),
                    'importance': 'high' if news_data.get('market_risk_level') == 'high' else 'medium',
                    'description': f"News sentiment: {news_data.get('overall_sentiment', 'neutral')}"
                })
            
            # Extract from economic factors
            econ_data = agent_data.get('economic_factors', {})
            if econ_data.get('central_bank_stance') in ['hawkish', 'dovish']:
                events.append({
                    'type': 'economic',
                    'impact': econ_data.get('central_bank_stance'),
                    'importance': 'critical',
                    'description': f"Central bank stance: {econ_data.get('central_bank_stance')}"
                })
            
            # Extract from technical analysis
            tech_data = agent_data.get('chart_analysis', {})
            if tech_data.get('confidence', 0) > 0.8:
                events.append({
                    'type': 'technical',
                    'impact': tech_data.get('overall_signal', 'neutral'),
                    'importance': 'high',
                    'description': f"Strong technical signal: {tech_data.get('overall_signal')}"
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error extracting key events: {str(e)}")
            return []
    
    def _generate_technical_summary(self, chart_data: Dict) -> Dict[str, Any]:
        """Generate technical analysis summary."""
        if not chart_data:
            return {'status': 'no_data'}
        
        return {
            'overall_signal': chart_data.get('overall_signal', 'neutral'),
            'trend_direction': chart_data.get('trend_direction', 'sideways'),
            'support_resistance': {
                'support': chart_data.get('support_levels', []),
                'resistance': chart_data.get('resistance_levels', [])
            },
            'momentum': {
                'rsi': chart_data.get('key_indicators', {}).get('rsi', 50),
                'macd': chart_data.get('key_indicators', {}).get('macd', 0)
            },
            'volatility': chart_data.get('volatility', 0.015),
            'confidence': chart_data.get('confidence', 0.5)
        }
    
    def _generate_fundamental_summary(self, econ_data: Dict, news_data: Dict) -> Dict[str, Any]:
        """Generate fundamental analysis summary."""
        return {
            'economic_health': econ_data.get('overall_health', 'neutral'),
            'monetary_policy': econ_data.get('monetary_policy_bias', 'neutral'),
            'news_sentiment': news_data.get('overall_sentiment', 'neutral'),
            'market_risk': news_data.get('market_risk_level', 'medium'),
            'outlook': econ_data.get('currency_outlook', {}),
            'central_bank_stance': econ_data.get('central_bank_stance', 'neutral')
        }
    
    async def _build_multi_agent_consensus(self, symbol: str, agent_data: Dict) -> MultiAgentConsensus:
        """Build consensus from multiple agent analyses."""
        try:
            # Extract signals from each agent
            signals = {}
            confidences = {}
            
            # Chart analysis signal
            chart_data = agent_data.get('chart_analysis', {})
            signals['chart'] = self._normalize_signal(chart_data.get('overall_signal', 'neutral'))
            confidences['chart'] = chart_data.get('confidence', 0.5)
            
            # Trend analysis signal
            trend_data = agent_data.get('trend_identification', {})
            signals['trend'] = self._normalize_signal(trend_data.get('primary_trend', 'neutral'))
            confidences['trend'] = trend_data.get('confluence_score', 0.5)
            
            # News sentiment signal
            news_data = agent_data.get('news_sentiment', {})
            signals['news'] = self._normalize_signal(news_data.get('overall_sentiment', 'neutral'))
            confidences['news'] = news_data.get('confidence', 0.5)
            
            # Economic factors signal
            econ_data = agent_data.get('economic_factors', {})
            econ_signal = 'bullish' if econ_data.get('currency_outlook', {}).get('1m', 0) > 0.2 else \
                         'bearish' if econ_data.get('currency_outlook', {}).get('1m', 0) < -0.2 else 'neutral'
            signals['economic'] = self._normalize_signal(econ_signal)
            confidences['economic'] = 0.7  # Default confidence for economic analysis
            
            # Calculate weighted consensus
            weights = self.default_agent_weights
            
            # Adjust weights based on confidence
            for agent in weights:
                agent_key = agent.split('_')[0]
                if agent_key in confidences:
                    weights[agent] *= (0.5 + confidences[agent_key] * 0.5)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            # Calculate consensus signal
            consensus_signal = 0
            for agent, weight in weights.items():
                agent_key = agent.split('_')[0]
                if agent_key in signals:
                    consensus_signal += signals[agent_key] * weight
            
            # Convert to direction
            if consensus_signal > 0.3:
                final_prediction = MarketDirection.BULLISH
            elif consensus_signal > 0.1:
                final_prediction = MarketDirection.SLIGHTLY_BULLISH
            elif consensus_signal > -0.1:
                final_prediction = MarketDirection.NEUTRAL
            elif consensus_signal > -0.3:
                final_prediction = MarketDirection.SLIGHTLY_BEARISH
            else:
                final_prediction = MarketDirection.BEARISH
            
            # Calculate consensus strength
            signal_variance = np.var(list(signals.values()))
            consensus_strength = max(0, 1 - signal_variance)
            
            # Identify conflicting signals
            conflicting_signals = []
            if signal_variance > 0.5:
                conflicting_signals.append("High variance between agent signals")
            
            # Calculate agreement score
            agreement_score = np.mean(list(confidences.values())) * consensus_strength
            
            return MultiAgentConsensus(
                chart_analysis_weight=weights.get('chart_analysis', 0),
                trend_analysis_weight=weights.get('trend_identification', 0),
                news_sentiment_weight=weights.get('news_sentiment', 0),
                economic_factors_weight=weights.get('economic_factors', 0),
                final_prediction=final_prediction,
                consensus_strength=consensus_strength,
                conflicting_signals=conflicting_signals,
                agreement_score=agreement_score
            )
            
        except Exception as e:
            logger.error(f"Error building consensus: {str(e)}")
            return MultiAgentConsensus(
                chart_analysis_weight=0.25, trend_analysis_weight=0.25,
                news_sentiment_weight=0.25, economic_factors_weight=0.25,
                final_prediction=MarketDirection.NEUTRAL, consensus_strength=0.5,
                conflicting_signals=[], agreement_score=0.5
            )
    
    def _normalize_signal(self, signal: str) -> float:
        """Convert signal string to numerical value."""
        signal = signal.lower()
        if signal in ['bullish', 'buy', 'positive', 'hawkish']:
            return 1.0
        elif signal in ['bearish', 'sell', 'negative', 'dovish']:
            return -1.0
        elif signal in ['slightly_bullish', 'slightly_positive']:
            return 0.5
        elif signal in ['slightly_bearish', 'slightly_negative']:
            return -0.5
        else:
            return 0.0
    
    def _detect_market_regime(self, symbol: str, past_week_data: Dict, agent_data: Dict) -> str:
        """Detect current market regime."""
        try:
            volatility = past_week_data.get('weekly_volatility', 0.015)
            weekly_return = abs(past_week_data.get('weekly_return', 0))
            
            # High volatility threshold
            if volatility > 0.02:
                if weekly_return > 0.015:
                    return "volatile_trending"
                else:
                    return "volatile_ranging"
            else:
                if weekly_return > 0.01:
                    return "calm_trending"
                else:
                    return "calm_ranging"
                    
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"
    
    async def _predict_volatility(self, symbol: str, past_week_data: Dict, agent_data: Dict) -> float:
        """Predict volatility for the upcoming week."""
        try:
            # Use historical volatility as base
            historical_vol = past_week_data.get('weekly_volatility', 0.015)
            
            # Adjust based on upcoming events
            vol_adjustment = 1.0
            
            # News impact
            news_data = agent_data.get('news_sentiment', {})
            if news_data.get('market_risk_level') == 'high':
                vol_adjustment *= 1.3
            elif news_data.get('time_sensitivity') == 'immediate':
                vol_adjustment *= 1.2
            
            # Economic events
            econ_data = agent_data.get('economic_factors', {})
            if econ_data.get('central_bank_stance') in ['hawkish', 'dovish']:
                vol_adjustment *= 1.15
            
            # Technical volatility
            chart_data = agent_data.get('chart_analysis', {})
            chart_vol = chart_data.get('volatility', historical_vol)
            
            # Combine factors
            predicted_vol = (historical_vol * 0.6 + chart_vol * 0.4) * vol_adjustment
            
            return min(max(predicted_vol, 0.005), 0.05)  # Bound between 0.5% and 5%
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {str(e)}")
            return 0.015
    
    async def _generate_direction_prediction(self, symbol: str, consensus: MultiAgentConsensus, 
                                           past_week_data: Dict, agent_data: Dict) -> MarketDirection:
        """Generate final direction prediction."""
        try:
            # Start with consensus prediction
            base_prediction = consensus.final_prediction
            
            # Adjust based on momentum
            momentum = past_week_data.get('closing_momentum', 0)
            
            if momentum > 0.005 and base_prediction in [MarketDirection.NEUTRAL, MarketDirection.SLIGHTLY_BULLISH]:
                base_prediction = MarketDirection.BULLISH
            elif momentum < -0.005 and base_prediction in [MarketDirection.NEUTRAL, MarketDirection.SLIGHTLY_BEARISH]:
                base_prediction = MarketDirection.BEARISH
            
            # Consider market regime
            regime = self._detect_market_regime(symbol, past_week_data, agent_data)
            if 'ranging' in regime and base_prediction in [MarketDirection.BULLISH, MarketDirection.BEARISH]:
                # Moderate prediction in ranging markets
                if base_prediction == MarketDirection.BULLISH:
                    base_prediction = MarketDirection.SLIGHTLY_BULLISH
                else:
                    base_prediction = MarketDirection.SLIGHTLY_BEARISH
            
            return base_prediction
            
        except Exception as e:
            logger.error(f"Error generating direction prediction: {str(e)}")
            return MarketDirection.NEUTRAL
    
    def _calculate_target_levels(self, symbol: str, direction: MarketDirection, 
                               volatility: float, past_week_data: Dict) -> Dict[str, float]:
        """Calculate target price levels."""
        try:
            # Mock current price based on symbol
            current_price = 1.1000 if 'EUR' in symbol else 1.3000 if 'GBP' in symbol else 150.0 if 'JPY' in symbol else 1.0000
            
            # Base target calculation on volatility
            weekly_range = volatility * 2.5  # Expected weekly range
            
            targets = {}
            
            if direction in [MarketDirection.BULLISH, MarketDirection.STRONG_BULLISH]:
                targets['target_1'] = current_price * (1 + weekly_range * 0.5)
                targets['target_2'] = current_price * (1 + weekly_range * 0.8)
                targets['target_3'] = current_price * (1 + weekly_range * 1.2)
            elif direction in [MarketDirection.BEARISH, MarketDirection.STRONG_BEARISH]:
                targets['target_1'] = current_price * (1 - weekly_range * 0.5)
                targets['target_2'] = current_price * (1 - weekly_range * 0.8)
                targets['target_3'] = current_price * (1 - weekly_range * 1.2)
            else:
                # Neutral - range targets
                targets['upper_range'] = current_price * (1 + weekly_range * 0.4)
                targets['lower_range'] = current_price * (1 - weekly_range * 0.4)
                targets['midpoint'] = current_price
            
            return targets
            
        except Exception as e:
            logger.error(f"Error calculating target levels: {str(e)}")
            return {}
    
    def _calculate_risk_levels(self, symbol: str, direction: MarketDirection, 
                             volatility: float) -> Dict[str, float]:
        """Calculate risk management levels."""
        try:
            # Mock current price
            current_price = 1.1000 if 'EUR' in symbol else 1.3000 if 'GBP' in symbol else 150.0 if 'JPY' in symbol else 1.0000
            
            # Risk levels based on volatility
            daily_atr = volatility / 5  # Approximate daily ATR
            
            risks = {}
            
            if direction in [MarketDirection.BULLISH, MarketDirection.STRONG_BULLISH]:
                risks['stop_loss'] = current_price * (1 - daily_atr * 2)
                risks['caution_level'] = current_price * (1 - daily_atr * 1)
            elif direction in [MarketDirection.BEARISH, MarketDirection.STRONG_BEARISH]:
                risks['stop_loss'] = current_price * (1 + daily_atr * 2)
                risks['caution_level'] = current_price * (1 + daily_atr * 1)
            else:
                risks['upper_risk'] = current_price * (1 + daily_atr * 1.5)
                risks['lower_risk'] = current_price * (1 - daily_atr * 1.5)
            
            return risks
            
        except Exception as e:
            logger.error(f"Error calculating risk levels: {str(e)}")
            return {}
    
    def _calculate_prediction_confidence(self, consensus: MultiAgentConsensus, 
                                       volatility: float, agent_data: Dict) -> PredictionConfidence:
        """Calculate overall prediction confidence."""
        try:
            # Base confidence from consensus
            base_confidence = consensus.agreement_score
            
            # Adjust for volatility (higher vol = lower confidence)
            vol_adjustment = max(0.5, 1 - (volatility - 0.01) * 10)
            
            # Adjust for conflicting signals
            conflict_adjustment = 1.0
            if consensus.conflicting_signals:
                conflict_adjustment = 0.8
            
            # Adjust for data quality
            data_quality = sum(1 for data in agent_data.values() if data) / 4  # 4 agents
            
            final_confidence = base_confidence * vol_adjustment * conflict_adjustment * data_quality
            
            if final_confidence >= 0.9:
                return PredictionConfidence.VERY_HIGH
            elif final_confidence >= 0.8:
                return PredictionConfidence.HIGH
            elif final_confidence >= 0.6:
                return PredictionConfidence.MEDIUM
            elif final_confidence >= 0.4:
                return PredictionConfidence.LOW
            else:
                return PredictionConfidence.VERY_LOW
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return PredictionConfidence.MEDIUM
    
    def _generate_probability_distribution(self, direction: MarketDirection, 
                                         confidence: PredictionConfidence, 
                                         consensus: MultiAgentConsensus) -> Dict[MarketDirection, float]:
        """Generate probability distribution for all possible outcomes."""
        try:
            # Base probabilities
            probs = {d: 0.0 for d in MarketDirection}
            
            # Confidence multiplier
            conf_multipliers = {
                PredictionConfidence.VERY_HIGH: 0.9,
                PredictionConfidence.HIGH: 0.8,
                PredictionConfidence.MEDIUM: 0.6,
                PredictionConfidence.LOW: 0.4,
                PredictionConfidence.VERY_LOW: 0.3
            }
            
            main_prob = conf_multipliers.get(confidence, 0.5)
            
            # Assign main probability to predicted direction
            probs[direction] = main_prob
            
            # Distribute remaining probability
            remaining_prob = 1.0 - main_prob
            other_directions = [d for d in MarketDirection if d != direction]
            
            # Higher probability to adjacent directions
            if direction == MarketDirection.BULLISH:
                probs[MarketDirection.SLIGHTLY_BULLISH] = remaining_prob * 0.4
                probs[MarketDirection.NEUTRAL] = remaining_prob * 0.3
                probs[MarketDirection.SLIGHTLY_BEARISH] = remaining_prob * 0.2
                probs[MarketDirection.BEARISH] = remaining_prob * 0.1
            elif direction == MarketDirection.BEARISH:
                probs[MarketDirection.SLIGHTLY_BEARISH] = remaining_prob * 0.4
                probs[MarketDirection.NEUTRAL] = remaining_prob * 0.3
                probs[MarketDirection.SLIGHTLY_BULLISH] = remaining_prob * 0.2
                probs[MarketDirection.BULLISH] = remaining_prob * 0.1
            else:
                # Distribute evenly for neutral predictions
                prob_per_direction = remaining_prob / len(other_directions)
                for d in other_directions:
                    probs[d] = prob_per_direction
            
            return probs
            
        except Exception as e:
            logger.error(f"Error generating probability distribution: {str(e)}")
            return {d: 1.0/len(MarketDirection) for d in MarketDirection}
    
    def _identify_key_factors(self, agent_data: Dict, consensus: MultiAgentConsensus) -> List[str]:
        """Identify key factors driving the prediction."""
        factors = []
        
        try:
            # Technical factors
            chart_data = agent_data.get('chart_analysis', {})
            if chart_data.get('confidence', 0) > 0.7:
                factors.append(f"Strong technical signal: {chart_data.get('overall_signal', 'neutral')}")
            
            # Trend factors
            trend_data = agent_data.get('trend_identification', {})
            if trend_data.get('confluence_score', 0) > 0.8:
                factors.append(f"Multi-timeframe trend confluence: {trend_data.get('primary_trend', 'neutral')}")
            
            # Fundamental factors
            econ_data = agent_data.get('economic_factors', {})
            if econ_data.get('central_bank_stance') in ['hawkish', 'dovish']:
                factors.append(f"Central bank policy stance: {econ_data.get('central_bank_stance')}")
            
            # News factors
            news_data = agent_data.get('news_sentiment', {})
            if news_data.get('market_risk_level') == 'high':
                factors.append(f"High-impact news sentiment: {news_data.get('overall_sentiment', 'neutral')}")
            
            # Consensus factor
            if consensus.agreement_score > 0.8:
                factors.append("Strong multi-agent consensus")
            
            return factors[:5]  # Top 5 factors
            
        except Exception as e:
            logger.error(f"Error identifying key factors: {str(e)}")
            return ["Analysis factors could not be determined"]
    
    def _identify_risk_warnings(self, agent_data: Dict, volatility: float) -> List[str]:
        """Identify potential risk warnings."""
        warnings = []
        
        try:
            # High volatility warning
            if volatility > 0.025:
                warnings.append("High volatility expected - use smaller position sizes")
            
            # Conflicting signals warning
            signals = []
            for agent, data in agent_data.items():
                if 'signal' in data or 'sentiment' in data:
                    signals.append(data.get('overall_signal', data.get('overall_sentiment', 'neutral')))
            
            if len(set(signals)) > 2:
                warnings.append("Conflicting signals between analysis methods")
            
            # News risk warning
            news_data = agent_data.get('news_sentiment', {})
            if news_data.get('time_sensitivity') == 'immediate':
                warnings.append("Immediate market-moving events expected")
            
            # Economic risk warning
            econ_data = agent_data.get('economic_factors', {})
            if econ_data.get('central_bank_stance') in ['hawkish', 'dovish']:
                warnings.append("Central bank policy changes may cause sudden moves")
            
            return warnings[:3]  # Top 3 warnings
            
        except Exception as e:
            logger.error(f"Error identifying risk warnings: {str(e)}")
            return ["Standard market risks apply"]
    
    def _generate_cross_pair_insights(self, results: Dict[str, WeeklyAnalysisResult]) -> Dict[str, Any]:
        """Generate insights across multiple currency pairs."""
        try:
            insights = {
                'usd_strength': 0.0,
                'eur_strength': 0.0,
                'gbp_strength': 0.0,
                'jpy_strength': 0.0,
                'overall_risk_level': 'medium',
                'correlation_warnings': []
            }
            
            # Calculate currency strength
            currency_scores = {'USD': [], 'EUR': [], 'GBP': [], 'JPY': []}
            
            for symbol, analysis in results.items():
                if len(symbol) == 6:
                    base_curr = symbol[:3]
                    quote_curr = symbol[3:6]
                    
                    direction_score = self._direction_to_score(analysis.direction_prediction)
                    
                    if base_curr in currency_scores:
                        currency_scores[base_curr].append(direction_score)
                    if quote_curr in currency_scores:
                        currency_scores[quote_curr].append(-direction_score)  # Inverse for quote currency
            
            # Calculate average strength
            for currency, scores in currency_scores.items():
                if scores:
                    avg_score = np.mean(scores)
                    insights[f'{currency.lower()}_strength'] = avg_score
            
            # Assess overall risk
            high_vol_count = sum(1 for analysis in results.values() if analysis.volatility_forecast > 0.02)
            if high_vol_count > len(results) * 0.6:
                insights['overall_risk_level'] = 'high'
            elif high_vol_count < len(results) * 0.3:
                insights['overall_risk_level'] = 'low'
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating cross-pair insights: {str(e)}")
            return {}
    
    def _direction_to_score(self, direction: MarketDirection) -> float:
        """Convert market direction to numerical score."""
        scores = {
            MarketDirection.STRONG_BULLISH: 1.0,
            MarketDirection.BULLISH: 0.6,
            MarketDirection.SLIGHTLY_BULLISH: 0.3,
            MarketDirection.NEUTRAL: 0.0,
            MarketDirection.SLIGHTLY_BEARISH: -0.3,
            MarketDirection.BEARISH: -0.6,
            MarketDirection.STRONG_BEARISH: -1.0
        }
        return scores.get(direction, 0.0)
    
    async def _update_prediction_models(self, results: Dict[str, WeeklyAnalysisResult]):
        """Update ML models with latest analysis results."""
        try:
            # This would train/update models with new data
            # For now, just log the update
            logger.info(f"Updated prediction models with {len(results)} new analyses")
            
        except Exception as e:
            logger.error(f"Error updating prediction models: {str(e)}")
    
    def _save_analysis_results(self, results: Dict[str, WeeklyAnalysisResult]):
        """Save analysis results to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_path}/weekly_analysis_{timestamp}.json"
            
            # Convert to serializable format
            serializable_results = {}
            for symbol, analysis in results.items():
                serializable_results[symbol] = {
                    'symbol': analysis.symbol,
                    'analysis_date': analysis.analysis_date.isoformat(),
                    'direction_prediction': analysis.direction_prediction.value,
                    'confidence': analysis.confidence.value,
                    'volatility_forecast': analysis.volatility_forecast,
                    'target_levels': analysis.target_levels,
                    'risk_levels': analysis.risk_levels,
                    'key_factors': analysis.key_factors,
                    'risk_warnings': analysis.risk_warnings
                }
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Analysis results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
    
    async def _send_weekly_alerts(self, results: Dict[str, WeeklyAnalysisResult]):
        """Send alerts for high-confidence predictions."""
        try:
            for symbol, analysis in results.items():
                if analysis.confidence in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]:
                    alert_message = {
                        'sender': self.agent_id,
                        'receiver': 'risk_manager',
                        'message_type': 'weekly_prediction_alert',
                        'data': {
                            'symbol': symbol,
                            'direction': analysis.direction_prediction.value,
                            'confidence': analysis.confidence.value,
                            'target_levels': analysis.target_levels,
                            'risk_levels': analysis.risk_levels,
                            'key_factors': analysis.key_factors
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Send alert
                    channel = "agent_messages_risk_manager"
                    self.redis_client.publish(channel, json.dumps(alert_message, default=str))
                    
                    logger.info(f"High-confidence alert sent for {symbol}: {analysis.direction_prediction.value}")
            
        except Exception as e:
            logger.error(f"Error sending weekly alerts: {str(e)}")
    
    def should_run_analysis(self) -> bool:
        """Check if Sunday analysis should be run."""
        now = datetime.now()
        
        # Run on Sundays
        if now.weekday() != 6:  # 6 = Sunday
            return False
        
        # Don't run more than once per day
        if self.last_sunday_analysis and \
           (now - self.last_sunday_analysis).total_seconds() < 86400:  # 24 hours
            return False
        
        return True
    
    def get_latest_analysis(self, symbol: str = None) -> Optional[WeeklyAnalysisResult]:
        """Get the latest analysis for a symbol or the most recent overall."""
        try:
            if not self.weekly_analyses:
                return None
            
            if symbol:
                # Find latest analysis for specific symbol
                for analysis in reversed(self.weekly_analyses):
                    if analysis['symbol'] == symbol:
                        return analysis['analysis']
                return None
            else:
                # Return most recent analysis
                return self.weekly_analyses[-1]['analysis']
                
        except Exception as e:
            logger.error(f"Error getting latest analysis: {str(e)}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'agent_id': self.agent_id,
            'performance_metrics': self.performance_metrics,
            'total_analyses': len(self.weekly_analyses),
            'last_analysis': self.last_sunday_analysis.isoformat() if self.last_sunday_analysis else None,
            'analysis_params': self.analysis_params,
            'agent_weights': self.default_agent_weights
        }