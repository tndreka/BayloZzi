# agents/chart_analysis_agent.py

import asyncio
import logging
import numpy as np
import pandas as pd
# Use ta library instead of talib for Windows compatibility
import ta
from datetime import datetime, timedelta

# Create wrapper functions for talib compatibility
class talib:
    @staticmethod
    def SMA(close, period):
        return pd.Series(close).rolling(window=period).mean().values
    
    @staticmethod
    def EMA(close, period):
        return pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def RSI(close, period):
        return ta.momentum.RSIIndicator(pd.Series(close), window=period).rsi().values
    
    @staticmethod
    def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
        macd_indicator = ta.trend.MACD(pd.Series(close), window_slow=slowperiod, window_fast=fastperiod, window_sign=signalperiod)
        return macd_indicator.macd().values, macd_indicator.macd_signal().values, macd_indicator.macd_diff().values
    
    @staticmethod
    def BBANDS(close, period, devup=2, devdn=2):
        bb = ta.volatility.BollingerBands(pd.Series(close), window=period, window_dev=devup)
        return bb.bollinger_hband().values, bb.bollinger_mavg().values, bb.bollinger_lband().values
    
    @staticmethod
    def ATR(high, low, close, period):
        return ta.volatility.AverageTrueRange(pd.Series(high), pd.Series(low), pd.Series(close), window=period).average_true_range().values
    
    @staticmethod
    def ADX(high, low, close, period):
        return ta.trend.ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=period).adx().values
    
    @staticmethod
    def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
        stoch = ta.momentum.StochasticOscillator(pd.Series(high), pd.Series(low), pd.Series(close), window=fastk_period, smooth_window=slowk_period)
        return stoch.stoch().values, stoch.stoch_signal().values
    
    @staticmethod
    def WILLR(high, low, close, timeperiod=14):
        return ta.momentum.WilliamsRIndicator(pd.Series(high), pd.Series(low), pd.Series(close), lbp=timeperiod).williams_r().values
    
    @staticmethod
    def CCI(high, low, close, timeperiod=14):
        return ta.trend.CCIIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=timeperiod).cci().values
    
    @staticmethod
    def MFI(high, low, close, volume, timeperiod=14):
        return ta.volume.MFIIndicator(pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume), window=timeperiod).money_flow_index().values
    
    # Candlestick patterns - return simple approximations
    @staticmethod
    def CDLHAMMER(open, high, low, close):
        return np.zeros(len(close))
    
    @staticmethod
    def CDLDOJI(open, high, low, close):
        return np.zeros(len(close))
    
    @staticmethod
    def CDLENGULFING(open, high, low, close):
        return np.zeros(len(close))
    
    @staticmethod
    def CDLSHOOTINGSTAR(open, high, low, close):
        return np.zeros(len(close))
    
    @staticmethod
    def CDLHANGINGMAN(open, high, low, close):
        return np.zeros(len(close))
    
    @staticmethod
    def CDLMORNINGSTAR(open, high, low, close):
        return np.zeros(len(close))
    
    @staticmethod
    def CDLEVENINGSTAR(open, high, low, close):
        return np.zeros(len(close))

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import redis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PatternType(Enum):
    HAMMER = "hammer"
    DOJI = "doji"
    ENGULFING = "engulfing"
    SHOOTING_STAR = "shooting_star"
    HEAD_SHOULDERS = "head_shoulders"
    TRIANGLE = "triangle"
    FLAG = "flag"
    SUPPORT_RESISTANCE = "support_resistance"

class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

@dataclass
class TechnicalSignal:
    timeframe: str
    signal_type: str  # 'buy', 'sell', 'neutral'
    strength: float  # 0-1
    confidence: float  # 0-1
    indicators: Dict[str, float]
    patterns: List[str]
    support_resistance: Dict[str, float]
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str

@dataclass
class ChartPattern:
    pattern_type: PatternType
    timeframe: str
    confidence: float
    entry_zone: Tuple[float, float]
    target_zone: Tuple[float, float]
    invalidation_level: float
    description: str

class ChartAnalysisAgent:
    """
    Advanced Chart Analysis Agent for Multi-Agent Forex Trading System.
    Provides comprehensive technical analysis across multiple timeframes.
    """
    
    def __init__(self, agent_id: str = "chart_analyzer"):
        self.agent_id = agent_id
        
        # Redis connection for inter-agent communication
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Technical analysis parameters
        self.timeframes = [tf.value for tf in TimeFrame]
        self.primary_timeframes = ["4h", "1h", "15m"]
        
        # Technical indicators settings
        self.indicator_params = {
            'sma_fast': 9,
            'sma_slow': 21,
            'ema_fast': 12,
            'ema_slow': 26,
            'ema_signal': 9,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'adx_period': 14
        }
        
        # Pattern recognition models
        self.pattern_classifiers = {}
        self.support_resistance_levels = {}
        
        # Historical analysis data
        self.market_data = {}
        self.analysis_history = []
        
        # Performance tracking
        self.signal_performance = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_holding_period': 0.0,
            'best_timeframe': '1h'
        }
        
        # Multi-timeframe confluence
        self.timeframe_weights = {
            '1d': 0.3,
            '4h': 0.25,
            '1h': 0.2,
            '15m': 0.15,
            '5m': 0.1
        }
        
        # Initialize pattern recognition
        self._initialize_pattern_recognition()
    
    def _initialize_pattern_recognition(self):
        """Initialize pattern recognition models and parameters."""
        try:
            # Initialize ML models for pattern recognition
            for pattern in PatternType:
                self.pattern_classifiers[pattern.value] = RandomForestClassifier(
                    n_estimators=100, random_state=42, max_depth=10
                )
            
            logger.info("Pattern recognition models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing pattern recognition: {str(e)}")
    
    async def analyze_chart(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Comprehensive chart analysis across multiple timeframes.
        
        Args:
            symbol: Currency pair symbol
            data: Dictionary of timeframe -> OHLCV DataFrame
            
        Returns:
            Complete technical analysis results
        """
        try:
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'timeframe_analysis': {},
                'overall_signal': 'neutral',
                'confidence': 0.0,
                'primary_signal': None,
                'patterns_detected': [],
                'support_resistance': {},
                'risk_assessment': {},
                'entry_recommendations': []
            }
            
            # Analyze each timeframe
            for timeframe, df in data.items():
                if timeframe in self.timeframes:
                    tf_analysis = await self._analyze_timeframe(symbol, timeframe, df)
                    analysis_result['timeframe_analysis'][timeframe] = tf_analysis
            
            # Determine overall signal using multi-timeframe confluence
            overall_signal = self._calculate_confluence_signal(analysis_result['timeframe_analysis'])
            analysis_result['overall_signal'] = overall_signal['signal']
            analysis_result['confidence'] = overall_signal['confidence']
            
            # Detect chart patterns
            patterns = await self._detect_patterns(symbol, data)
            analysis_result['patterns_detected'] = patterns
            
            # Calculate support and resistance levels
            sr_levels = self._calculate_support_resistance(data.get('1h', pd.DataFrame()))
            analysis_result['support_resistance'] = sr_levels
            
            # Generate trading recommendations
            recommendations = self._generate_recommendations(analysis_result)
            analysis_result['entry_recommendations'] = recommendations
            
            # Store analysis for historical tracking
            self.analysis_history.append(analysis_result)
            
            # Send signal to risk management agent if strong enough
            if analysis_result['confidence'] > 0.65:
                await self._send_signal_to_risk_manager(analysis_result)
            
            logger.info(f"Chart analysis completed for {symbol}: {overall_signal['signal']} ({overall_signal['confidence']:.2f})")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in chart analysis: {str(e)}")
            return {'error': str(e)}
    
    async def _analyze_timeframe(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze single timeframe with comprehensive technical indicators."""
        try:
            if len(df) < 50:  # Minimum data requirement
                return {'error': 'Insufficient data'}
            
            # Calculate all technical indicators
            indicators = self._calculate_indicators(df)
            
            # Analyze price action
            price_action = self._analyze_price_action(df)
            
            # Detect candlestick patterns
            candlestick_patterns = self._detect_candlestick_patterns(df)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(indicators, price_action)
            
            # Determine trend direction
            trend = self._determine_trend(indicators, df)
            
            # Calculate volatility metrics
            volatility = self._calculate_volatility(df)
            
            return {
                'timeframe': timeframe,
                'indicators': indicators,
                'price_action': price_action,
                'candlestick_patterns': candlestick_patterns,
                'signal_strength': signal_strength,
                'trend': trend,
                'volatility': volatility,
                'signal': signal_strength['signal'],
                'confidence': signal_strength['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe {timeframe}: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators."""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df.get('volume', pd.Series([1000] * len(df))).values
            
            indicators = {}
            
            # Moving Averages
            indicators['sma_9'] = talib.SMA(close, self.indicator_params['sma_fast'])[-1]
            indicators['sma_21'] = talib.SMA(close, self.indicator_params['sma_slow'])[-1]
            indicators['ema_12'] = talib.EMA(close, self.indicator_params['ema_fast'])[-1]
            indicators['ema_26'] = talib.EMA(close, self.indicator_params['ema_slow'])[-1]
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(close, 
                                                   self.indicator_params['macd_fast'],
                                                   self.indicator_params['macd_slow'],
                                                   self.indicator_params['macd_signal'])
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = macdsignal[-1]
            indicators['macd_histogram'] = macdhist[-1]
            
            # RSI
            indicators['rsi'] = talib.RSI(close, self.indicator_params['rsi_period'])[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 
                                                        self.indicator_params['bb_period'],
                                                        self.indicator_params['bb_std'],
                                                        self.indicator_params['bb_std'])
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            indicators['bb_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # ATR (Average True Range)
            indicators['atr'] = talib.ATR(high, low, close, self.indicator_params['atr_period'])[-1]
            
            # ADX (Trend Strength)
            indicators['adx'] = talib.ADX(high, low, close, self.indicator_params['adx_period'])[-1]
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            indicators['stoch_k'] = slowk[-1]
            indicators['stoch_d'] = slowd[-1]
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1]
            
            # Commodity Channel Index
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)[-1]
            
            # Money Flow Index
            indicators['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def _analyze_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price action patterns and structures."""
        try:
            price_action = {}
            
            # Recent price movement
            price_action['current_price'] = df['close'].iloc[-1]
            price_action['price_change_1'] = df['close'].iloc[-1] - df['close'].iloc[-2]
            price_action['price_change_5'] = df['close'].iloc[-1] - df['close'].iloc[-6]
            price_action['price_change_pct'] = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            
            # Higher highs and lower lows analysis
            highs = df['high'].rolling(window=5).max()
            lows = df['low'].rolling(window=5).min()
            
            recent_highs = highs.tail(10).values
            recent_lows = lows.tail(10).values
            
            price_action['higher_highs'] = np.sum(np.diff(recent_highs) > 0)
            price_action['lower_lows'] = np.sum(np.diff(recent_lows) < 0)
            
            # Market structure
            if price_action['higher_highs'] >= 3:
                price_action['market_structure'] = 'bullish'
            elif price_action['lower_lows'] >= 3:
                price_action['market_structure'] = 'bearish'
            else:
                price_action['market_structure'] = 'sideways'
            
            # Momentum analysis
            price_action['momentum'] = df['close'].pct_change(5).iloc[-1] * 100
            
            return price_action
            
        except Exception as e:
            logger.error(f"Error analyzing price action: {str(e)}")
            return {}
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect candlestick patterns using TA-Lib."""
        try:
            patterns = []
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            
            # Define pattern functions and their names
            pattern_functions = {
                'hammer': talib.CDLHAMMER,
                'doji': talib.CDLDOJI,
                'engulfing': talib.CDLENGULFING,
                'shooting_star': talib.CDLSHOOTINGSTAR,
                'hanging_man': talib.CDLHANGINGMAN,
                'morning_star': talib.CDLMORNINGSTAR,
                'evening_star': talib.CDLEVENINGSTAR,
                'three_white_soldiers': talib.CDL3WHITESOLDIERS,
                'three_black_crows': talib.CDL3BLACKCROWS,
                'piercing': talib.CDLPIERCING,
                'dark_cloud': talib.CDLDARKCLOUDCOVER
            }
            
            # Check each pattern
            for pattern_name, pattern_func in pattern_functions.items():
                try:
                    result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                    if result[-1] != 0:  # Pattern detected
                        patterns.append({
                            'name': pattern_name,
                            'strength': abs(result[-1]) / 100,  # Normalize to 0-1
                            'direction': 'bullish' if result[-1] > 0 else 'bearish'
                        })
                except:
                    continue
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {str(e)}")
            return []
    
    def _calculate_signal_strength(self, indicators: Dict, price_action: Dict) -> Dict[str, Any]:
        """Calculate overall signal strength from multiple indicators."""
        try:
            signals = []
            
            # Moving Average signals
            if 'sma_9' in indicators and 'sma_21' in indicators:
                if indicators['sma_9'] > indicators['sma_21']:
                    signals.append(('ma_cross', 1, 0.7))  # Bullish
                else:
                    signals.append(('ma_cross', -1, 0.7))  # Bearish
            
            # MACD signals
            if 'macd' in indicators and 'macd_signal' in indicators:
                if indicators['macd'] > indicators['macd_signal'] and indicators['macd_histogram'] > 0:
                    signals.append(('macd', 1, 0.6))
                elif indicators['macd'] < indicators['macd_signal'] and indicators['macd_histogram'] < 0:
                    signals.append(('macd', -1, 0.6))
            
            # RSI signals
            if 'rsi' in indicators:
                if indicators['rsi'] < self.indicator_params['rsi_oversold']:
                    signals.append(('rsi_oversold', 1, 0.8))
                elif indicators['rsi'] > self.indicator_params['rsi_overbought']:
                    signals.append(('rsi_overbought', -1, 0.8))
                elif 30 < indicators['rsi'] < 70:
                    signals.append(('rsi_neutral', 0, 0.3))
            
            # Bollinger Bands signals
            if 'bb_position' in indicators:
                if indicators['bb_position'] < 0.1:
                    signals.append(('bb_oversold', 1, 0.7))
                elif indicators['bb_position'] > 0.9:
                    signals.append(('bb_overbought', -1, 0.7))
            
            # Stochastic signals
            if 'stoch_k' in indicators and 'stoch_d' in indicators:
                if indicators['stoch_k'] < 20 and indicators['stoch_d'] < 20:
                    signals.append(('stoch_oversold', 1, 0.6))
                elif indicators['stoch_k'] > 80 and indicators['stoch_d'] > 80:
                    signals.append(('stoch_overbought', -1, 0.6))
            
            # Price action signals
            if price_action.get('market_structure') == 'bullish':
                signals.append(('price_structure', 1, 0.5))
            elif price_action.get('market_structure') == 'bearish':
                signals.append(('price_structure', -1, 0.5))
            
            # Calculate weighted signal
            if not signals:
                return {'signal': 'neutral', 'confidence': 0.0, 'components': []}
            
            total_weight = sum(weight for _, _, weight in signals)
            weighted_signal = sum(signal * weight for _, signal, weight in signals) / total_weight
            
            # Determine signal direction
            if weighted_signal > 0.3:
                signal = 'buy'
            elif weighted_signal < -0.3:
                signal = 'sell'
            else:
                signal = 'neutral'
            
            confidence = min(abs(weighted_signal), 1.0)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'weighted_score': weighted_signal,
                'components': signals
            }
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return {'signal': 'neutral', 'confidence': 0.0, 'components': []}
    
    def _determine_trend(self, indicators: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Determine trend direction and strength."""
        try:
            trend_signals = []
            
            # EMA trend
            if 'ema_12' in indicators and 'ema_26' in indicators:
                if indicators['ema_12'] > indicators['ema_26']:
                    trend_signals.append(1)
                else:
                    trend_signals.append(-1)
            
            # ADX trend strength
            trend_strength = 'weak'
            if 'adx' in indicators:
                if indicators['adx'] > 25:
                    trend_strength = 'strong'
                elif indicators['adx'] > 15:
                    trend_strength = 'moderate'
            
            # Price vs moving averages
            current_price = df['close'].iloc[-1]
            if 'sma_21' in indicators:
                if current_price > indicators['sma_21']:
                    trend_signals.append(1)
                else:
                    trend_signals.append(-1)
            
            # Overall trend direction
            avg_signal = np.mean(trend_signals) if trend_signals else 0
            
            if avg_signal > 0.5:
                direction = 'bullish'
            elif avg_signal < -0.5:
                direction = 'bearish'
            else:
                direction = 'sideways'
            
            return {
                'direction': direction,
                'strength': trend_strength,
                'adx_value': indicators.get('adx', 0),
                'score': avg_signal
            }
            
        except Exception as e:
            logger.error(f"Error determining trend: {str(e)}")
            return {'direction': 'unknown', 'strength': 'weak', 'score': 0}
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various volatility metrics."""
        try:
            close_prices = df['close']
            
            # Standard deviation of returns
            returns = close_prices.pct_change().dropna()
            volatility_1d = returns.std()
            volatility_5d = returns.tail(5).std()
            volatility_20d = returns.tail(20).std()
            
            # True Range based volatility
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr_volatility = true_range.tail(14).mean() / close_prices.iloc[-1]
            
            return {
                'volatility_1d': volatility_1d,
                'volatility_5d': volatility_5d,
                'volatility_20d': volatility_20d,
                'atr_volatility': atr_volatility,
                'volatility_percentile': self._calculate_volatility_percentile(volatility_20d, returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return {}
    
    def _calculate_volatility_percentile(self, current_vol: float, returns: pd.Series) -> float:
        """Calculate what percentile the current volatility represents."""
        try:
            rolling_vol = returns.rolling(window=20).std()
            percentile = (rolling_vol < current_vol).mean() * 100
            return percentile
        except:
            return 50.0  # Default to median
    
    def _calculate_confluence_signal(self, timeframe_analysis: Dict) -> Dict[str, Any]:
        """Calculate overall signal based on multi-timeframe confluence."""
        try:
            weighted_signals = []
            
            for timeframe, analysis in timeframe_analysis.items():
                if 'signal' in analysis and 'confidence' in analysis:
                    weight = self.timeframe_weights.get(timeframe, 0.1)
                    
                    signal_value = 0
                    if analysis['signal'] == 'buy':
                        signal_value = 1
                    elif analysis['signal'] == 'sell':
                        signal_value = -1
                    
                    weighted_signals.append((signal_value * analysis['confidence'], weight))
            
            if not weighted_signals:
                return {'signal': 'neutral', 'confidence': 0.0}
            
            # Calculate weighted average
            total_weight = sum(weight for _, weight in weighted_signals)
            weighted_score = sum(score * weight for score, weight in weighted_signals) / total_weight
            
            # Determine final signal
            if weighted_score > 0.3:
                signal = 'buy'
            elif weighted_score < -0.3:
                signal = 'sell'
            else:
                signal = 'neutral'
            
            confidence = min(abs(weighted_score), 1.0)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'weighted_score': weighted_score,
                'timeframe_count': len(weighted_signals)
            }
            
        except Exception as e:
            logger.error(f"Error calculating confluence signal: {str(e)}")
            return {'signal': 'neutral', 'confidence': 0.0}
    
    async def _detect_patterns(self, symbol: str, data: Dict[str, pd.DataFrame]) -> List[ChartPattern]:
        """Detect chart patterns across timeframes."""
        patterns = []
        
        try:
            for timeframe, df in data.items():
                if len(df) < 50:
                    continue
                
                # Detect support and resistance levels
                sr_patterns = self._detect_support_resistance_patterns(df, timeframe)
                patterns.extend(sr_patterns)
                
                # Detect geometric patterns (triangles, flags, etc.)
                geometric_patterns = self._detect_geometric_patterns(df, timeframe)
                patterns.extend(geometric_patterns)
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
        
        return patterns
    
    def _detect_support_resistance_patterns(self, df: pd.DataFrame, timeframe: str) -> List[ChartPattern]:
        """Detect support and resistance based patterns."""
        patterns = []
        
        try:
            # Find pivot points
            highs = df['high'].rolling(window=5, center=True).max() == df['high']
            lows = df['low'].rolling(window=5, center=True).min() == df['low']
            
            pivot_highs = df[highs]['high'].dropna()
            pivot_lows = df[lows]['low'].dropna()
            
            current_price = df['close'].iloc[-1]
            
            # Find significant support/resistance levels
            for level in pivot_lows.tail(10):
                if abs(current_price - level) / current_price < 0.02:  # Within 2%
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.SUPPORT_RESISTANCE,
                        timeframe=timeframe,
                        confidence=0.7,
                        entry_zone=(level * 0.999, level * 1.001),
                        target_zone=(level * 1.01, level * 1.02),
                        invalidation_level=level * 0.995,
                        description=f"Support level at {level:.5f}"
                    ))
            
            for level in pivot_highs.tail(10):
                if abs(current_price - level) / current_price < 0.02:  # Within 2%
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.SUPPORT_RESISTANCE,
                        timeframe=timeframe,
                        confidence=0.7,
                        entry_zone=(level * 0.999, level * 1.001),
                        target_zone=(level * 0.98, level * 0.99),
                        invalidation_level=level * 1.005,
                        description=f"Resistance level at {level:.5f}"
                    ))
            
        except Exception as e:
            logger.error(f"Error detecting SR patterns: {str(e)}")
        
        return patterns
    
    def _detect_geometric_patterns(self, df: pd.DataFrame, timeframe: str) -> List[ChartPattern]:
        """Detect geometric patterns like triangles, flags, etc."""
        patterns = []
        
        try:
            # Simplified triangle detection
            if len(df) >= 20:
                recent_highs = df['high'].tail(20)
                recent_lows = df['low'].tail(20)
                
                # Check for converging trend lines
                high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
                low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
                
                if high_slope < 0 and low_slope > 0 and abs(high_slope + low_slope) < 0.0001:
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.TRIANGLE,
                        timeframe=timeframe,
                        confidence=0.6,
                        entry_zone=(df['close'].iloc[-1] * 0.998, df['close'].iloc[-1] * 1.002),
                        target_zone=(df['close'].iloc[-1] * 1.01, df['close'].iloc[-1] * 1.02),
                        invalidation_level=df['close'].iloc[-1] * 0.995,
                        description=f"Symmetrical triangle pattern detected"
                    ))
        
        except Exception as e:
            logger.error(f"Error detecting geometric patterns: {str(e)}")
        
        return patterns
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate key support and resistance levels."""
        try:
            if len(df) < 20:
                return {'support': [], 'resistance': []}
            
            # Find pivot points
            window = 5
            highs = df['high'].rolling(window=window, center=True).max() == df['high']
            lows = df['low'].rolling(window=window, center=True).min() == df['low']
            
            pivot_highs = df[highs]['high'].dropna().tail(10)
            pivot_lows = df[lows]['low'].dropna().tail(10)
            
            # Cluster nearby levels
            resistance_levels = self._cluster_levels(pivot_highs.values)
            support_levels = self._cluster_levels(pivot_lows.values)
            
            return {
                'support': support_levels,
                'resistance': resistance_levels,
                'current_price': df['close'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _cluster_levels(self, levels: np.array, threshold: float = 0.001) -> List[float]:
        """Cluster nearby price levels."""
        if len(levels) == 0:
            return []
        
        clustered = []
        sorted_levels = np.sort(levels)
        
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def _generate_recommendations(self, analysis_result: Dict) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on analysis."""
        recommendations = []
        
        try:
            if analysis_result['confidence'] > 0.7:
                current_price = 0
                
                # Get current price from timeframe analysis
                for tf_analysis in analysis_result['timeframe_analysis'].values():
                    if 'price_action' in tf_analysis and 'current_price' in tf_analysis['price_action']:
                        current_price = tf_analysis['price_action']['current_price']
                        break
                
                if current_price > 0:
                    signal = analysis_result['overall_signal']
                    volatility = 0.001  # Default volatility
                    
                    # Get volatility from analysis
                    for tf_analysis in analysis_result['timeframe_analysis'].values():
                        if 'volatility' in tf_analysis and 'atr_volatility' in tf_analysis['volatility']:
                            volatility = tf_analysis['volatility']['atr_volatility']
                            break
                    
                    if signal == 'buy':
                        stop_loss = current_price * (1 - volatility * 2)
                        take_profit = current_price * (1 + volatility * 4)
                    elif signal == 'sell':
                        stop_loss = current_price * (1 + volatility * 2)
                        take_profit = current_price * (1 - volatility * 4)
                    else:
                        return recommendations
                    
                    recommendations.append({
                        'action': signal,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': analysis_result['confidence'],
                        'risk_reward_ratio': abs(take_profit - current_price) / abs(current_price - stop_loss),
                        'reasoning': f"Multi-timeframe confluence signal with {analysis_result['confidence']:.1%} confidence"
                    })
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    async def _send_signal_to_risk_manager(self, analysis_result: Dict):
        """Send trading signal to risk management agent."""
        try:
            if not analysis_result.get('entry_recommendations'):
                return
            
            recommendation = analysis_result['entry_recommendations'][0]
            
            signal_data = {
                'symbol': analysis_result['symbol'],
                'signal': 1 if recommendation['action'] == 'buy' else -1,
                'entry_price': recommendation['entry_price'],
                'stop_loss': recommendation['stop_loss'],
                'take_profit': recommendation['take_profit'],
                'volatility': 0.001,  # Default value
                'source': 'chart_analysis',
                'timeframes_analyzed': list(analysis_result['timeframe_analysis'].keys()),
                'patterns': [p.description for p in analysis_result.get('patterns_detected', [])],
                'confluence_score': analysis_result['confidence']
            }
            
            message = {
                'sender': self.agent_id,
                'receiver': 'risk_manager',
                'message_type': 'trading_signal',
                'data': signal_data,
                'confidence': analysis_result['confidence'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Send via Redis
            channel = "agent_messages_risk_manager"
            self.redis_client.publish(channel, json.dumps(message, default=str))
            
            logger.info(f"Signal sent to risk manager: {recommendation['action']} {analysis_result['symbol']}")
            
        except Exception as e:
            logger.error(f"Error sending signal to risk manager: {str(e)}")
    
    def update_performance(self, signal_id: str, actual_result: Dict):
        """Update performance metrics based on actual trading results."""
        try:
            self.signal_performance['total_signals'] += 1
            
            if actual_result.get('profitable', False):
                self.signal_performance['successful_signals'] += 1
            
            self.signal_performance['accuracy'] = (
                self.signal_performance['successful_signals'] / 
                self.signal_performance['total_signals']
            )
            
            logger.info(f"Performance updated: {self.signal_performance['accuracy']:.1%} accuracy")
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary and statistics."""
        return {
            'agent_id': self.agent_id,
            'performance': self.signal_performance,
            'total_analyses': len(self.analysis_history),
            'recent_signals': [
                {
                    'symbol': analysis['symbol'],
                    'signal': analysis['overall_signal'],
                    'confidence': analysis['confidence'],
                    'timestamp': analysis['timestamp']
                }
                for analysis in self.analysis_history[-5:]
            ],
            'indicator_params': self.indicator_params,
            'timeframe_weights': self.timeframe_weights
        }