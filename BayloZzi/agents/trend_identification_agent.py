# agents/trend_identification_agent.py

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
    def SMA(close, timeperiod):
        return pd.Series(close).rolling(window=timeperiod).mean().values
    
    @staticmethod
    def EMA(close, timeperiod):
        return pd.Series(close).ewm(span=timeperiod, adjust=False).mean().values
    
    @staticmethod
    def RSI(close, timeperiod):
        return ta.momentum.RSIIndicator(pd.Series(close), window=timeperiod).rsi().values
    
    @staticmethod
    def ADX(high, low, close, timeperiod):
        return ta.trend.ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=timeperiod).adx().values
    
    @staticmethod
    def PLUS_DI(high, low, close, timeperiod):
        return ta.trend.ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=timeperiod).adx_pos().values
    
    @staticmethod
    def MINUS_DI(high, low, close, timeperiod):
        return ta.trend.ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=timeperiod).adx_neg().values
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import redis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    STRONG_BULLISH = "strong_bullish"
    MODERATE_BULLISH = "moderate_bullish"
    WEAK_BULLISH = "weak_bullish"
    SIDEWAYS = "sideways"
    WEAK_BEARISH = "weak_bearish"
    MODERATE_BEARISH = "moderate_bearish"
    STRONG_BEARISH = "strong_bearish"

class TrendQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class TrendAnalysis:
    timeframe: str
    direction: TrendDirection
    quality: TrendQuality
    strength: float  # 0-100
    confidence: float  # 0-1
    start_time: datetime
    current_duration: int  # in periods
    slope: float
    r_squared: float
    support_resistance: Dict[str, float]
    trend_line_equation: Tuple[float, float]  # (slope, intercept)
    expected_targets: Dict[str, float]
    invalidation_level: float

@dataclass
class MultiTimeframeTrend:
    symbol: str
    primary_trend: TrendAnalysis  # Longer timeframe
    secondary_trend: TrendAnalysis  # Medium timeframe
    tertiary_trend: TrendAnalysis  # Shorter timeframe
    confluence_score: float  # 0-1, how well trends align
    overall_direction: TrendDirection
    trend_change_probability: float
    next_major_level: float
    recommendation: str

class TrendIdentificationAgent:
    """
    Advanced Trend Identification Agent for Multi-Agent Forex Trading System.
    Specializes in multi-timeframe trend analysis and trend change detection.
    """
    
    def __init__(self, agent_id: str = "trend_identifier"):
        self.agent_id = agent_id
        
        # Redis connection for inter-agent communication
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Timeframe hierarchy for trend analysis
        self.timeframe_hierarchy = {
            'primary': ['1d', '4h'],      # Major trend
            'secondary': ['1h', '30m'],   # Intermediate trend
            'tertiary': ['15m', '5m']     # Short-term trend
        }
        
        # Trend identification parameters
        self.trend_params = {
            'min_periods_for_trend': 10,
            'trend_strength_threshold': 0.6,
            'confluence_threshold': 0.7,
            'trend_change_threshold': 0.75,
            'adx_strong_trend': 25,
            'adx_moderate_trend': 20,
            'r_squared_threshold': 0.5
        }
        
        # Trend tracking
        self.active_trends = {}
        self.trend_history = []
        self.trend_performance = {
            'successful_trend_calls': 0,
            'total_trend_calls': 0,
            'accuracy': 0.0,
            'avg_trend_duration': 0,
            'false_breakout_rate': 0.0
        }
        
        # Regression models for trend analysis
        self.trend_models = {}
        
        # Price level tracking
        self.key_levels = {}
        
        # Confluence weights for different timeframes
        self.confluence_weights = {
            '1d': 0.4,
            '4h': 0.3,
            '1h': 0.2,
            '15m': 0.1
        }
    
    async def analyze_trends(self, symbol: str, data: Dict[str, pd.DataFrame]) -> MultiTimeframeTrend:
        """
        Comprehensive multi-timeframe trend analysis.
        
        Args:
            symbol: Currency pair symbol
            data: Dictionary of timeframe -> OHLCV DataFrame
            
        Returns:
            Complete multi-timeframe trend analysis
        """
        try:
            # Analyze trends for each timeframe category
            primary_analysis = await self._analyze_timeframe_category(
                symbol, data, self.timeframe_hierarchy['primary']
            )
            
            secondary_analysis = await self._analyze_timeframe_category(
                symbol, data, self.timeframe_hierarchy['secondary']
            )
            
            tertiary_analysis = await self._analyze_timeframe_category(
                symbol, data, self.timeframe_hierarchy['tertiary']
            )
            
            # Calculate confluence and overall trend
            confluence_score = self._calculate_trend_confluence(
                primary_analysis, secondary_analysis, tertiary_analysis
            )
            
            overall_direction = self._determine_overall_trend(
                primary_analysis, secondary_analysis, tertiary_analysis, confluence_score
            )
            
            # Calculate trend change probability
            trend_change_prob = await self._calculate_trend_change_probability(
                symbol, data, primary_analysis, secondary_analysis
            )
            
            # Find next major level
            next_level = self._find_next_major_level(symbol, data, overall_direction)
            
            # Generate recommendation
            recommendation = self._generate_trend_recommendation(
                overall_direction, confluence_score, trend_change_prob
            )
            
            # Create multi-timeframe trend object
            mtf_trend = MultiTimeframeTrend(
                symbol=symbol,
                primary_trend=primary_analysis,
                secondary_trend=secondary_analysis,
                tertiary_trend=tertiary_analysis,
                confluence_score=confluence_score,
                overall_direction=overall_direction,
                trend_change_probability=trend_change_prob,
                next_major_level=next_level,
                recommendation=recommendation
            )
            
            # Store for tracking
            self.active_trends[symbol] = mtf_trend
            self.trend_history.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'analysis': mtf_trend
            })
            
            # Send to other agents if confidence is high
            if confluence_score > self.trend_params['confluence_threshold']:
                await self._send_trend_signal(mtf_trend)
            
            logger.info(f"Trend analysis completed for {symbol}: {overall_direction.value} (confluence: {confluence_score:.2f})")
            
            return mtf_trend
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return None
    
    async def _analyze_timeframe_category(self, symbol: str, data: Dict[str, pd.DataFrame], 
                                        timeframes: List[str]) -> TrendAnalysis:
        """Analyze trend for a category of timeframes (primary/secondary/tertiary)."""
        try:
            best_analysis = None
            best_confidence = 0
            
            for timeframe in timeframes:
                if timeframe in data and len(data[timeframe]) >= self.trend_params['min_periods_for_trend']:
                    analysis = await self._analyze_single_timeframe_trend(
                        symbol, timeframe, data[timeframe]
                    )
                    
                    if analysis and analysis.confidence > best_confidence:
                        best_analysis = analysis
                        best_confidence = analysis.confidence
            
            return best_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe category: {str(e)}")
            return None
    
    async def _analyze_single_timeframe_trend(self, symbol: str, timeframe: str, 
                                            df: pd.DataFrame) -> TrendAnalysis:
        """Perform comprehensive trend analysis for a single timeframe."""
        try:
            if len(df) < self.trend_params['min_periods_for_trend']:
                return None
            
            # Prepare data
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            timestamps = df.index
            
            # 1. Linear regression trend analysis
            slope, intercept, r_squared = self._calculate_trend_line(close_prices)
            
            # 2. ADX trend strength
            adx_strength = self._calculate_adx_strength(df)
            
            # 3. Moving average trend analysis
            ma_trend = self._analyze_moving_average_trend(df)
            
            # 4. Market structure analysis
            market_structure = self._analyze_market_structure(df)
            
            # 5. Volume trend analysis (if available)
            volume_trend = self._analyze_volume_trend(df)
            
            # 6. Determine trend direction and strength
            direction, strength = self._determine_trend_direction_strength(
                slope, adx_strength, ma_trend, market_structure
            )
            
            # 7. Calculate trend quality
            quality = self._assess_trend_quality(r_squared, adx_strength, market_structure)
            
            # 8. Find trend start time and duration
            start_time, duration = self._find_trend_start_duration(df, direction)
            
            # 9. Calculate support/resistance levels
            sr_levels = self._calculate_trend_support_resistance(df, slope, intercept)
            
            # 10. Predict targets and invalidation levels
            targets = self._calculate_trend_targets(close_prices[-1], slope, df['high'].max(), df['low'].min())
            invalidation = self._calculate_invalidation_level(df, direction, sr_levels)
            
            # 11. Calculate overall confidence
            confidence = self._calculate_trend_confidence(
                r_squared, adx_strength, strength, quality, market_structure
            )
            
            return TrendAnalysis(
                timeframe=timeframe,
                direction=direction,
                quality=quality,
                strength=strength,
                confidence=confidence,
                start_time=start_time,
                current_duration=duration,
                slope=slope,
                r_squared=r_squared,
                support_resistance=sr_levels,
                trend_line_equation=(slope, intercept),
                expected_targets=targets,
                invalidation_level=invalidation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing single timeframe trend: {str(e)}")
            return None
    
    def _calculate_trend_line(self, prices: np.array) -> Tuple[float, float, float]:
        """Calculate trend line using linear regression."""
        try:
            x = np.arange(len(prices)).reshape(-1, 1)
            y = prices
            
            model = LinearRegression()
            model.fit(x, y)
            
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # Calculate R-squared
            y_pred = model.predict(x)
            r_squared = r2_score(y, y_pred)
            
            return slope, intercept, r_squared
            
        except Exception as e:
            logger.error(f"Error calculating trend line: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def _calculate_adx_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate ADX-based trend strength."""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate ADX
            adx = talib.ADX(high, low, close, timeperiod=14)
            
            # Calculate +DI and -DI for direction
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
            current_plus_di = plus_di[-1] if not np.isnan(plus_di[-1]) else 0
            current_minus_di = minus_di[-1] if not np.isnan(minus_di[-1]) else 0
            
            return {
                'adx': current_adx,
                'plus_di': current_plus_di,
                'minus_di': current_minus_di,
                'di_spread': abs(current_plus_di - current_minus_di)
            }
            
        except Exception as e:
            logger.error(f"Error calculating ADX strength: {str(e)}")
            return {'adx': 0, 'plus_di': 0, 'minus_di': 0, 'di_spread': 0}
    
    def _analyze_moving_average_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend using multiple moving averages."""
        try:
            close = df['close'].values
            
            # Calculate multiple moving averages
            sma_9 = talib.SMA(close, timeperiod=9)
            sma_21 = talib.SMA(close, timeperiod=21)
            sma_50 = talib.SMA(close, timeperiod=50)
            ema_12 = talib.EMA(close, timeperiod=12)
            ema_26 = talib.EMA(close, timeperiod=26)
            
            current_price = close[-1]
            
            # Analyze MA alignment
            ma_alignment = 0
            if current_price > sma_9[-1] > sma_21[-1] > sma_50[-1]:
                ma_alignment = 1  # Bullish alignment
            elif current_price < sma_9[-1] < sma_21[-1] < sma_50[-1]:
                ma_alignment = -1  # Bearish alignment
            
            # MA slope analysis
            ma_slopes = {
                'sma_9_slope': (sma_9[-1] - sma_9[-5]) / 4 if len(sma_9) >= 5 else 0,
                'sma_21_slope': (sma_21[-1] - sma_21[-5]) / 4 if len(sma_21) >= 5 else 0,
                'ema_12_slope': (ema_12[-1] - ema_12[-5]) / 4 if len(ema_12) >= 5 else 0
            }
            
            return {
                'alignment': ma_alignment,
                'slopes': ma_slopes,
                'price_vs_ma': {
                    'above_sma_9': current_price > sma_9[-1],
                    'above_sma_21': current_price > sma_21[-1],
                    'above_sma_50': current_price > sma_50[-1]
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing MA trend: {str(e)}")
            return {'alignment': 0, 'slopes': {}, 'price_vs_ma': {}}
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure (higher highs, lower lows, etc.)."""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(high) - 2):
                # Swing high: current high > previous 2 and next 2 highs
                if (high[i] > high[i-1] and high[i] > high[i-2] and 
                    high[i] > high[i+1] and high[i] > high[i+2]):
                    swing_highs.append((i, high[i]))
                
                # Swing low: current low < previous 2 and next 2 lows
                if (low[i] < low[i-1] and low[i] < low[i-2] and 
                    low[i] < low[i+1] and low[i] < low[i+2]):
                    swing_lows.append((i, low[i]))
            
            # Analyze recent structure (last 5 swings)
            recent_highs = swing_highs[-5:] if len(swing_highs) >= 5 else swing_highs
            recent_lows = swing_lows[-5:] if len(swing_lows) >= 5 else swing_lows
            
            # Count higher highs and lower lows
            higher_highs = 0
            lower_lows = 0
            
            for i in range(1, len(recent_highs)):
                if recent_highs[i][1] > recent_highs[i-1][1]:
                    higher_highs += 1
            
            for i in range(1, len(recent_lows)):
                if recent_lows[i][1] < recent_lows[i-1][1]:
                    lower_lows += 1
            
            # Determine structure bias
            if higher_highs >= 2 and lower_lows <= 1:
                structure_bias = 'bullish'
            elif lower_lows >= 2 and higher_highs <= 1:
                structure_bias = 'bearish'
            else:
                structure_bias = 'mixed'
            
            return {
                'structure_bias': structure_bias,
                'higher_highs': higher_highs,
                'lower_lows': lower_lows,
                'swing_highs': recent_highs,
                'swing_lows': recent_lows
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return {'structure_bias': 'mixed', 'higher_highs': 0, 'lower_lows': 0}
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume trend if volume data is available."""
        try:
            if 'volume' not in df.columns:
                return {'volume_trend': 'unknown', 'volume_confirmation': False}
            
            volume = df['volume'].values
            close = df['close'].values
            
            # Calculate volume moving average
            volume_ma = talib.SMA(volume, timeperiod=20)
            
            # Price-volume relationship
            price_change = np.diff(close)
            volume_change = np.diff(volume[1:])  # Align with price_change
            
            # Volume confirmation of trend
            bullish_volume = np.sum((price_change > 0) & (volume_change > 0))
            bearish_volume = np.sum((price_change < 0) & (volume_change > 0))
            
            volume_confirmation = bullish_volume > bearish_volume
            
            return {
                'volume_trend': 'increasing' if volume[-1] > volume_ma[-1] else 'decreasing',
                'volume_confirmation': volume_confirmation,
                'avg_volume': np.mean(volume[-20:])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume trend: {str(e)}")
            return {'volume_trend': 'unknown', 'volume_confirmation': False}
    
    def _determine_trend_direction_strength(self, slope: float, adx_data: Dict, 
                                          ma_trend: Dict, structure: Dict) -> Tuple[TrendDirection, float]:
        """Determine overall trend direction and strength."""
        try:
            # Normalize slope
            slope_normalized = np.tanh(slope * 10000) * 100  # Scale and bound to -100, 100
            
            # Get ADX strength
            adx_strength = adx_data.get('adx', 0)
            di_difference = adx_data.get('plus_di', 0) - adx_data.get('minus_di', 0)
            
            # MA alignment score
            ma_score = ma_trend.get('alignment', 0) * 30
            
            # Structure score
            structure_score = 0
            if structure.get('structure_bias') == 'bullish':
                structure_score = 20
            elif structure.get('structure_bias') == 'bearish':
                structure_score = -20
            
            # Combined score
            total_score = slope_normalized + ma_score + structure_score + di_difference
            
            # Determine direction based on score and ADX
            if total_score > 50 and adx_strength > self.trend_params['adx_strong_trend']:
                direction = TrendDirection.STRONG_BULLISH
                strength = min(100, total_score)
            elif total_score > 30 and adx_strength > self.trend_params['adx_moderate_trend']:
                direction = TrendDirection.MODERATE_BULLISH
                strength = min(100, total_score)
            elif total_score > 10:
                direction = TrendDirection.WEAK_BULLISH
                strength = min(100, total_score)
            elif total_score < -50 and adx_strength > self.trend_params['adx_strong_trend']:
                direction = TrendDirection.STRONG_BEARISH
                strength = min(100, abs(total_score))
            elif total_score < -30 and adx_strength > self.trend_params['adx_moderate_trend']:
                direction = TrendDirection.MODERATE_BEARISH
                strength = min(100, abs(total_score))
            elif total_score < -10:
                direction = TrendDirection.WEAK_BEARISH
                strength = min(100, abs(total_score))
            else:
                direction = TrendDirection.SIDEWAYS
                strength = 100 - abs(total_score)
            
            return direction, strength
            
        except Exception as e:
            logger.error(f"Error determining trend direction: {str(e)}")
            return TrendDirection.SIDEWAYS, 0
    
    def _assess_trend_quality(self, r_squared: float, adx_data: Dict, structure: Dict) -> TrendQuality:
        """Assess the quality of the identified trend."""
        try:
            adx_strength = adx_data.get('adx', 0)
            structure_bias = structure.get('structure_bias', 'mixed')
            
            quality_score = 0
            
            # R-squared contribution (40%)
            if r_squared > 0.8:
                quality_score += 40
            elif r_squared > 0.6:
                quality_score += 30
            elif r_squared > 0.4:
                quality_score += 20
            else:
                quality_score += 10
            
            # ADX contribution (35%)
            if adx_strength > 30:
                quality_score += 35
            elif adx_strength > 25:
                quality_score += 25
            elif adx_strength > 20:
                quality_score += 15
            else:
                quality_score += 5
            
            # Structure contribution (25%)
            if structure_bias in ['bullish', 'bearish']:
                quality_score += 25
            else:
                quality_score += 10
            
            # Determine quality level
            if quality_score >= 80:
                return TrendQuality.EXCELLENT
            elif quality_score >= 60:
                return TrendQuality.GOOD
            elif quality_score >= 40:
                return TrendQuality.FAIR
            else:
                return TrendQuality.POOR
                
        except Exception as e:
            logger.error(f"Error assessing trend quality: {str(e)}")
            return TrendQuality.POOR
    
    def _find_trend_start_duration(self, df: pd.DataFrame, direction: TrendDirection) -> Tuple[datetime, int]:
        """Find when the current trend started and its duration."""
        try:
            close = df['close'].values
            timestamps = df.index
            
            # Simple trend start detection (when price crossed key MA)
            sma_21 = talib.SMA(close, timeperiod=21)
            
            trend_start_idx = len(df) - 1
            
            if 'bullish' in direction.value:
                # Find where price last crossed above SMA
                for i in range(len(close) - 1, 0, -1):
                    if close[i] > sma_21[i] and close[i-1] <= sma_21[i-1]:
                        trend_start_idx = i
                        break
            elif 'bearish' in direction.value:
                # Find where price last crossed below SMA
                for i in range(len(close) - 1, 0, -1):
                    if close[i] < sma_21[i] and close[i-1] >= sma_21[i-1]:
                        trend_start_idx = i
                        break
            
            start_time = timestamps[trend_start_idx]
            duration = len(df) - trend_start_idx
            
            return start_time, duration
            
        except Exception as e:
            logger.error(f"Error finding trend start: {str(e)}")
            return datetime.now(), 0
    
    def _calculate_trend_support_resistance(self, df: pd.DataFrame, slope: float, 
                                          intercept: float) -> Dict[str, float]:
        """Calculate dynamic support/resistance based on trend."""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            trend_line_value = slope * (len(close) - 1) + intercept
            
            # Calculate parallel support/resistance lines
            price_deviations = close - (slope * np.arange(len(close)) + intercept)
            
            # Support and resistance based on standard deviations
            std_dev = np.std(price_deviations)
            
            if slope > 0:  # Uptrend
                support_1 = trend_line_value - std_dev
                support_2 = trend_line_value - 2 * std_dev
                resistance_1 = trend_line_value + std_dev
                resistance_2 = trend_line_value + 2 * std_dev
            else:  # Downtrend
                resistance_1 = trend_line_value + std_dev
                resistance_2 = trend_line_value + 2 * std_dev
                support_1 = trend_line_value - std_dev
                support_2 = trend_line_value - 2 * std_dev
            
            return {
                'trend_line': trend_line_value,
                'support_1': support_1,
                'support_2': support_2,
                'resistance_1': resistance_1,
                'resistance_2': resistance_2,
                'dynamic_support': min(support_1, support_2),
                'dynamic_resistance': max(resistance_1, resistance_2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend SR: {str(e)}")
            return {}
    
    def _calculate_trend_targets(self, current_price: float, slope: float, 
                               recent_high: float, recent_low: float) -> Dict[str, float]:
        """Calculate trend targets based on slope and recent levels."""
        try:
            # Project trend forward
            projected_periods = [10, 20, 50]  # periods ahead
            targets = {}
            
            for periods in projected_periods:
                projected_price = current_price + (slope * periods)
                targets[f'target_{periods}p'] = projected_price
            
            # Add percentage-based targets
            if slope > 0:  # Bullish trend
                targets['target_2pct'] = current_price * 1.02
                targets['target_5pct'] = current_price * 1.05
                targets['target_swing_high'] = recent_high * 1.01
            else:  # Bearish trend
                targets['target_2pct'] = current_price * 0.98
                targets['target_5pct'] = current_price * 0.95
                targets['target_swing_low'] = recent_low * 0.99
            
            return targets
            
        except Exception as e:
            logger.error(f"Error calculating trend targets: {str(e)}")
            return {}
    
    def _calculate_invalidation_level(self, df: pd.DataFrame, direction: TrendDirection, 
                                    sr_levels: Dict) -> float:
        """Calculate level where trend would be invalidated."""
        try:
            close = df['close'].values
            current_price = close[-1]
            
            if 'bullish' in direction.value:
                # For bullish trends, invalidation is below recent support
                invalidation = sr_levels.get('dynamic_support', current_price * 0.98)
                # Also consider recent swing low
                recent_low = min(df['low'].tail(20))
                invalidation = min(invalidation, recent_low * 0.999)
            else:
                # For bearish trends, invalidation is above recent resistance
                invalidation = sr_levels.get('dynamic_resistance', current_price * 1.02)
                # Also consider recent swing high
                recent_high = max(df['high'].tail(20))
                invalidation = max(invalidation, recent_high * 1.001)
            
            return invalidation
            
        except Exception as e:
            logger.error(f"Error calculating invalidation level: {str(e)}")
            return 0.0
    
    def _calculate_trend_confidence(self, r_squared: float, adx_data: Dict, 
                                  strength: float, quality: TrendQuality, 
                                  structure: Dict) -> float:
        """Calculate overall confidence in trend analysis."""
        try:
            confidence_factors = []
            
            # R-squared factor (30%)
            confidence_factors.append(r_squared * 0.3)
            
            # ADX factor (25%)
            adx_normalized = min(adx_data.get('adx', 0) / 30, 1.0)
            confidence_factors.append(adx_normalized * 0.25)
            
            # Strength factor (20%)
            strength_normalized = strength / 100
            confidence_factors.append(strength_normalized * 0.2)
            
            # Quality factor (15%)
            quality_scores = {
                TrendQuality.EXCELLENT: 1.0,
                TrendQuality.GOOD: 0.8,
                TrendQuality.FAIR: 0.6,
                TrendQuality.POOR: 0.3
            }
            confidence_factors.append(quality_scores.get(quality, 0.3) * 0.15)
            
            # Structure factor (10%)
            structure_score = 1.0 if structure.get('structure_bias') in ['bullish', 'bearish'] else 0.5
            confidence_factors.append(structure_score * 0.1)
            
            total_confidence = sum(confidence_factors)
            return min(max(total_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating trend confidence: {str(e)}")
            return 0.5
    
    def _calculate_trend_confluence(self, primary: TrendAnalysis, secondary: TrendAnalysis, 
                                  tertiary: TrendAnalysis) -> float:
        """Calculate confluence score between different timeframe trends."""
        try:
            if not all([primary, secondary, tertiary]):
                return 0.0
            
            trends = [primary, secondary, tertiary]
            weights = [0.5, 0.3, 0.2]  # Primary timeframe has most weight
            
            # Check direction alignment
            directions = [trend.direction.value for trend in trends]
            
            bullish_count = sum(1 for d in directions if 'bullish' in d)
            bearish_count = sum(1 for d in directions if 'bearish' in d)
            sideways_count = sum(1 for d in directions if 'sideways' in d)
            
            # Calculate alignment score
            if bullish_count >= 2:
                alignment_score = bullish_count / 3
            elif bearish_count >= 2:
                alignment_score = bearish_count / 3
            else:
                alignment_score = 0.3  # Mixed signals
            
            # Weight by confidence
            weighted_confidence = sum(
                trend.confidence * weight for trend, weight in zip(trends, weights)
            )
            
            # Combine alignment and confidence
            confluence = (alignment_score * 0.6) + (weighted_confidence * 0.4)
            
            return min(max(confluence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confluence: {str(e)}")
            return 0.0
    
    def _determine_overall_trend(self, primary: TrendAnalysis, secondary: TrendAnalysis,
                               tertiary: TrendAnalysis, confluence: float) -> TrendDirection:
        """Determine overall trend direction from multi-timeframe analysis."""
        try:
            if not primary:
                return TrendDirection.SIDEWAYS
            
            # Primary timeframe has the most influence
            primary_direction = primary.direction
            
            # If confluence is high, trust the primary trend
            if confluence > 0.8:
                return primary_direction
            
            # Otherwise, consider secondary trend
            if secondary and confluence > 0.6:
                # If primary and secondary align, use primary strength
                if ('bullish' in primary_direction.value and 'bullish' in secondary.direction.value) or \
                   ('bearish' in primary_direction.value and 'bearish' in secondary.direction.value):
                    return primary_direction
            
            # If low confluence, be more conservative
            if confluence < 0.5:
                if primary_direction in [TrendDirection.STRONG_BULLISH, TrendDirection.STRONG_BEARISH]:
                    # Downgrade strong trends to moderate
                    if 'bullish' in primary_direction.value:
                        return TrendDirection.MODERATE_BULLISH
                    else:
                        return TrendDirection.MODERATE_BEARISH
                else:
                    return TrendDirection.SIDEWAYS
            
            return primary_direction
            
        except Exception as e:
            logger.error(f"Error determining overall trend: {str(e)}")
            return TrendDirection.SIDEWAYS
    
    async def _calculate_trend_change_probability(self, symbol: str, data: Dict[str, pd.DataFrame],
                                                primary: TrendAnalysis, secondary: TrendAnalysis) -> float:
        """Calculate probability of trend change."""
        try:
            probability_factors = []
            
            # Factor 1: Trend duration vs historical average
            if primary and primary.current_duration > 0:
                avg_trend_duration = self.trend_performance.get('avg_trend_duration', 20)
                duration_factor = min(primary.current_duration / avg_trend_duration, 1.5)
                probability_factors.append(duration_factor * 0.3)
            
            # Factor 2: Momentum divergence
            momentum_divergence = await self._detect_momentum_divergence(data.get('1h', pd.DataFrame()))
            probability_factors.append(momentum_divergence * 0.25)
            
            # Factor 3: Key level proximity
            key_level_factor = self._calculate_key_level_proximity(symbol, data)
            probability_factors.append(key_level_factor * 0.2)
            
            # Factor 4: Timeframe conflicts
            if primary and secondary:
                conflict_factor = 0
                if ('bullish' in primary.direction.value and 'bearish' in secondary.direction.value) or \
                   ('bearish' in primary.direction.value and 'bullish' in secondary.direction.value):
                    conflict_factor = 0.5
                probability_factors.append(conflict_factor * 0.15)
            
            # Factor 5: Overextension
            overextension_factor = self._calculate_overextension(data.get('4h', pd.DataFrame()))
            probability_factors.append(overextension_factor * 0.1)
            
            total_probability = sum(probability_factors)
            return min(max(total_probability, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating trend change probability: {str(e)}")
            return 0.5
    
    async def _detect_momentum_divergence(self, df: pd.DataFrame) -> float:
        """Detect momentum divergence that could signal trend change."""
        try:
            if len(df) < 20:
                return 0.0
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Calculate RSI
            rsi = talib.RSI(close, timeperiod=14)
            
            # Find recent peaks in price and RSI
            price_peaks = []
            rsi_peaks = []
            
            for i in range(5, len(close) - 5):
                if close[i] == max(close[i-5:i+6]):
                    price_peaks.append((i, close[i]))
                if rsi[i] == max(rsi[i-5:i+6]):
                    rsi_peaks.append((i, rsi[i]))
            
            # Check for divergence in last 2 peaks
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                last_price_peaks = price_peaks[-2:]
                last_rsi_peaks = rsi_peaks[-2:]
                
                price_direction = last_price_peaks[1][1] - last_price_peaks[0][1]
                rsi_direction = last_rsi_peaks[1][1] - last_rsi_peaks[0][1]
                
                # Bearish divergence: higher prices, lower RSI
                if price_direction > 0 and rsi_direction < 0:
                    return 0.8
                # Bullish divergence: lower prices, higher RSI
                elif price_direction < 0 and rsi_direction > 0:
                    return 0.8
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error detecting momentum divergence: {str(e)}")
            return 0.0
    
    def _calculate_key_level_proximity(self, symbol: str, data: Dict[str, pd.DataFrame]) -> float:
        """Calculate proximity to key support/resistance levels."""
        try:
            df = data.get('1d', data.get('4h', pd.DataFrame()))
            if df.empty:
                return 0.0
            
            current_price = df['close'].iloc[-1]
            
            # Calculate major support/resistance levels
            high_levels = df['high'].rolling(window=20).max().dropna().unique()
            low_levels = df['low'].rolling(window=20).min().dropna().unique()
            
            key_levels = np.concatenate([high_levels, low_levels])
            
            # Find closest level
            distances = [abs(current_price - level) / current_price for level in key_levels]
            min_distance = min(distances) if distances else 1.0
            
            # Proximity factor (closer = higher probability of reaction)
            if min_distance < 0.005:  # Within 0.5%
                return 0.9
            elif min_distance < 0.01:  # Within 1%
                return 0.6
            elif min_distance < 0.02:  # Within 2%
                return 0.3
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating key level proximity: {str(e)}")
            return 0.0
    
    def _calculate_overextension(self, df: pd.DataFrame) -> float:
        """Calculate if price is overextended from moving averages."""
        try:
            if len(df) < 50:
                return 0.0
            
            close = df['close'].values
            current_price = close[-1]
            
            # Calculate distance from key moving averages
            sma_50 = talib.SMA(close, timeperiod=50)[-1]
            sma_200 = talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else sma_50
            
            # Calculate percentage distances
            dist_50 = abs(current_price - sma_50) / sma_50
            dist_200 = abs(current_price - sma_200) / sma_200
            
            # Overextension thresholds
            if dist_50 > 0.05 or dist_200 > 0.1:  # More than 5% from SMA50 or 10% from SMA200
                return 0.7
            elif dist_50 > 0.03 or dist_200 > 0.06:
                return 0.4
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating overextension: {str(e)}")
            return 0.0
    
    def _find_next_major_level(self, symbol: str, data: Dict[str, pd.DataFrame], 
                             direction: TrendDirection) -> float:
        """Find next major support/resistance level in trend direction."""
        try:
            df = data.get('1d', data.get('4h', pd.DataFrame()))
            if df.empty:
                return 0.0
            
            current_price = df['close'].iloc[-1]
            
            if 'bullish' in direction.value:
                # Find next resistance
                recent_highs = df['high'].tail(50)
                resistance_levels = recent_highs[recent_highs > current_price].sort_values()
                return resistance_levels.iloc[0] if not resistance_levels.empty else current_price * 1.02
            
            elif 'bearish' in direction.value:
                # Find next support
                recent_lows = df['low'].tail(50)
                support_levels = recent_lows[recent_lows < current_price].sort_values(ascending=False)
                return support_levels.iloc[0] if not support_levels.empty else current_price * 0.98
            
            else:
                return current_price
                
        except Exception as e:
            logger.error(f"Error finding next major level: {str(e)}")
            return 0.0
    
    def _generate_trend_recommendation(self, direction: TrendDirection, confluence: float, 
                                     change_probability: float) -> str:
        """Generate trading recommendation based on trend analysis."""
        try:
            if change_probability > 0.7:
                return "CAUTION: High probability of trend change. Avoid new positions."
            
            if confluence < 0.5:
                return "MIXED SIGNALS: Low timeframe confluence. Wait for clearer direction."
            
            if direction == TrendDirection.STRONG_BULLISH and confluence > 0.8:
                return "STRONG BUY: Robust bullish trend with high confluence."
            
            elif direction == TrendDirection.MODERATE_BULLISH and confluence > 0.7:
                return "BUY: Moderate bullish trend. Consider long positions."
            
            elif direction == TrendDirection.WEAK_BULLISH:
                return "WEAK BUY: Weak bullish bias. Use tight stops."
            
            elif direction == TrendDirection.STRONG_BEARISH and confluence > 0.8:
                return "STRONG SELL: Robust bearish trend with high confluence."
            
            elif direction == TrendDirection.MODERATE_BEARISH and confluence > 0.7:
                return "SELL: Moderate bearish trend. Consider short positions."
            
            elif direction == TrendDirection.WEAK_BEARISH:
                return "WEAK SELL: Weak bearish bias. Use tight stops."
            
            else:
                return "NEUTRAL: Sideways trend. Range trading opportunities."
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return "NEUTRAL: Unable to determine trend direction."
    
    async def _send_trend_signal(self, mtf_trend: MultiTimeframeTrend):
        """Send trend signal to other agents."""
        try:
            signal_data = {
                'symbol': mtf_trend.symbol,
                'overall_direction': mtf_trend.overall_direction.value,
                'confluence_score': mtf_trend.confluence_score,
                'trend_change_probability': mtf_trend.trend_change_probability,
                'next_major_level': mtf_trend.next_major_level,
                'recommendation': mtf_trend.recommendation,
                'primary_trend': {
                    'direction': mtf_trend.primary_trend.direction.value,
                    'strength': mtf_trend.primary_trend.strength,
                    'confidence': mtf_trend.primary_trend.confidence
                } if mtf_trend.primary_trend else None
            }
            
            message = {
                'sender': self.agent_id,
                'receiver': 'risk_manager',
                'message_type': 'trend_analysis',
                'data': signal_data,
                'confidence': mtf_trend.confluence_score,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to risk manager
            channel = "agent_messages_risk_manager"
            self.redis_client.publish(channel, json.dumps(message, default=str))
            
            # Also send to chart analyzer for confluence
            message['receiver'] = 'chart_analyzer'
            channel = "agent_messages_chart_analyzer"
            self.redis_client.publish(channel, json.dumps(message, default=str))
            
            logger.info(f"Trend signal sent: {mtf_trend.overall_direction.value} for {mtf_trend.symbol}")
            
        except Exception as e:
            logger.error(f"Error sending trend signal: {str(e)}")
    
    def update_performance(self, prediction_result: Dict):
        """Update trend identification performance metrics."""
        try:
            self.trend_performance['total_trend_calls'] += 1
            
            if prediction_result.get('successful', False):
                self.trend_performance['successful_trend_calls'] += 1
            
            self.trend_performance['accuracy'] = (
                self.trend_performance['successful_trend_calls'] / 
                self.trend_performance['total_trend_calls']
            )
            
            # Update average trend duration
            if 'duration' in prediction_result:
                current_avg = self.trend_performance.get('avg_trend_duration', 0)
                total_calls = self.trend_performance['total_trend_calls']
                new_avg = (current_avg * (total_calls - 1) + prediction_result['duration']) / total_calls
                self.trend_performance['avg_trend_duration'] = new_avg
            
            logger.info(f"Trend performance updated: {self.trend_performance['accuracy']:.1%} accuracy")
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
    
    def get_trend_summary(self) -> Dict[str, Any]:
        """Get comprehensive trend analysis summary."""
        return {
            'agent_id': self.agent_id,
            'performance': self.trend_performance,
            'active_trends': {symbol: {
                'direction': trend.overall_direction.value,
                'confluence': trend.confluence_score,
                'change_probability': trend.trend_change_probability
            } for symbol, trend in self.active_trends.items()},
            'total_analyses': len(self.trend_history),
            'trend_params': self.trend_params
        }