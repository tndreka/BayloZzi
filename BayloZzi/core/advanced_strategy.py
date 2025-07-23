# core/advanced_strategy.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdvancedForexStrategy:
    """
    Enhanced forex trading strategy incorporating successful trader techniques:
    1. ICT/Smart Money Concepts
    2. Harmonic Pattern Recognition  
    3. Supply/Demand Zone Analysis
    4. Advanced Risk Management
    """
    
    def __init__(self):
        self.patterns = {}
        self.market_structure = {}
        self.liquidity_zones = {}
        self.smart_money_levels = {}
        
    def detect_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Detect market structure using ICT concepts.
        Identifies swing highs/lows, trend, and structure shifts.
        """
        highs = df['high'].rolling(window=5, center=True).max() == df['high']
        lows = df['low'].rolling(window=5, center=True).min() == df['low']
        
        # Identify swing points
        swing_highs = df[highs].copy()
        swing_lows = df[lows].copy()
        
        # Determine trend based on higher highs/lows or lower highs/lows
        trend = 'neutral'
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            recent_highs = swing_highs.tail(2)['high'].values
            recent_lows = swing_lows.tail(2)['low'].values
            
            if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
                trend = 'bullish'
            elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
                trend = 'bearish'
                
        # Detect market structure shift (MSS)
        mss_levels = []
        if trend == 'bullish' and len(swing_lows) > 1:
            # Look for lower low in uptrend
            for i in range(1, len(swing_lows)):
                if swing_lows.iloc[i]['low'] < swing_lows.iloc[i-1]['low']:
                    mss_levels.append({
                        'level': swing_lows.iloc[i]['low'],
                        'date': swing_lows.index[i],
                        'type': 'bearish_shift'
                    })
                    
        return {
            'trend': trend,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'mss_levels': mss_levels
        }
    
    def identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify institutional order blocks (Smart Money concept).
        """
        order_blocks = []
        
        # Bullish order blocks: Last bearish candle before bullish move
        for i in range(2, len(df)-2):
            # Check for bearish candle followed by strong bullish move
            if (df.iloc[i]['close'] < df.iloc[i]['open'] and  # Bearish candle
                df.iloc[i+1]['close'] > df.iloc[i+1]['open'] and  # Bullish candle
                df.iloc[i+1]['close'] > df.iloc[i]['high'] and  # Strong move
                df.iloc[i+2]['close'] > df.iloc[i+1]['close']):  # Continuation
                
                order_blocks.append({
                    'type': 'bullish',
                    'high': df.iloc[i]['high'],
                    'low': df.iloc[i]['low'],
                    'date': df.index[i],
                    'strength': abs(df.iloc[i+1]['close'] - df.iloc[i]['low']) / df.iloc[i]['low']
                })
                
        # Bearish order blocks: Last bullish candle before bearish move
        for i in range(2, len(df)-2):
            if (df.iloc[i]['close'] > df.iloc[i]['open'] and  # Bullish candle
                df.iloc[i+1]['close'] < df.iloc[i+1]['open'] and  # Bearish candle
                df.iloc[i+1]['close'] < df.iloc[i]['low'] and  # Strong move
                df.iloc[i+2]['close'] < df.iloc[i+1]['close']):  # Continuation
                
                order_blocks.append({
                    'type': 'bearish',
                    'high': df.iloc[i]['high'],
                    'low': df.iloc[i]['low'],
                    'date': df.index[i],
                    'strength': abs(df.iloc[i]['high'] - df.iloc[i+1]['close']) / df.iloc[i]['high']
                })
                
        return order_blocks
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Fair Value Gaps (FVG) - ICT concept.
        """
        fvgs = []
        
        for i in range(1, len(df)-1):
            # Bullish FVG: Gap between candle 1 high and candle 3 low
            if df.iloc[i-1]['high'] < df.iloc[i+1]['low']:
                fvgs.append({
                    'type': 'bullish',
                    'gap_high': df.iloc[i+1]['low'],
                    'gap_low': df.iloc[i-1]['high'],
                    'date': df.index[i],
                    'size': df.iloc[i+1]['low'] - df.iloc[i-1]['high']
                })
                
            # Bearish FVG: Gap between candle 1 low and candle 3 high
            elif df.iloc[i-1]['low'] > df.iloc[i+1]['high']:
                fvgs.append({
                    'type': 'bearish',
                    'gap_high': df.iloc[i-1]['low'],
                    'gap_low': df.iloc[i+1]['high'],
                    'date': df.index[i],
                    'size': df.iloc[i-1]['low'] - df.iloc[i+1]['high']
                })
                
        return fvgs
    
    def identify_liquidity_zones(self, df: pd.DataFrame) -> Dict:
        """
        Identify buy-side and sell-side liquidity zones.
        """
        # Recent highs (buy-side liquidity - where shorts have stops)
        recent_highs = df['high'].rolling(window=20).max()
        buy_side_liquidity = df[df['high'] == recent_highs].copy()
        
        # Recent lows (sell-side liquidity - where longs have stops)
        recent_lows = df['low'].rolling(window=20).min()
        sell_side_liquidity = df[df['low'] == recent_lows].copy()
        
        return {
            'buy_side': buy_side_liquidity,
            'sell_side': sell_side_liquidity
        }
    
    def detect_harmonic_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect harmonic patterns (Gartley, Bat, Butterfly).
        """
        patterns = []
        
        # Simplified Gartley pattern detection
        for i in range(10, len(df)-5):
            # Find potential X point (local extremum)
            window = df.iloc[i-10:i+5]
            
            # Potential bullish Gartley
            x_idx = window['low'].idxmin()
            if x_idx == window.index[0]:  # X is at the beginning
                continue
                
            x_price = window.loc[x_idx, 'low']
            
            # Find A (high after X)
            after_x = window[window.index > x_idx]
            if len(after_x) < 4:
                continue
                
            a_idx = after_x['high'].idxmax()
            a_price = after_x.loc[a_idx, 'high']
            
            # Find B (retracement from A)
            after_a = after_x[after_x.index > a_idx]
            if len(after_a) < 3:
                continue
                
            b_idx = after_a['low'].idxmin()
            b_price = after_a.loc[b_idx, 'low']
            
            # Check Fibonacci ratios
            xa_move = a_price - x_price
            ab_retrace = (a_price - b_price) / xa_move
            
            # Gartley B retracement should be 0.618
            if 0.55 <= ab_retrace <= 0.65:
                # Find C (rally from B)
                after_b = after_a[after_a.index > b_idx]
                if len(after_b) < 2:
                    continue
                    
                c_idx = after_b['high'].idxmax()
                c_price = after_b.loc[c_idx, 'high']
                
                # Check BC retracement
                ab_move = a_price - b_price
                bc_retrace = (c_price - b_price) / ab_move
                
                if 0.382 <= bc_retrace <= 0.886:
                    patterns.append({
                        'type': 'bullish_gartley',
                        'x': {'price': x_price, 'date': x_idx},
                        'a': {'price': a_price, 'date': a_idx},
                        'b': {'price': b_price, 'date': b_idx},
                        'c': {'price': c_price, 'date': c_idx},
                        'd_target': x_price + (0.786 * xa_move),  # D completion
                        'confidence': 0.7 if 0.6 <= ab_retrace <= 0.62 else 0.5
                    })
                    
        return patterns
    
    def calculate_supply_demand_zones(self, df: pd.DataFrame) -> Dict:
        """
        Identify supply and demand zones using price action.
        """
        zones = {'supply': [], 'demand': []}
        
        # Demand zones: Areas where price rallied strongly
        for i in range(2, len(df)-2):
            # Check for consolidation followed by strong rally
            consolidation = abs(df.iloc[i]['close'] - df.iloc[i]['open']) / df.iloc[i]['close'] < 0.001
            strong_rally = df.iloc[i+1]['close'] > df.iloc[i]['high'] * 1.002
            
            if consolidation and strong_rally:
                zones['demand'].append({
                    'zone_high': df.iloc[i]['high'],
                    'zone_low': df.iloc[i]['low'],
                    'date': df.index[i],
                    'strength': (df.iloc[i+1]['close'] - df.iloc[i]['high']) / df.iloc[i]['high']
                })
                
        # Supply zones: Areas where price dropped strongly
        for i in range(2, len(df)-2):
            consolidation = abs(df.iloc[i]['close'] - df.iloc[i]['open']) / df.iloc[i]['close'] < 0.001
            strong_drop = df.iloc[i+1]['close'] < df.iloc[i]['low'] * 0.998
            
            if consolidation and strong_drop:
                zones['supply'].append({
                    'zone_high': df.iloc[i]['high'],
                    'zone_low': df.iloc[i]['low'],
                    'date': df.index[i],
                    'strength': (df.iloc[i]['low'] - df.iloc[i+1]['close']) / df.iloc[i]['low']
                })
                
        return zones
    
    def generate_advanced_signals(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Generate trading signals using multiple advanced concepts.
        """
        if current_idx < 20 or current_idx >= len(df) - 1:
            return {'signal': 0, 'confidence': 0, 'reasons': []}
            
        current_price = df.iloc[current_idx]['close']
        signals = []
        confidence_scores = []
        reasons = []
        
        # 1. Check market structure
        market_struct = self.detect_market_structure(df.iloc[:current_idx+1])
        if market_struct['trend'] == 'bullish':
            signals.append(1)
            confidence_scores.append(0.2)
            reasons.append("Bullish market structure")
        elif market_struct['trend'] == 'bearish':
            signals.append(-1)
            confidence_scores.append(0.2)
            reasons.append("Bearish market structure")
            
        # 2. Check order blocks
        order_blocks = self.identify_order_blocks(df.iloc[:current_idx+1])
        recent_obs = [ob for ob in order_blocks if (current_idx - df.index.get_loc(ob['date'])) < 10]
        
        for ob in recent_obs:
            if ob['type'] == 'bullish' and ob['low'] <= current_price <= ob['high']:
                signals.append(1)
                confidence_scores.append(0.3 * ob['strength'])
                reasons.append(f"At bullish order block (strength: {ob['strength']:.2f})")
            elif ob['type'] == 'bearish' and ob['low'] <= current_price <= ob['high']:
                signals.append(-1)
                confidence_scores.append(0.3 * ob['strength'])
                reasons.append(f"At bearish order block (strength: {ob['strength']:.2f})")
                
        # 3. Check FVGs
        fvgs = self.detect_fair_value_gaps(df.iloc[:current_idx+1])
        recent_fvgs = [fvg for fvg in fvgs if (current_idx - df.index.get_loc(fvg['date'])) < 5]
        
        for fvg in recent_fvgs:
            if fvg['type'] == 'bullish' and fvg['gap_low'] <= current_price <= fvg['gap_high']:
                signals.append(1)
                confidence_scores.append(0.25)
                reasons.append("Price in bullish FVG")
            elif fvg['type'] == 'bearish' and fvg['gap_low'] <= current_price <= fvg['gap_high']:
                signals.append(-1)
                confidence_scores.append(0.25)
                reasons.append("Price in bearish FVG")
                
        # 4. Check harmonic patterns
        patterns = self.detect_harmonic_patterns(df.iloc[:current_idx+1])
        recent_patterns = [p for p in patterns if (current_idx - df.index.get_loc(p['c']['date'])) < 3]
        
        for pattern in recent_patterns:
            if pattern['type'].startswith('bullish'):
                signals.append(1)
                confidence_scores.append(pattern['confidence'] * 0.4)
                reasons.append(f"{pattern['type']} pattern detected")
            elif pattern['type'].startswith('bearish'):
                signals.append(-1)
                confidence_scores.append(pattern['confidence'] * 0.4)
                reasons.append(f"{pattern['type']} pattern detected")
                
        # 5. Check supply/demand zones
        zones = self.calculate_supply_demand_zones(df.iloc[:current_idx+1])
        
        for zone in zones['demand'][-5:]:  # Check last 5 demand zones
            if zone['zone_low'] <= current_price <= zone['zone_high']:
                signals.append(1)
                confidence_scores.append(0.2 + zone['strength'])
                reasons.append(f"At demand zone (strength: {zone['strength']:.3f})")
                
        for zone in zones['supply'][-5:]:  # Check last 5 supply zones
            if zone['zone_low'] <= current_price <= zone['zone_high']:
                signals.append(-1)
                confidence_scores.append(0.2 + zone['strength'])
                reasons.append(f"At supply zone (strength: {zone['strength']:.3f})")
                
        # Aggregate signals
        if not signals:
            return {'signal': 0, 'confidence': 0, 'reasons': []}
            
        # Weight signals by confidence
        weighted_signal = sum(s * c for s, c in zip(signals, confidence_scores)) / sum(confidence_scores)
        
        # Determine final signal
        if weighted_signal > 0.3:
            final_signal = 1
        elif weighted_signal < -0.3:
            final_signal = -1
        else:
            final_signal = 0
            
        final_confidence = min(0.95, sum(confidence_scores) / len(confidence_scores))
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'reasons': reasons,
            'weighted_signal': weighted_signal
        }
    
    def calculate_dynamic_sl_tp(self, df: pd.DataFrame, signal: int, entry_price: float, 
                               current_idx: int) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit using market structure and ATR.
        """
        # Get recent market structure
        market_struct = self.detect_market_structure(df.iloc[:current_idx+1])
        
        # Calculate ATR for volatility-based stops
        atr = df.iloc[max(0, current_idx-14):current_idx+1]['high'].values - \
              df.iloc[max(0, current_idx-14):current_idx+1]['low'].values
        current_atr = np.mean(atr) if len(atr) > 0 else 0.001
        
        if signal == 1:  # Long position
            # Stop loss below recent swing low or order block
            recent_lows = market_struct['swing_lows'].tail(3)
            if not recent_lows.empty:
                structure_stop = recent_lows['low'].min() - (0.5 * current_atr)
            else:
                structure_stop = entry_price - (2 * current_atr)
                
            stop_loss = max(structure_stop, entry_price - (2.5 * current_atr))
            
            # Take profit at next resistance or using risk-reward
            stop_distance = entry_price - stop_loss
            take_profit = entry_price + (stop_distance * 2.5)  # 2.5:1 risk-reward
            
        else:  # Short position
            recent_highs = market_struct['swing_highs'].tail(3)
            if not recent_highs.empty:
                structure_stop = recent_highs['high'].max() + (0.5 * current_atr)
            else:
                structure_stop = entry_price + (2 * current_atr)
                
            stop_loss = min(structure_stop, entry_price + (2.5 * current_atr))
            
            stop_distance = stop_loss - entry_price
            take_profit = entry_price - (stop_distance * 2.5)
            
        return stop_loss, take_profit