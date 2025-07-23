"""Backtrader data feed exposing engineered feature columns.

Backtrader's built-in PandasData only gives OHLCV. Our ML-driven strategy
needs technical-indicator columns (MA5, RSI, etc.). This custom feed maps
those DataFrame columns to Backtrader *lines* so they are accessible inside
the strategy via `self.data.<column>`.

Usage:
    from core.bt_feed import FeatureData
    cerebro.adddata(FeatureData(dataname=df))
"""

import backtrader as bt


class FeatureData(bt.feeds.PandasData):
    # List all lines: default OHLCV + engineered features
    lines = (
        'open', 'high', 'low', 'close', 'volume',
        'MA5', 'MA10', 'MA20',
        'RSI',
        'MACD', 'MACD_signal',
        'BB_upper', 'BB_lower',
        'Momentum', 'Volatility',
        'Volume_MA',
        'Price_Range', 'Prev_Direction',
        'Resistance', 'Support',
    )

    # Map each line to DataFrame column name
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('MA5', 'MA5'),
        ('MA10', 'MA10'),
        ('MA20', 'MA20'),
        ('RSI', 'RSI'),
        ('MACD', 'MACD'),
        ('MACD_signal', 'MACD_signal'),
        ('BB_upper', 'BB_upper'),
        ('BB_lower', 'BB_lower'),
        ('Momentum', 'Momentum'),
        ('Volatility', 'Volatility'),
        ('Volume_MA', 'Volume_MA'),
        ('Price_Range', 'Price_Range'),
        ('Prev_Direction', 'Prev_Direction'),
        ('Resistance', 'Resistance'),
        ('Support', 'Support'),
        # Backtrader internal parameters
        ('openinterest', None),
    )
