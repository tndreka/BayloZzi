# core/features.py

import pandas as pd
import numpy as np
import ta  # You may need to install with: pip install ta

def add_features(df):
    """
    Add technical indicators and features to the price DataFrame.
    Assumes OHLCV structure: ['open', 'high', 'low', 'close', 'volume'].

    Returns:
        pd.DataFrame: DataFrame with new features and 'Signal' column.
    """

    # Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()

    # Momentum
    df['Momentum'] = ta.momentum.ROCIndicator(close=df['close'], window=5).roc()

    # Volatility
    df['Volatility'] = df['close'].rolling(window=10).std()

    # Volume Moving Average
    df['Volume_MA'] = df['volume'].rolling(window=10).mean()

    # Price Range (High - Low)
    df['Price_Range'] = df['high'] - df['low']

    # Previous Candle Direction
    df['Prev_Direction'] = np.where(df['close'].shift(1) > df['open'].shift(1), 1, -1)

    # Support & Resistance (basic pivot points)
    df['Resistance'] = (df['high'].shift(1) + df['close'].shift(1)) / 2
    df['Support'] = (df['low'].shift(1) + df['close'].shift(1)) / 2

    # Generate Trade Signal — simple logic for now (can be refined)
    df['Signal'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    # Drop NaNs and keep original datetime index
    df = df.dropna()

    return df
