# core/data_loader.py

import pandas as pd
import yfinance as yf
import os

def download_data(symbol='EURUSD=X', start='2020-01-01', end='2024-01-01', interval='1h', save_path='data/historical.csv'):
    """
    Download historical forex data from Yahoo Finance and save it locally.
    
    Parameters:
        symbol (str): Forex pair (e.g., 'EURUSD=X' for EUR/USD).
        start (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).
        interval (str): Time interval ('1h', '1d', etc.).
        save_path (str): File path to save the data.
    
    Returns:
        pd.DataFrame: Historical price data.
    """
    print(f"Downloading {symbol} data from Yahoo Finance...")
    df = yf.download(symbol, start=start, end=end, interval=interval)

    if df.empty:
        raise ValueError("Downloaded data is empty. Check the symbol or internet connection.")

    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.dropna(inplace=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(f"Saved data to {save_path}")
    return df


def load_local_data(path='data/historical.csv'):
    """
    Load local historical data from CSV.
    
    Returns:
        pd.DataFrame: Loaded historical price data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}. Download data first.")
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df
