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
    """Load local historical data from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}. Download data first.")
    return pd.read_csv(path, index_col=0, parse_dates=True)

# ---------------------------------------------------------------------------
# Alpha Vantage FX helper

# ---------------------------------------------------------------------------

import requests
from dotenv import load_dotenv

load_dotenv()
ALPHA_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

ALPHA_URL = (
    "https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={from_symbol}"
    "&to_symbol={to_symbol}&interval={interval}&outputsize=full&apikey={apikey}"
)


def download_alpha_fx(from_symbol="EUR", to_symbol="USD", interval="60min", save_path="data/eurusd_alpha.csv"):
    """Download intraday FX data via Alpha Vantage and save locally.

    Parameters
    ----------
    from_symbol : str
        Base currency (e.g., 'EUR').
    to_symbol : str
        Quote currency (e.g., 'USD').
    interval : str
        One of '5min', '15min', '30min', '60min'.
    save_path : str
        CSV path for saving.
    Returns
    -------
    pd.DataFrame
    """
    if not ALPHA_KEY:
        raise EnvironmentError("ALPHAVANTAGE_API_KEY not set in environment/.env")

    url = ALPHA_URL.format(from_symbol=from_symbol, to_symbol=to_symbol, interval=interval, apikey=ALPHA_KEY)
    print(f"Downloading {from_symbol}/{to_symbol} {interval} data from Alpha Vantage …")
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    js = res.json()
    key = next(k for k in js.keys() if k.startswith("Time Series"))
    data = (
        pd.DataFrame.from_dict(js[key], orient="index")
        .rename(columns=lambda c: c.split()[-1])
        .astype(float)
        .sort_index()
    )
    data.index = pd.to_datetime(data.index)
    data.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}, inplace=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path)
    print(f"Saved Alpha Vantage data to {save_path}")
    return data


def download_alpha_fx_daily(from_symbol="EUR", to_symbol="USD", save_path="data/eurusd_daily_alpha.csv"):
    """Download daily FX data (FX_DAILY) from Alpha Vantage.

    API returns JSON with 'Time Series FX (Daily)'. We convert to DataFrame in OHLCV form.
    """
    if not ALPHA_KEY:
        raise EnvironmentError("ALPHAVANTAGE_API_KEY not set in environment/.env")

    url = (
        "https://www.alphavantage.co/query?function=FX_DAILY&from_symbol="
        f"{from_symbol}&to_symbol={to_symbol}&outputsize=full&apikey={ALPHA_KEY}"
    )
    print(f"Downloading {from_symbol}/{to_symbol} daily data from Alpha Vantage …")
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    js = res.json()
    if "Time Series FX (Daily)" not in js:
        raise ValueError(f"Alpha Vantage FX_DAILY missing data section: {js.get('Note') or js}")
    data = (
        pd.DataFrame.from_dict(js["Time Series FX (Daily)"], orient="index")
        .rename(columns=lambda c: c.split()[-1])
        .astype(float)
        .sort_index()
    )
    data.index = pd.to_datetime(data.index)
    data.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}, inplace=True)
    data["volume"] = 0.0  # Alpha FX daily has no volume; placeholder

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path)
    print(f"Saved Alpha Vantage daily data to {save_path}")
    return data
