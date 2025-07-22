import os

# General config values
DATA_PATH = os.getenv('ACE_DATA_PATH', 'data/historical.csv')
MODEL_PATH = os.getenv('ACE_MODEL_PATH', 'models/model.pkl')
NEWS_PATH = os.getenv('ACE_NEWS_PATH', 'data/news_sentiment.json')

# Trading parameters
def get_trading_config():
    return {
        'initial_cash': 10000,
        'risk_per_trade': 0.02,
        'reward_risk': 1.5,
        'commission': 0.0005,
        'slippage': 0.001,
    }
