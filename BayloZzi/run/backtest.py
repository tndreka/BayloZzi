# run/backtest.py

import backtrader as bt
from core.data_loader import download_alpha_fx_daily
from core.features import add_features
from core.model import load_model
from core.bt_feed import FeatureData
from core.strategy import AceStrategy

def run_backtest():
    # === Step 1: Load and prepare data ===
    daily = download_alpha_fx_daily()
    hourly = daily.resample('1H').ffill()
    df = add_features(hourly)

    # === Step 2: Load existing ML model ===
    model = load_model()

    # Use most recent 10k rows for backtest
    test_df = df.tail(10000).copy()
    feed = FeatureData(dataname=test_df)

    # === Step 4: Configure Cerebro ===
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.broker.set_slippage_perc(0.001)
    cerebro.adddata(feed)

    cerebro.addstrategy(
        AceStrategy,
        model=model,
        features=[
            'open','high','low','close','volume',
            'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'Momentum', 'Volatility',
            'Volume_MA', 'Price_Range', 'Prev_Direction',
            'Resistance', 'Support'
        ]
    )

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # === Step 5: Run Backtest ===
    print("\nStarting backtest...")
    results = cerebro.run()
    strat = results[0]

    print("\n=== RESULTS ===")
    print("Final Portfolio Value: $%.2f" % cerebro.broker.getvalue())
    print("Sharpe Ratio:", strat.analyzers.sharpe.get_analysis())
    print("Drawdown:", strat.analyzers.drawdown.get_analysis())
    print("Trade Analysis:", strat.analyzers.trades.get_analysis())

    # Plotting usually requires a GUI backend (tkinter). Skip if not available.
    try:
        cerebro.plot(style='candlestick')
    except ModuleNotFoundError:
        print("Plot skipped: GUI backend not available in this environment.")

if __name__ == "__main__":
    run_backtest()
