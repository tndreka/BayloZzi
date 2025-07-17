# run/backtest.py

import backtrader as bt
from core.data_loader import download_data
from core.features import add_features
from core.model import train_enhanced_model
from core.strategy import AceStrategy

def run_backtest():
    # === Step 1: Load and prepare data ===
    df = download_data()             # Download EURUSD 1h data
    df = add_features(df)            # Add technical indicators

    # === Step 2: Train ML model ===
    model, X_test, y_test, preds, preds_proba = train_enhanced_model(df)

    # === Step 3: Prepare Backtrader feed ===
    test_df = df.iloc[-len(X_test):].copy()
    feed = bt.feeds.PandasData(dataname=test_df)

    # === Step 4: Configure Cerebro ===
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.broker.set_slippage_perc(0.001)
    cerebro.adddata(feed)

    cerebro.addstrategy(
        AceStrategy,
        model=model,
        features=[
            'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'Momentum', 'Volatility',
            'Volume_MA', 'Price_Range', 'Prev_Direction',
            'Resistance', 'Support'
        ]
    )

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # === Step 5: Run Backtest ===
    print("\nStarting backtest...")
    results = cerebro.run()
    strat = results[0]

    print("\n=== RESULTS ===")
    print("Final Portfolio Value: $%.2f" % cerebro.broker.getvalue())
    print("Sharpe Ratio:", strat.analyzers.sharpe.get_analysis())
    print("Drawdown:", strat.analyzers.drawdown.get_analysis())

    cerebro.plot(style='candlestick')

if __name__ == "__main__":
    run_backtest()
