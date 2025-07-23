# Quick demo for testing the forex trading system
import backtrader as bt
from core.data_loader import download_alpha_fx_daily
from core.features import add_features
from core.model import load_model
from core.bt_feed import FeatureData
from core.strategy import AceStrategy

def run_quick_demo():
    print("Starting Forex Trading System Demo...")
    print("=" * 60)
    
    # === Step 1: Load and prepare data ===
    print("Loading EUR/USD historical data...")
    daily = download_alpha_fx_daily()
    hourly = daily.resample('1h').ffill()  # Use lowercase 'h'
    df = add_features(hourly)
    
    # === Step 2: Load ML model ===
    print("Loading machine learning model...")
    model = load_model()
    
    # Use only last 1000 hours for quick demo (about 40 days)
    print("Preparing last 1000 hours of data for quick backtest...")
    test_df = df.tail(1000).copy()
    feed = FeatureData(dataname=test_df)
    
    # === Step 3: Configure Cerebro with realistic capital ===
    cerebro = bt.Cerebro()
    
    # Start with $10,000 for realistic trading
    initial_cash = 10000
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0002)  # 0.02% commission
    cerebro.broker.set_slippage_perc(0.0001)  # 0.01% slippage
    
    cerebro.adddata(feed)
    
    # Add strategy with model
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
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # === Step 4: Run Backtest ===
    print("\nRunning backtest simulation...")
    print("=" * 60)
    
    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    # === Step 5: Display Results ===
    strat = results[0]
    
    print(f"\nBACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Net Profit/Loss: ${final_value - initial_cash:,.2f}")
    print(f"Return: {((final_value - initial_cash) / initial_cash) * 100:.2f}%")
    
    # Trade analysis
    trades = strat.analyzers.trades.get_analysis()
    if trades.total.total > 0:
        print(f"\nTRADE STATISTICS")
        print("=" * 60)
        print(f"Total Trades: {trades.total.total}")
        print(f"Winning Trades: {trades.won.total}")
        print(f"Losing Trades: {trades.lost.total}")
        if trades.won.total > 0:
            print(f"Win Rate: {(trades.won.total / trades.total.total) * 100:.2f}%")
            print(f"Average Win: ${trades.won.pnl.average:.2f}")
        if trades.lost.total > 0:
            print(f"Average Loss: ${abs(trades.lost.pnl.average):.2f}")
    
    # Risk metrics
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"\nRISK METRICS")
    print("=" * 60)
    print(f"Maximum Drawdown: {drawdown.max.drawdown:.2f}%")
    print(f"Longest Drawdown Period: {drawdown.max.len} bars")
    
    # Sharpe ratio
    sharpe = strat.analyzers.sharpe.get_analysis()
    if sharpe.get('sharperatio') is not None:
        print(f"Sharpe Ratio: {sharpe['sharperatio']:.2f}")
    
    print("\nDemo completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    run_quick_demo()