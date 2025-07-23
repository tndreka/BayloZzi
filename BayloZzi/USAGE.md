# Forex Multi-Agent Trading System - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Running Backtests](#running-backtests)
3. [Live Trading](#live-trading)
4. [Agent System](#agent-system)
5. [Configuration Guide](#configuration-guide)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Getting Started

### Prerequisites
- Python 3.11 or higher installed
- Alpha Vantage API key (free from https://www.alphavantage.co)
- Basic understanding of forex trading concepts

### Quick Installation

1. **Install the package:**
```bash
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

3. **Run your first backtest:**
```bash
python -m run.backtest
```

## Running Backtests

### Basic Backtest
The simplest way to test the trading strategy:

```bash
python -m run.backtest
```

This will:
- Load historical EUR/USD data
- Apply technical indicators
- Run the ML model predictions
- Execute trades based on signals
- Display performance metrics

### Advanced Backtest Options

You can modify the backtest parameters by editing `run/backtest.py`:

```python
# Change the trading pair
symbol = "GBPUSD"  # Default is EURUSD

# Adjust the time period
test_df = df.tail(20000)  # Use last 20,000 hourly candles

# Modify initial capital
cerebro.broker.setcash(10000)  # Start with $10,000

# Adjust position sizing
cerebro.addsizer(bt.sizers.PercentSizer, percents=10)  # Risk 10% per trade
```

### Understanding Backtest Results

The backtest will show:
- **Starting Portfolio Value**: Initial capital
- **Final Portfolio Value**: Ending capital
- **Total Return**: Percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Trade Analysis**: Number of trades, average profit/loss

## Live Trading

⚠️ **WARNING**: Live trading involves real money and risk. Start with paper trading first!

### Paper Trading (Recommended)
Test the system with simulated trades:

```bash
# Set in .env
ENABLE_PAPER_TRADING=true
ENABLE_LIVE_TRADING=false

# Run live trading in paper mode
python -m run.live_trade
```

### Live Trading Setup

1. **Configure broker credentials** (in .env):
```env
BROKER_API_KEY=your-broker-api-key
BROKER_API_SECRET=your-broker-api-secret
BROKER_ACCOUNT_ID=your-account-id
```

2. **Set risk parameters**:
```env
MAX_RISK_PERCENT=1.0          # Risk only 1% per trade
MAX_DAILY_LOSS_PERCENT=3.0    # Stop trading after 3% daily loss
POSITION_SIZE=0.01            # Trade 0.01 lots (micro lot)
```

3. **Enable live trading**:
```env
ENABLE_LIVE_TRADING=true
DRY_RUN=false  # Set to true for final safety check
```

4. **Run live trading**:
```bash
python -m run.live_trade
```

## Agent System

The multi-agent system provides specialized analysis:

### Running Individual Agents

```bash
# Risk Management Agent
python -m BayloZzi.main --service risk-manager

# Technical Analysis Agent
python -m BayloZzi.main --service chart-analyzer

# Market Trend Agent
python -m BayloZzi.main --service trend-identifier

# News Sentiment Agent
python -m BayloZzi.main --service news-sentiment

# Economic Data Agent
python -m BayloZzi.main --service economic-factors
```

### Running All Agents with Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Agent Coordination

The agents work together:
1. **Data Manager** fetches and stores market data
2. **Chart Analyzer** identifies technical patterns
3. **Trend Identifier** confirms market direction
4. **News Sentiment** checks for market-moving news
5. **Economic Factors** monitors economic calendars
6. **Risk Manager** sizes positions and sets stops

## Configuration Guide

### Essential Settings

#### API Keys
```env
ALPHAVANTAGE_API_KEY=demo     # Required for market data
NEWS_API_KEY=your-key         # Optional for news sentiment
OPENAI_API_KEY=your-key       # Optional for AI analysis
```

#### Trading Parameters
```env
TRADING_SYMBOLS=EURUSD,GBPUSD  # Pairs to trade
TRADING_INTERVAL=1h             # Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
MAX_POSITIONS=3                 # Maximum concurrent trades
```

#### Risk Management
```env
STOP_LOSS_PIPS=30              # Stop loss distance
TAKE_PROFIT_PIPS=60            # Take profit target
MAX_RISK_PERCENT=1.0           # Risk per trade
RISK_REWARD_RATIO=2.0          # Minimum R:R ratio
```

### Performance Tuning

#### For Faster Backtesting
```env
BATCH_SIZE=5000                # Process more data at once
CACHE_TTL_SECONDS=600          # Cache data longer
LOG_LEVEL=WARNING              # Reduce logging overhead
```

#### For Live Trading
```env
HEALTH_CHECK_INTERVAL=30       # More frequent health checks
LOG_LEVEL=INFO                 # Detailed logging
ENABLE_METRICS=true            # Performance monitoring
```

## Troubleshooting

### Common Issues

#### "No module named 'talib'"
The system uses the 'ta' library instead of TA-Lib for easier installation.

#### "API rate limit exceeded"
- Use a paid Alpha Vantage subscription
- Increase `CACHE_TTL_SECONDS` to cache data longer
- Reduce `TRADING_SYMBOLS` to fewer pairs

#### "Order Canceled/Margin/Rejected"
- Increase initial capital in backtest
- Reduce `POSITION_SIZE` percentage
- Check margin requirements for your broker

#### "Connection refused" errors
- Ensure Redis and PostgreSQL are running (if using)
- Check firewall settings
- Verify service health with `docker-compose ps`

### Debug Mode

Enable detailed debugging:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

Check logs:
```bash
tail -f logs/trading.log
tail -f logs/errors.log
```

## Best Practices

### For Beginners
1. Start with backtesting only
2. Use default settings initially
3. Trade only EURUSD to start
4. Risk maximum 1% per trade
5. Run paper trading for at least 1 month

### For Production Use
1. Use Docker deployment for stability
2. Set up monitoring and alerts
3. Implement daily loss limits
4. Regular backup of models and data
5. Monitor system performance metrics

### Risk Management Rules
1. Never risk more than 2% per trade
2. Set daily loss limit at 5%
3. Use stop losses on every trade
4. Maintain risk/reward ratio above 1.5
5. Don't overtrade - quality over quantity

### System Maintenance
1. Update dependencies monthly
2. Retrain models quarterly
3. Review and adjust parameters based on performance
4. Monitor API usage and costs
5. Keep logs for audit trail

## Getting Help

### Resources
- Check `logs/errors.log` for detailed error messages
- Review the README.md for setup instructions
- Examine example configurations in `.env.example`

### Performance Monitoring
Track these key metrics:
- Win rate (target > 50%)
- Average risk/reward (target > 1.5)
- Maximum drawdown (keep < 20%)
- Sharpe ratio (target > 1.0)
- Monthly returns consistency

Remember: Past performance does not guarantee future results. Always trade responsibly!