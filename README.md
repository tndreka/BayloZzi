# 🚀 PROJECT ACE - Autonomous AI Trading Agent

## 📋 Project Overview
Building a sophisticated autonomous AI trading agent that learns, thinks, and improves like a professional trader using price data, economic events, news, and feedback loops.

## 🏁 Current Status: **Phase 2 - Real Strategy Simulator**

### ✅ **Completed Phases:**

#### **Phase 1 - Pattern Learner** ✅
- ✅ Learn price patterns using technical indicators (MA5, MA10, RSI, Volatility)
- ✅ Predict next-day move (UP/DOWN) using Random Forest
- ✅ Simulate trades and evaluate accuracy & returns
- ✅ Basic backtesting framework

#### **Phase 2 - Real Strategy Simulator** ✅
- ✅ Risk management system (2% risk per trade)
- ✅ Position sizing based on volatility
- ✅ Stop-loss and take-profit logic
- ✅ Comprehensive performance tracking
- ✅ Professional-grade backtesting
- ✅ Multiple file implementations for different approaches

## 📁 Project Structure

```
/BayloZzi/
├── ace_working.py          # Main Phase 2 implementation with full features
├── ace_lightning.py        # Optimized fast version for quick testing
├── lightning_results.json  # Latest trading results
├── data/
│   └── ace_trading_results.json  # Historical results storage
├── models/                 # Reserved for future ML models
└── README.md              # Project documentation
```

## 🔧 Key Files

### `ace_working.py` - **Main Trading Bot**
- Full-featured Phase 2 implementation
- Complete risk management system
- Comprehensive performance metrics
- Chart generation and visualization
- Professional backtesting engine

### `ace_lightning.py` - **Fast Testing Version**
- Lightweight implementation for rapid testing
- Core functionality without heavy visualization
- Quick validation and debugging
- Optimized for development cycles

## 📊 Current Performance
- **Dataset**: 117 days of EUR/USD forex data (2024)
- **Model**: Random Forest Classifier
- **Features**: MA5, MA10, RSI, Volatility
- **Test Results**: 24 predictions with varying accuracy
- **Risk Management**: 2% capital risk per trade
- **Status**: Ready for Phase 3 enhancement

## 🛠 Technical Stack
- **Data Source**: Yahoo Finance (yfinance)
- **ML Framework**: scikit-learn
- **GPU Support**: PyTorch with ROCm (AMD GPU ready)
- **Visualization**: matplotlib
- **Data Processing**: pandas, numpy

## 🚧 **Next Phase: Phase 3 - News & Economic Event Awareness**

### Planned Features:
- 📢 News API integration (NewsAPI.org)
- 🧠 NLP AI models (FinBERT/OpenAI GPT)
- 📅 Economic calendar integration
- 🔗 News sentiment analysis
- 🎯 Context-aware trading decisions

## 📈 **Future Roadmap:**

- **Phase 4**: Reinforcement Learning Brain
- **Phase 5**: Real-time Trading Assistant
- **Phase 6**: Strategic Thinking AI (Multi-agent logic)

## 🚀 **Getting Started**

```bash
# Install dependencies
pip install pandas numpy yfinance scikit-learn matplotlib

# Run main bot
python3 ace_working.py

# Run fast version
python3 ace_lightning.py
```

## 💡 **Key Achievements**
1. ✅ **Working AI Pipeline**: From data download to predictions
2. ✅ **Risk Management**: Professional-grade position sizing
3. ✅ **Performance Tracking**: Comprehensive metrics system
4. ✅ **Multiple Implementations**: Different approaches for various needs
5. ✅ **GPU Ready**: PyTorch with ROCm support for AMD GPUs

---

**Status**: Phase 2 Complete ✅ | Ready for Phase 3 Development 🚀
