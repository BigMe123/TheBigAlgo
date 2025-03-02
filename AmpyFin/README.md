# TheBigAlgo

An advanced algorithmic trading system leveraging machine learning, news sentiment analysis, and multi-strategy optimization for intelligent trading decisions.

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Signal         │     │  Ranking        │     │  Trading        │     │  News           │
│  Integrator     │◄────┤  Client         │◄────┤  Client         │◄────┤  Sentiment      │
│  (ML Engine)    │─────►                 │─────►                 │─────►  Analyzer       │
│                 │     │                 │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │                       │
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                MongoDB (Centralized Data Management)                         │
│  Collections:                                                                               │
│  - trading_signals     - market_sentiment    - trades               - HistoricalDatabase    │
│  - ranking_decisions   - algorithm_holdings  - signal_weights       - market_data           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                       Trading Platforms & Accounts                                          │
│  - Alpaca             - Interactive Brokers   - Paper Trading                               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

1. **Signal Integrator (ML Engine)**
   - Generates predictive signals using machine learning
   - Combines multiple data sources and models
   - Adaptive signal generation

2. **Ranking Client**
   - Simulates and ranks trading strategies
   - Dynamically adjusts strategy weights
   - Continuous performance optimization

3. **Trading Client**
   - Executes trades based on combined signals
   - Implements risk management
   - Real-time trade decision-making

4. **News Sentiment Analyzer**
   - Aggregates news from multiple sources
   - Computes sentiment scores
   - Provides additional market insight

## Key Features

- **Multi-Source Signal Generation**
  - Machine Learning Predictions
  - Technical Analysis Strategies
  - News Sentiment Analysis
  - Market Regime Detection

- **Adaptive Strategy Weighting**
  - Dynamic strategy performance tracking
  - Automated strategy ranking
  - Continuous learning and optimization

- **Advanced Risk Management**
  - Portfolio diversification
  - Intelligent position sizing
  - Stop-loss and take-profit mechanisms

## Prerequisites

- Python 3.8+
- MongoDB
- API Keys:
  - Alpaca
  - Polygon.io
  - Alpha Vantage
  - NewsAPI
  - Finnhub
  - Financial Prep

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/thebigalgo.git
cd thebigalgo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configuration

Create a `config.py`:

```python
# Trading Platform Keys
API_KEY = "your_alpaca_api_key"
API_SECRET = "your_alpaca_api_secret"
BASE_URL = "https://paper-api.alpaca.markets"

# Data Source API Keys
POLYGON_API_KEY = "your_polygon_api_key"
ALPHA_VANTAGE_KEY = "your_alpha_vantage_key"
NEWSAPI_KEY = "your_newsapi_key"
FINNHUB_KEY = "your_finnhub_key"

# MongoDB Configuration
MONGO_DB_USER = "your_mongodb_username"
MONGO_DB_PASS = "your_mongodb_password"
MONGO_URL = "mongodb://localhost:27017/"
```

## Running the System

### 1. Start Signal Integrator

```bash
python deployed_model/signal_integrator.py
```

### 2. Start Ranking Client

```bash
python ranking_client.py
```

### 3. Start Trading Client

```bash
python trading_client.py
```

## Monitoring

- Comprehensive logging
- MongoDB-based performance tracking
- Real-time signal and trade monitoring

## Important Considerations

⚠️ **DISCLAIMER**: 
This software is for educational purposes only. Trading involves significant financial risk. Always consult a financial advisor before making investment decisions.

## License

Ownership proprietary 

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

© 2025 TheBigAlgo Trading Systems
