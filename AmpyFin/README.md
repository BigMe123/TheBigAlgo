# AlphaSynth

AlphaSynth is an integrated algorithmic trading system that combines traditional technical analysis strategies with machine learning signal generation to make optimized trading decisions on the NASDAQ-100 stocks.

![AlphaSynth Banner](https://via.placeholder.com/1200x300/0073e6/ffffff?text=AlphaSynth+Trading+System)

## Overview

AlphaSynth consists of three core components working in harmony:

1. **Signal Integrator (ML Engine)** - Generates buy/sell predictions using machine learning
2. **Trading Client** - Executes real-time trades based on combined signals
3. **Ranking Client** - Simulates trades, ranks strategies, and continuously optimizes system performance

The system automatically processes all NASDAQ-100 stocks, making intelligent trading decisions by synthesizing multiple signal sources.

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Signal         │     │  Ranking        │     │  Trading        │
│  Integrator     │◄────┤  Client         │◄────┤  Client         │
│  (ML Engine)    │─────►                 │─────►                 │
│                 │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                           MongoDB                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                       Trading Accounts                          │
│                     (Alpaca, Interactive                        │
│                      Brokers, etc.)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Full NASDAQ-100 Coverage**: Processes the complete NASDAQ-100 index
- **ML-Enhanced Trading**: Combines machine learning predictions with traditional technical analysis
- **Strategy Ranking**: Automatically identifies and prioritizes the most effective strategies
- **Adaptive Weighting**: Dynamically adjusts influence of different signals based on performance
- **Real-Time Monitoring**: Comprehensive logging and performance tracking
- **Market Regime Detection**: Adjusts behavior based on broader market conditions
- **Portfolio Protection**: Implements stop-loss and take-profit mechanisms

## Prerequisites

- Python 3.8+
- MongoDB 4.4+
- Alpaca trading account
- API keys for:
  - Polygon.io
  - Financial Prep
  - Alpha Vantage
  - NewsAPI
  - Finnhub

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/alphasynth.git
cd alphasynth
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` with these dependencies:

```
polygon-api-client
pymongo
yfinance
alpaca-py
pandas
numpy
certifi
nltk
scikit-learn
matplotlib
requests
schedule
joblib
tensorflow
scipy
```

### 3. Configure API Keys

Create a `config.py` file:

```python
# API Keys
POLYGON_API_KEY = "your_polygon_api_key"
FINANCIAL_PREP_API_KEY = "your_financial_prep_api_key"
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key"
NEWSAPI_KEY = "your_newsapi_key"
FINNHUB_API_KEY = "your_finnhub_key"

# Alpaca Trading
API_KEY = "your_alpaca_api_key"
API_SECRET = "your_alpaca_api_secret"
BASE_URL = "https://paper-api.alpaca.markets"

# MongoDB Configuration
MONGO_DB_USER = "your_mongodb_username"
MONGO_DB_PASS = "your_mongodb_password"
mongo_url = "mongodb://localhost:27017/"  # Local MongoDB
```

## Usage

### Start Signal Integrator

```bash
python deployed_model/signal_integrator.py --interval 60
```

### Start Ranking Client

```bash
python ranking_client.py
```

### Start Trading Client

```bash
python trading_client.py
```

## Key Components

- `signal_integrator.py`: Machine learning signal generation
- `ranking_client.py`: Strategy performance tracking
- `trading_client.py`: Trade execution
- `news_analyzer.py`: Sentiment analysis from multiple news sources
- `market_regime_detector.py`: Market condition analysis

## Performance Monitoring

Monitor through:
- Log files (`system.log`, `rank_system.log`, `amplify_signal.log`)
- MongoDB collections

## Troubleshooting

- Verify MongoDB connection
- Check API key configurations
- Adjust monitoring intervals if needed

## Warning ⚠️

**Disclaimer**: AlphaSynth is for educational purposes. Trading involves significant financial risk. Always consult with a financial advisor before making investment decisions.

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Project Link: [https://github.com/yourusername/alphasynth](https://github.com/yourusername/alphasynth)

---

© 2025 AlphaSynth Trading Systems
