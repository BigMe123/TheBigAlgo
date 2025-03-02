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

- **Full NASDAQ-100 Coverage**: Processes the complete NASDAQ-100 index, not just a subset of stocks
- **ML-Enhanced Trading**: Combines machine learning predictions with traditional technical analysis
- **Strategy Ranking**: Automatically identifies and prioritizes the most effective strategies
- **Adaptive Weighting**: Dynamically adjusts influence of different signals based on performance
- **Real-Time Monitoring**: Comprehensive logging and performance tracking
- **Market Regime Detection**: Adjusts behavior based on broader market conditions
- **Portfolio Protection**: Implements stop-loss and take-profit mechanisms to protect capital

## Installation

### Prerequisites

- Python 3.8 or higher
- MongoDB 4.4 or higher
- Alpaca trading account
- Polygon.io API key
- Financial Prep API key

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/alphasynth.git
cd alphasynth
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include:

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
```

### Step 3: Set up MongoDB

1. Install MongoDB following the [official instructions](https://docs.mongodb.com/manual/installation/)
2. Start the MongoDB service:
   ```bash
   sudo systemctl start mongod    # Linux
   brew services start mongodb    # macOS
   ```

### Step 4: Configure API keys

Create a `config.py` file in the root directory with your API keys:

```python
# API Keys
POLYGON_API_KEY = "your_polygon_api_key"
FINANCIAL_PREP_API_KEY = "your_financial_prep_api_key"
API_KEY = "your_alpaca_api_key"
API_SECRET = "your_alpaca_api_secret"
BASE_URL = "https://paper-api.alpaca.markets"  # Use paper trading initially

# MongoDB Configuration
MONGO_DB_USER = "your_mongodb_username"
MONGO_DB_PASS = "your_mongodb_password"
mongo_url = "mongodb://localhost:27017/"  # Local MongoDB
# mongo_url = f"mongodb+srv://{MONGO_DB_USER}:{MONGO_DB_PASS}@cluster0.mongodb.net/alphasynth"  # Atlas
```

## Usage

### Step 1: Initialize MongoDB Collections

For first-time setup, run the initialization script to create the required MongoDB collections:

```bash
python scripts/initialize_db.py
```

### Step 2: Start the Signal Integrator (ML Engine)

Start the ML prediction engine first:

```bash
python deployed_model/signal_integrator.py --interval 60
```

Parameters:
- `--interval`: Signal generation interval in seconds (default: 60)
- `--model_path`: Custom path to model directory (optional)
- `--mongo_url`: Custom MongoDB URL (optional, uses config.py by default)

### Step 3: Start the Ranking Client

Launch the ranking client to manage strategy rankings:

```bash
python ranking_client.py
```

### Step 4: Start the Trading Client

Finally, start the trading client to execute actual trades:

```bash
python trading_client.py
```

For production use, it's recommended to run each component in a separate terminal or as background services.

## How AlphaSynth Works

### 1. Data Acquisition and Processing

AlphaSynth automatically fetches data for all NASDAQ-100 stocks, including:
- Historical price data from yfinance
- Real-time quotes from Polygon.io
- Company fundamentals from Financial Prep API

### 2. ML Signal Generation

For each stock, the system:
1. Enhances data with advanced technical indicators
2. Detects market regimes (bull/bear/sideways)
3. Applies machine learning models to generate predictions
4. Calculates buying and selling signals with confidence scores

### 3. Strategy Ranking

The ranking client:
1. Simulates trades for each strategy on historical data
2. Tracks performance metrics (win rate, return, drawdown)
3. Awards points based on successful predictions
4. Dynamically adjusts strategy weights based on performance

### 4. Decision Making

For each NASDAQ-100 stock, the trading client:
1. Retrieves ML signals from the integrator
2. Gets recommendations from each ranked strategy
3. Applies a weighted majority decision algorithm
4. Prioritizes trades based on conviction and potential return

### 5. Trade Execution

Based on the final decisions:
1. Buy orders are prioritized by a heap-based algorithm
2. Sell orders are executed immediately when signals are strong
3. Portfolio constraints ensure proper diversification
4. Stop-loss and take-profit levels are set for risk management

## Performance Monitoring

Monitor system performance through:

### Log Files

- `system.log`: Trading client activity
- `rank_system.log`: Ranking client simulations
- `amplify_signal.log`: Signal integrator predictions

### MongoDB Collections

You can query MongoDB directly to monitor system state:

```bash
# Connect to MongoDB
mongo

# View current portfolio
db.trades.assets_quantities.find()

# View strategy rankings
db.trading_simulator.rank.find().sort({rank:1})

# View latest ML signals
db.trading_signals.signals.find()
```

## Troubleshooting

### Common Issues

**"No signals available" Warning**
- This is normal during initial startup
- The system needs a few minutes to generate the first signals
- Verify MongoDB connection if this persists after 10 minutes

**MongoDB Connection Errors**
- Ensure MongoDB service is running: `systemctl status mongod`
- Check MongoDB URL in config.py
- Verify network connectivity to MongoDB Atlas if using cloud deployment

**API Rate Limiting**
- Adjust intervals if you encounter rate limiting
- Consider upgrading API plans for production use

### Advanced Configurations

For advanced users, you can adjust system behavior by modifying:

- `SIGNAL_WEIGHT` in trading_client.py: Controls ML signal influence (0.0-1.0)
- `SIGNAL_INFLUENCE` in ranking_client.py: Controls ML impact on strategy rankings
- Monitoring interval: Use `--interval` parameter when starting signal_integrator.py

## Production Deployment

For production use:

1. Use systemd services (Linux) or launchd agents (macOS) to run components as background services
2. Set up a monitoring dashboard using MongoDB charts
3. Implement automated notifications via email/SMS for critical events
4. Use real trading endpoints instead of paper trading

## License

AlphaSynth is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

AlphaSynth is provided for informational purposes only. It is not financial advice and should not be used as the basis for any financial decisions. Trading securities involves risk, and past performance is not indicative of future results.

---

© 2025 AlphaSynth Trading Systems