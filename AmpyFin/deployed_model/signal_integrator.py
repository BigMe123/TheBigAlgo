#!/usr/bin/env python
# AmplifySignalIntegrator - Integrated Trading Signal Generator
import os
import sys
import logging
import pandas as pd
import numpy as np
import importlib.util
from datetime import datetime, timedelta
import traceback
import time
import threading
from pymongo import MongoClient
import certifi
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("amplify_signal.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AmplifySignal")

class AmplifySignalIntegrator:
    """
    Integrated signal generator that works alongside the trading and ranking clients.
    Automatically generates and stores signals for stocks being analyzed.
    """
    
    def __init__(self, model_path=None, mongo_url=None):
        """
        Initialize the trading signal integrator.
        
        Args:
            model_path: Path to the deployed model directory (default: Current directory)
            mongo_url: MongoDB connection URL (default: load from config)
        """
        # Set model_path to current directory if not provided
        self.model_path = model_path or os.getcwd()
        
        # Ensure model path exists
        if not os.path.exists(self.model_path):
            try:
                os.makedirs(self.model_path)
                logger.info(f"Created model path directory: {self.model_path}")
            except Exception as e:
                logger.error(f"Could not create model path: {e}")
                raise FileNotFoundError(f"Model path '{self.model_path}' does not exist and could not be created")
        
        # Add model path to Python path to enable imports
        sys.path.append(os.path.dirname(self.model_path))
        
        # Load MongoDB connection URL from config if not provided
        if mongo_url is None:
            try:
                # Try to import from local_config.py
                sys.path.append(self.model_path)
                try:
                    # First, check if local_config.py exists in the model path
                    config_path = os.path.join(self.model_path, "local_config.py")
                    if os.path.exists(config_path):
                        spec = importlib.util.spec_from_file_location("local_config", config_path)
                        local_config = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(local_config)
                        self.mongo_url = getattr(local_config, "mongo_url", None)
                        if self.mongo_url:
                            logger.info("Loaded MongoDB URL from local_config")
                        else:
                            logger.warning("mongo_url not found in local_config.py")
                    else:
                        # Create default local_config with empty mongo_url
                        with open(config_path, 'w') as f:
                            f.write("# MongoDB connection URL\n")
                            f.write("mongo_url = ''\n")
                        logger.info(f"Created empty local_config.py at {config_path}")
                        self.mongo_url = None
                except Exception as e:
                    logger.warning(f"Error loading local_config: {e}")
                    self.mongo_url = None
            except Exception as e:
                logger.warning(f"Could not load MongoDB URL from config: {e}")
                self.mongo_url = None
        else:
            self.mongo_url = mongo_url
        
        # Initialize components
        self.feature_engineer = None
        self.ensemble_model = None
        self.regime_detector = None
        self.hft_analyzer = None
        self.rl_weight_generator = None
        
        # MongoDB client
        self.mongo_client = None
        
        # Threading
        self.stop_event = threading.Event()
        self.monitor_thread = None
        
        # Create required directories
        self._ensure_model_directories()
        
        # Load components
        self._load_components()
        
        # Connect to MongoDB
        self._connect_mongodb()
        
        # Initialize signals storage
        self.signals_storage = {}
        
        # Last processed tickers cache for integration
        self.last_processed_tickers = set()
        
        logger.info("AmplifySignalIntegrator initialized successfully")
    
    def _ensure_model_directories(self):
        """Create necessary model directories if they don't exist."""
        model_dirs = [
            os.path.join(self.model_path, "models"),
            os.path.join(self.model_path, "models/ensemble"),
            os.path.join(self.model_path, "models/regimes"),
            os.path.join(self.model_path, "models/rl_weights")
        ]
        
        for directory in model_dirs:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    logger.info(f"Created directory: {directory}")
                except Exception as e:
                    logger.warning(f"Could not create directory {directory}: {e}")
    
    def _load_components(self):
        """Load and initialize the trading system components."""
        try:
            # Try to import advanced_feature_engineering
            try:
                sys.path.append(self.model_path)
                
                # Try direct import first since we've already added model_path to sys.path
                spec = importlib.util.find_spec("advanced_feature_engineering", [self.model_path])
                if spec:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.feature_engineer = module.AdvancedFeatureEngineering()
                    logger.info("Loaded AdvancedFeatureEngineering from direct import")
                else:
                    # Try with relative import
                    try:
                        from advanced_feature_engineering import AdvancedFeatureEngineering
                        self.feature_engineer = AdvancedFeatureEngineering()
                        logger.info("Loaded AdvancedFeatureEngineering from relative import")
                    except ImportError:
                        logger.warning("AdvancedFeatureEngineering not found, falling back to basic features")
                        # Create a simple stub class
                        class BasicFeatureEngineering:
                            def add_all_features(self, data):
                                return data
                        self.feature_engineer = BasicFeatureEngineering()
                        logger.info("Created basic feature engineering stub")
            except Exception as e:
                logger.warning(f"Error loading feature engineering: {e}")
                # Create a simple stub class
                class BasicFeatureEngineering:
                    def add_all_features(self, data):
                        return data
                self.feature_engineer = BasicFeatureEngineering()
                logger.info("Created basic feature engineering stub")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            logger.error(traceback.format_exc())
    
    def _connect_mongodb(self):
        """Connect to MongoDB."""
        if not self.mongo_url:
            logger.warning("MongoDB URL not available, using in-memory signal storage")
            # Create an in-memory signal storage
            self.signals_storage = {}
            return False
        
        try:
            # Connect to MongoDB
            ca = certifi.where()
            self.mongo_client = MongoClient(self.mongo_url, tlsCAFile=ca)
            
            # Create signal collection if it doesn't exist
            db = self.mongo_client.trading_signals
            if 'signals' not in db.list_collection_names():
                db.create_collection('signals')
                logger.info("Created signals collection in MongoDB")
            
            # Test connection
            db.signals.find_one()
            logger.info("Connected to MongoDB successfully")
            return True
        except Exception as e:
            logger.warning(f"Error connecting to MongoDB: {e}. Using in-memory signal storage.")
            # Create an in-memory signal storage
            self.signals_storage = {}
            self.mongo_client = None
            return False
    
    def _get_ticker_data(self, ticker, period='1y'):
        """
        Fetch historical data for a ticker using either MongoDB cache or yfinance.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period to fetch ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', etc.)
            
        Returns:
            DataFrame with historical stock data
        """
        try:
            # Try to get data from MongoDB cache first
            if self.mongo_client:
                db = self.mongo_client.HistoricalDatabase
                collection = db.HistoricalDatabase
                
                # Look for recent data in cache
                cached_data = collection.find_one({
                    "ticker": ticker,
                    "period": period,
                    "timestamp": {"$gt": datetime.now() - timedelta(hours=12)}
                })
                
                if cached_data and 'data' in cached_data:
                    # Convert dict to DataFrame
                    df_data = pd.DataFrame(cached_data['data'])
                    logger.info(f"Using cached data for {ticker} from MongoDB")
                    return df_data
            
            # If not in cache or no MongoDB, fetch from yfinance
            try:
                logger.info(f"Fetching {ticker} data from yfinance for period {period}")
                ticker_data = yf.Ticker(ticker)
                historical_data = ticker_data.history(period=period)
                
                # Reset index to make date a column
                historical_data = historical_data.reset_index()
                
                # Rename columns to match expected format
                if 'Date' in historical_data.columns:
                    historical_data = historical_data.rename(columns={'Date': 'Timestamp'})
                
                # Add Symbol column
                historical_data['Symbol'] = ticker
                
                # Store in MongoDB for future use if available
                if self.mongo_client:
                    collection.update_one(
                        {"ticker": ticker, "period": period},
                        {"$set": {
                            "data": historical_data.to_dict('records'),
                            "timestamp": datetime.now()
                        }},
                        upsert=True
                    )
                
                return historical_data
            except Exception as e:
                logger.error(f"Error fetching {ticker} data from yfinance: {e}")
                
                # Create synthetic data for testing/fallback
                logger.warning(f"Creating synthetic data for {ticker}")
                dates = pd.date_range(end=datetime.now(), periods=100)
                data = pd.DataFrame({
                    'Timestamp': dates,
                    'Symbol': ticker,
                    'Open': np.random.randn(100) * 10 + 100,
                    'High': np.random.randn(100) * 10 + 105,
                    'Low': np.random.randn(100) * 10 + 95,
                    'Close': np.random.randn(100) * 10 + 100,
                    'Volume': np.random.randint(1000, 1000000, 100),
                })
                
                # Create more realistic price sequence
                for i in range(1, len(data)):
                    data.loc[i, 'Close'] = data.loc[i-1, 'Close'] * (1 + np.random.normal(0, 0.01))
                    data.loc[i, 'Open'] = data.loc[i-1, 'Close'] * (1 + np.random.normal(0, 0.005))
                    data.loc[i, 'High'] = max(data.loc[i, 'Open'], data.loc[i, 'Close']) * (1 + abs(np.random.normal(0, 0.003)))
                    data.loc[i, 'Low'] = min(data.loc[i, 'Open'], data.loc[i, 'Close']) * (1 - abs(np.random.normal(0, 0.003)))
                
                return data
                
        except Exception as e:
            logger.error(f"Error in _get_ticker_data for {ticker}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_signal(self, ticker=None, data=None):
        """
        Generate a trading signal for the given ticker or data.
        
        Args:
            ticker: Stock ticker symbol (required if data is None)
            data: Optional pre-loaded DataFrame with stock data
            
        Returns:
            dict: Signal information including action, probability, and confidence
        """
        try:
            # Validate inputs
            if data is None and ticker is None:
                logger.error("Either ticker or data must be provided")
                return {"error": "Either ticker or data must be provided"}
            
            # Get ticker from data if not provided
            if ticker is None and data is not None:
                if isinstance(data, pd.DataFrame) and 'Symbol' in data.columns:
                    ticker = data['Symbol'].iloc[0]
                elif isinstance(data, list) and data and 'Symbol' in data[0]:
                    ticker = data[0]['Symbol']
                else:
                    logger.error("No ticker provided and no Symbol column in data")
                    return {"error": "No ticker provided and no Symbol column in data"}
            
            logger.info(f"Generating signal for {ticker}")
            
            # If data is a list (from MongoDB), convert to DataFrame
            if isinstance(data, list):
                data = pd.DataFrame(data)
            
            # If no data is provided, fetch it
            if data is None:
                # Get ideal period from indicators database if available
                period = '1y'  # Default period
                try:
                    if self.mongo_client:
                        indicator_tb = self.mongo_client.IndicatorsDatabase
                        if 'Indicators' in indicator_tb.list_collection_names():
                            indicator_collection = indicator_tb.Indicators
                            period_doc = indicator_collection.find_one({'indicator': 'AmplifySignalIntegrator'})
                            if period_doc and 'ideal_period' in period_doc:
                                period = period_doc['ideal_period']
                except Exception as e:
                    logger.warning(f"Error getting ideal period: {e}. Using default 1y.")
                
                data = self._get_ticker_data(ticker, period)
            
            if data is None or len(data) == 0:
                logger.error(f"No data available for {ticker}")
                return {"error": f"No data available for {ticker}"}
            
            # Convert column names if needed
            if 'Date' in data.columns and 'Timestamp' not in data.columns:
                data = data.rename(columns={'Date': 'Timestamp'})
            
            # Ensure Symbol column exists
            if 'Symbol' not in data.columns:
                data['Symbol'] = ticker
            
            # Enhance features if feature engineering is available
            if self.feature_engineer is not None:
                try:
                    enhanced_data = self.feature_engineer.add_all_features(data)
                    logger.info(f"Enhanced data with features: {len(enhanced_data.columns)} columns")
                except Exception as e:
                    logger.error(f"Error enhancing features: {e}")
                    enhanced_data = data  # Use original data if enhancement fails
            else:
                enhanced_data = data
            
            # Get latest data for prediction
            latest_data = enhanced_data.iloc[-1]
            
            # Initialize signal result
            signal = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "price": float(enhanced_data['Close'].iloc[-1]),
                "date": str(enhanced_data['Timestamp'].iloc[-1]),
                "signals": {},
                "probability": 0.5,  # Default neutral
                "confidence": 0,
                "action": "HOLD",
                "buy_weight": 0,
                "sell_weight": 0
            }
            
            # Market regime detection
            regime_id, regime_description = self._detect_market_regime(enhanced_data)
            signal["market_regime"] = {
                "id": regime_id,
                "description": regime_description
            }
            logger.info(f"Market regime for {ticker}: {regime_description} (ID: {regime_id})")
            
            # Ensemble model prediction
            ensemble_prob, ensemble_conf = self._generate_ensemble_signal(enhanced_data)
            signal["ensemble"] = {
                "probability": float(ensemble_prob),
                "confidence": float(ensemble_conf),
                "signal": 1 if ensemble_prob > 0.55 else -1 if ensemble_prob < 0.45 else 0
            }
            signal["signals"]["ensemble"] = signal["ensemble"]["signal"]
            logger.info(f"Ensemble prediction for {ticker}: prob={ensemble_prob:.4f}, confidence={ensemble_conf:.4f}")
            
            # HFT signals
            hft_signal, hft_conf, buy_weight, sell_weight = self._generate_hft_signal(enhanced_data)
            signal["hft"] = {
                "buy_weight": float(buy_weight),
                "sell_weight": float(sell_weight),
                "signal": hft_signal,
                "confidence": float(hft_conf)
            }
            signal["signals"]["hft"] = signal["hft"]["signal"]
            logger.info(f"HFT signal for {ticker}: {hft_signal}, confidence={hft_conf:.4f}")
            
            # RL weights
            rl_buy_weight, rl_sell_weight = self._generate_rl_weights(enhanced_data)
            signal["rl_weight"] = {
                "buy_weight": float(rl_buy_weight),
                "sell_weight": float(rl_sell_weight),
                "signal": 1 if rl_buy_weight > rl_sell_weight else -1 if rl_sell_weight > rl_buy_weight else 0
            }
            signal["signals"]["rl"] = signal["rl_weight"]["signal"]
            signal["buy_weight"] = float(rl_buy_weight)
            signal["sell_weight"] = float(rl_sell_weight)
            logger.info(f"RL weights for {ticker}: buy={rl_buy_weight:.2f}, sell={rl_sell_weight:.2f}")
            
            # Combine signals with regime-specific weights
            self._combine_signals(signal, regime_id)
            
            # Store signal in MongoDB if available or in in-memory storage
            if self.mongo_client:
                try:
                    # Convert all numpy types to Python native types for MongoDB
                    signal_json = self._convert_to_json_safe(signal)
                    
                    # Create or update signal in MongoDB
                    db = self.mongo_client.trading_signals
                    db.signals.update_one(
                        {"ticker": ticker},
                        {"$set": signal_json},
                        upsert=True
                    )
                    logger.info(f"Stored signal for {ticker} in MongoDB")
                except Exception as e:
                    logger.error(f"Error storing signal in MongoDB: {e}")
            else:
                # Store in in-memory dictionary
                self.signals_storage[ticker] = signal
                logger.info(f"Stored signal for {ticker} in memory")
            
            # Add to last processed tickers
            self.last_processed_tickers.add(ticker)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _detect_market_regime(self, data):
        """
        Detect the market regime based on price data.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            tuple: (regime_id, regime_description)
        """
        if self.regime_detector:
            # Use real regime detector if available
            return self.regime_detector.detect_regime(data)
        
        # Simple regime detection based on trends and volatility
        try:
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            # Calculate indicators
            trend = returns.mean() * 100  # Mean return in percent
            volatility = returns.std() * 100  # Volatility in percent
            
            # Check for bull market
            if trend > 0.05 and volatility < 1.5:
                return 1, "Bull Market"
            
            # Check for bear market
            elif trend < -0.05 and volatility > 1.0:
                return 2, "Bear Market"
            
            # Check for sideways market
            elif abs(trend) < 0.03 and volatility < 1.0:
                return 3, "Sideways Market"
            
            # Default
            else:
                return 0, "Default"
                
        except Exception as e:
            logger.warning(f"Error in market regime detection: {e}")
            # Return random regime for fallback
            regime_id = np.random.randint(0, 4)
            regime_descriptions = {
                0: "Default",
                1: "Bull Market",
                2: "Bear Market",
                3: "Sideways Market"
            }
            return regime_id, regime_descriptions[regime_id]
    
    def _generate_ensemble_signal(self, data):
        """
        Generate signal from ensemble prediction models.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            tuple: (probability, confidence)
        """
        if self.ensemble_model:
            # Use real ensemble model if available
            return self.ensemble_model.predict(data)
        
        # Generate synthetic prediction based on recent price trend and indicators
        try:
            if len(data) >= 20:
                # Calculate some basic indicators
                data['sma20'] = data['Close'].rolling(20).mean()
                data['sma50'] = data['Close'].rolling(50).mean()
                data['rsi'] = self._calculate_rsi(data['Close'], 14)
                
                # Recent trend
                recent_trend = data['Close'].pct_change(5).iloc[-1]
                
                # Trend direction from moving averages
                trend_signal = 1 if data['sma20'].iloc[-1] > data['sma50'].iloc[-1] else -1
                
                # RSI signal (overbought/oversold)
                rsi = data['rsi'].iloc[-1]
                rsi_signal = 1 if rsi < 30 else -1 if rsi > 70 else trend_signal * 0.5
                
                # Combined signal (-1 to 1 range)
                combined_signal = (recent_trend * 10 + trend_signal + rsi_signal) / 3
                
                # Map to probability (0.3 to 0.7)
                prob = 0.5 + min(max(combined_signal * 0.2, -0.2), 0.2)
                
                # Confidence based on agreement between signals
                agreement = (abs(recent_trend * 10) + abs(trend_signal) + abs(rsi_signal)) / 3
                confidence = min(agreement, 1.0)
                
                return prob, confidence
            else:
                return 0.5, 0.0
                
        except Exception as e:
            logger.warning(f"Error in ensemble signal generation: {e}")
            # Fallback to simple trend-based prediction
            if len(data) >= 5:
                recent_trend = data['Close'].pct_change(5).iloc[-1]
                trend_prob = 0.5 + min(max(recent_trend * 10, -0.2), 0.2)
                confidence = abs(recent_trend) * 5  # 0 to 1
                return float(trend_prob), float(confidence)
            else:
                return 0.5, 0.0
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _generate_hft_signal(self, data):
        """
        Generate high-frequency trading signals.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            tuple: (signal, confidence, buy_weight, sell_weight)
        """
        if self.hft_analyzer:
            # Use real HFT analyzer if available
            return self.hft_analyzer.analyze(data)
        
        # Generate synthetic HFT signals
        try:
            if len(data) >= 20:
                # Calculate moving averages
                sma5 = data['Close'].rolling(5).mean()
                sma20 = data['Close'].rolling(20).mean()
                
                # Calculate MACD
                ema12 = data['Close'].ewm(span=12).mean()
                ema26 = data['Close'].ewm(span=26).mean()
                macd = ema12 - ema26
                signal_line = macd.ewm(span=9).mean()
                macd_hist = macd - signal_line
                
                # Bollinger Bands
                sma20 = data['Close'].rolling(20).mean()
                std20 = data['Close'].rolling(20).std()
                upper_band = sma20 + (std20 * 2)
                lower_band = sma20 - (std20 * 2)
                
                # Check signals
                price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                
                # Moving average crossover
                ma_signal = 0
                if sma5.iloc[-1] > sma20.iloc[-1] and sma5.iloc[-2] <= sma20.iloc[-2]:
                    ma_signal = 1  # Bullish crossover
                elif sma5.iloc[-1] < sma20.iloc[-1] and sma5.iloc[-2] >= sma20.iloc[-2]:
                    ma_signal = -1  # Bearish crossover
                
                # MACD signal
                macd_signal = 0
                if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
                    macd_signal = 1  # Bullish crossover
                elif macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]:
                    macd_signal = -1  # Bearish crossover
                
                # Bollinger band signal
                bb_signal = 0
                if price < lower_band.iloc[-1]:
                    bb_signal = 1  # Price below lower band (potential buy)
                elif price > upper_band.iloc[-1]:
                    bb_signal = -1  # Price above upper band (potential sell)
                
                # Combine signals
                signals = [ma_signal, macd_signal, bb_signal]
                signals = [s for s in signals if s != 0]  # Remove neutral signals
                
                if signals:
                    # Calculate overall signal as average of non-zero signals
                    overall_signal = sum(signals) / len(signals)
                    
                    # Calculate confidence based on agreement
                    if all(s > 0 for s in signals):
                        confidence = 0.8  # Strong buy
                    elif all(s < 0 for s in signals):
                        confidence = 0.8  # Strong sell
                    else:
                        confidence = 0.3  # Mixed signals
                else:
                    # No clear signals
                    overall_signal = 0
                    confidence = 0.1
                
                # Calculate buy/sell weights based on signal strength
                if overall_signal > 0:
                    buy_weight = 500 * overall_signal * confidence
                    sell_weight = 0
                elif overall_signal < 0:
                    buy_weight = 0
                    sell_weight = -500 * overall_signal * confidence
                else:
                    buy_weight = 200
                    sell_weight = 200
                
                # Convert signal to integer for storage
                final_signal = 1 if overall_signal > 0.2 else -1 if overall_signal < -0.2 else 0
                
                return final_signal, confidence, buy_weight, sell_weight
            else:
                return 0, 0.1, 200, 200
                
        except Exception as e:
            logger.warning(f"Error in HFT signal generation: {e}")
            # Fallback to simple SMA crossover
            try:
                if len(data) >= 20:
                    sma5 = data['Close'].rolling(5).mean()
                    sma20 = data['Close'].rolling(20).mean()
                    
                    if sma5.iloc[-1] > sma20.iloc[-1] and sma5.iloc[-2] <= sma20.iloc[-2]:
                        hft_signal = 1  # Buy
                        hft_confidence = 0.8
                    elif sma5.iloc[-1] < sma20.iloc[-1] and sma5.iloc[-2] >= sma20.iloc[-2]:
                        hft_signal = -1  # Sell
                        hft_confidence = 0.8
                    else:
                        if sma5.iloc[-1] > sma20.iloc[-1]:
                            hft_signal = 0.5  # Weak buy
                            hft_confidence = 0.3
                        elif sma5.iloc[-1] < sma20.iloc[-1]:
                            hft_signal = -0.5  # Weak sell
                            hft_confidence = 0.3
                        else:
                            hft_signal = 0  # Neutral
                            hft_confidence = 0.1
                else:
                    hft_signal = 0
                    hft_confidence = 0.1
                
                buy_weight = max(0, hft_signal) * 500
                sell_weight = max(0, -hft_signal) * 500
                
                return int(np.sign(hft_signal)), hft_confidence, buy_weight, sell_weight
            except:
                return 0, 0.1, 200, 200
    
    def _generate_rl_weights(self, data):
        """
        Generate reinforcement learning-based trading weights.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            tuple: (buy_weight, sell_weight)
        """
        if self.rl_weight_generator:
            # Use real RL weight generator if available
            return self.rl_weight_generator.generate(data)
        
        # Generate synthetic RL weights based on recent trend and volatility
        try:
            if len(data) >= 10:
                recent_returns = data['Close'].pct_change().iloc[-10:].dropna()
                trend = recent_returns.mean() * 100  # Average daily return in percent
                volatility = recent_returns.std() * 100  # Volatility in percent
                
                # Adjust buy/sell weights based on trend and volatility
                if trend > 0.5:  # Strong positive trend
                    buy_weight = 800 + np.random.randint(0, 200)
                    sell_weight = 0
                elif trend > 0.1:  # Weak positive trend
                    buy_weight = 500 + np.random.randint(0, 300)
                    sell_weight = np.random.randint(0, 200)
                elif trend < -0.5:  # Strong negative trend
                    buy_weight = 0
                    sell_weight = 800 + np.random.randint(0, 200)
                elif trend < -0.1:  # Weak negative trend
                    buy_weight = np.random.randint(0, 200)
                    sell_weight = 500 + np.random.randint(0, 300)
                else:  # Neutral
                    buy_weight = 200 + np.random.randint(0, 300)
                    sell_weight = 200 + np.random.randint(0, 300)
                
                # Adjust for volatility
                if volatility > 2.0:  # High volatility
                    # Reduce both weights in high volatility
                    buy_weight = int(buy_weight * 0.8)
                    sell_weight = int(sell_weight * 0.8)
                
                return float(buy_weight), float(sell_weight)
            else:
                # Not enough data
                buy_weight = np.random.randint(0, 500)
                sell_weight = np.random.randint(0, 500)
                return float(buy_weight), float(sell_weight)
                
        except Exception as e:
            logger.warning(f"Error in RL weights generation: {e}")
            # Fallback to random weights
            buy_weight = np.random.randint(0, 500)
            sell_weight = np.random.randint(0, 500)
            return float(buy_weight), float(sell_weight)
    
    def _combine_signals(self, signal, regime_id):
        """
        Combine signals with regime-specific weights.
        
        Args:
            signal: Signal dictionary to be updated
            regime_id: Current market regime ID
        """
        if not signal["signals"]:
            return
        
        # Different weights for different regimes
        regime_weights = {
            0: {"ensemble": 0.5, "hft": 0.3, "rl": 0.2},  # Default regime
            1: {"ensemble": 0.6, "hft": 0.2, "rl": 0.2},  # Regime 1 (e.g., Bull market)
            2: {"ensemble": 0.4, "hft": 0.4, "rl": 0.2},  # Regime 2 (e.g., Bear market)
            3: {"ensemble": 0.3, "hft": 0.5, "rl": 0.2}   # Regime 3 (e.g., Sideways market)
        }
        
        # Use weights for current regime or default
        weights = regime_weights.get(regime_id, regime_weights[0])
        logger.info(f"Using regime weights: {weights}")
        
        components_available = set(signal["signals"].keys())
        
        # Calculate combined probability if we have at least one signal component
        if components_available:
            # Initialize with default values
            ensemble_prob = 0.5
            hft_signal = 0
            rl_signal = 0
            
            # Get values from available components
            if "ensemble" in signal:
                ensemble_prob = signal["ensemble"]["probability"]
            
            if "hft" in signal:
                hft_signal = signal["hft"]["signal"]
            
            if "rl_weight" in signal:
                rl_signal = signal["rl_weight"]["signal"]
            
            # Convert signals to probabilities
            hft_prob = 0.5 + (hft_signal * 0.25)  # Scale to 0.25-0.75 range
            rl_prob = 0.5 + (rl_signal * 0.25)    # Scale to 0.25-0.75 range
            
            # Calculate the weighted contributions
            weighted_comps = []
            
            if "ensemble" in components_available:
                weighted_comps.append(weights["ensemble"] * ensemble_prob)
            
            if "hft" in components_available:
                weighted_comps.append(weights["hft"] * hft_prob)
            
            if "rl" in components_available:
                weighted_comps.append(weights["rl"] * rl_prob)
            
            # Calculate combined probability (normalize weights to sum to 1)
            weight_sum = sum([weights[comp] for comp in components_available if comp in weights])
            if weight_sum > 0:
                combined_prob = sum(weighted_comps) / weight_sum
            else:
                combined_prob = 0.5
            
            # Get confidence as agreement between signals
            signals = list(signal["signals"].values())
            unique_signals = set(signals)
            if len(unique_signals) == 1:
                confidence = 1.0  # All agree
            elif len(unique_signals) == 2 and 0 in unique_signals:
                confidence = 0.5  # Partial agreement
            else:
                confidence = 0.0  # Disagreement
            
            # Apply confidence to probability (move toward 0.5 for low confidence)
            adjusted_prob = 0.5 + (combined_prob - 0.5) * confidence
            
            signal["probability"] = float(adjusted_prob)
            signal["confidence"] = float(confidence)
            
            # Determine action
            if adjusted_prob > 0.65:  # Strong buy
                signal["action"] = "STRONG BUY"
            elif adjusted_prob > 0.55:  # Buy
                signal["action"] = "BUY"
            elif adjusted_prob < 0.35:  # Strong sell
                signal["action"] = "STRONG SELL"
            elif adjusted_prob < 0.45:  # Sell
                signal["action"] = "SELL"
            else:
                signal["action"] = "HOLD"
            
            # Also combine buy/sell weights
            if "hft" in signal and "rl_weight" in signal:
                # Weighted sum of HFT and RL weights
                hft_weight = weights["hft"] / (weights["hft"] + weights["rl"])
                rl_weight = weights["rl"] / (weights["hft"] + weights["rl"])
                
                # Scale by confidence
                buy_weight = (signal["hft"]["buy_weight"] * hft_weight + 
                             signal["rl_weight"]["buy_weight"] * rl_weight) * (1 + confidence)
                
                sell_weight = (signal["hft"]["sell_weight"] * hft_weight + 
                               signal["rl_weight"]["sell_weight"] * rl_weight) * (1 + confidence)
                
                # Adjust weights based on final action
                if signal["action"] == "STRONG BUY":
                    buy_weight = max(buy_weight, 1000)
                    sell_weight = 0
                elif signal["action"] == "BUY":
                    buy_weight = max(buy_weight, 500)
                    sell_weight = min(sell_weight, 200)
                elif signal["action"] == "STRONG SELL":
                    sell_weight = max(sell_weight, 1000)
                    buy_weight = 0
                elif signal["action"] == "SELL":
                    sell_weight = max(sell_weight, 500)
                    buy_weight = min(buy_weight, 200)
                
                signal["buy_weight"] = float(buy_weight)
                signal["sell_weight"] = float(sell_weight)
            
            logger.info(f"Final signal: {signal['action']}, probability={adjusted_prob:.4f}, confidence={confidence:.4f}")
    
    def _convert_to_json_safe(self, obj):
        """Convert numpy types to Python native types for MongoDB."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_safe(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_json_safe(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        else:
            return obj
    
    def process_historical_data(self, data, ticker=None):
        """
        Process a DataFrame of stock data and generate a signal.
        
        This method is designed to be called by the trading or ranking clients
        when they already have data loaded for a ticker.
        
        Args:
            data: DataFrame with stock data or list of dicts from MongoDB
            ticker: Optional ticker symbol (read from data if not provided)
            
        Returns:
            dict: Trading signal
        """
        try:
            # Convert list of dicts to DataFrame if needed
            if isinstance(data, list) and data:
                data = pd.DataFrame(data)
            
            # Make sure data has Symbol column
            if ticker is not None and 'Symbol' not in data.columns:
                data = data.copy()
                data['Symbol'] = ticker
            
            # Fix column names if they don't match expected format
            if ('open' in data.columns and 'Open' not in data.columns):
                data = data.rename(columns={
                    'open': 'Open', 
                    'high': 'High', 
                    'low': 'Low', 
                    'close': 'Close', 
                    'volume': 'Volume'
                })
            
            # Ensure timestamp is datetime
            if 'Timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['Timestamp']):
                data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            
            # Generate and return signal
            return self.generate_signal(ticker=ticker, data=data)
            
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def start_monitoring(self, interval=60):
        """
        Start monitoring and processing signals for stocks in the database.
        
        This runs in a separate thread and periodically checks for new data
        in the MongoDB database, generating signals for any new or updated tickers.
        
        Args:
            interval: Check interval in seconds (default: 60)
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return False
        
        # Reset stop event
        self.stop_event.clear()
        
        # Create and start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"Started signal monitoring thread (interval: {interval}s)")
        return True
    
    def stop_monitoring(self):
        """
        Stop the monitoring thread.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            logger.warning("Monitoring thread not running")
            return False
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for thread to stop
        self.monitor_thread.join(timeout=5)
        
        logger.info("Stopped signal monitoring thread")
        return True
    
    def _monitoring_loop(self, interval):
        """
        Main monitoring loop that runs in a separate thread.
        
        Args:
            interval: Check interval in seconds
        """
        logger.info("Signal monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                active_tickers = []
                
                # Try to get tickers from trading database first
                if self.mongo_client:
                    try:
                        # Check the trading database for active tickers
                        trading_db = self.mongo_client.trades
                        if 'assets_quantities' in trading_db.list_collection_names():
                            # Get tickers from actual portfolio
                            portfolio_tickers = list(trading_db.assets_quantities.find({}, {"symbol": 1}))
                            active_tickers.extend([ticker['symbol'] for ticker in portfolio_tickers if 'symbol' in ticker])
                            logger.info(f"Found {len(active_tickers)} tickers in portfolio")
                        
                        # Check ranking database for additional tickers
                        rank_db = self.mongo_client.trading_simulator
                        if 'algorithm_holdings' in rank_db.list_collection_names():
                            for strategy_doc in rank_db.algorithm_holdings.find({}):
                                if 'holdings' in strategy_doc:
                                    strategy_tickers = list(strategy_doc['holdings'].keys())
                                    active_tickers.extend(strategy_tickers)
                            active_tickers = list(set(active_tickers))  # Remove duplicates
                            logger.info(f"Found total of {len(active_tickers)} unique tickers in portfolio and algorithms")
                        
                        # If still no tickers found, try to look at the historical database
                        if not active_tickers and 'HistoricalDatabase' in self.mongo_client.list_database_names():
                            hist_db = self.mongo_client.HistoricalDatabase
                            if 'HistoricalDatabase' in hist_db.list_collection_names():
                                recent_data = list(hist_db.HistoricalDatabase.find(
                                    {"timestamp": {"$gt": datetime.now() - timedelta(days=1)}},
                                    {"ticker": 1}
                                ))
                                active_tickers = list(set([doc['ticker'] for doc in recent_data if 'ticker' in doc]))
                                logger.info(f"Found {len(active_tickers)} recent tickers in historical database")
                    except Exception as e:
                        logger.error(f"Error finding active tickers in database: {e}")
                
                # If no tickers found in database, fall back to tickers.txt
                if not active_tickers:
                    # Try to read from a 'tickers.txt' file if it exists
                    try:
                        if os.path.exists(os.path.join(self.model_path, "tickers.txt")):
                            with open(os.path.join(self.model_path, "tickers.txt"), 'r') as f:
                                active_tickers = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                            logger.info(f"Loaded {len(active_tickers)} tickers from tickers.txt")
                    except Exception as e:
                        logger.warning(f"Error reading tickers.txt: {e}")
                
                # If still no tickers, log warning but don't use demo tickers
                if not active_tickers:
                    logger.warning("No active tickers found, skipping signal generation")
                    # Wait for next interval without generating any signals
                    self.stop_event.wait(interval)
                    continue
                
                # Generate signals for all active tickers
                logger.info(f"Generating signals for {len(active_tickers)} tickers")
                
                for ticker in active_tickers:
                    try:
                        # Get data and generate a signal
                        self.generate_signal(ticker=ticker)
                    except Exception as e:
                        logger.error(f"Error generating signal for {ticker}: {e}")
                
                # Insert signals into trading system
                self.insert_signals_into_trading_system()
                
                # Wait for next interval
                self.stop_event.wait(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                logger.error(traceback.format_exc())
                # Sleep for a while to avoid flooding logs with errors
                self.stop_event.wait(interval * 2)
        
        logger.info("Signal monitoring loop stopped")
    
    def insert_signals_into_trading_system(self):
        """
        Insert signals into the trading client's decision-making process.
        This augments the existing trading strategy with ML-based signals.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all current signals
            if self.mongo_client:
                signal_db = self.mongo_client.trading_signals
                signals = list(signal_db.signals.find({}))
            else:
                # Use in-memory signals
                signals = [signal for signal in self.signals_storage.values()]
            
            if not signals:
                logger.warning("No signals available to insert into trading system")
                return False
            
            # If we have MongoDB, store in trading system database
            if self.mongo_client:
                try:
                    # Get trading system database
                    trading_db = self.mongo_client.trades
                    
                    # Create or update signal_weights collection
                    if 'signal_weights' not in trading_db.list_collection_names():
                        trading_db.create_collection('signal_weights')
                        logger.info("Created signal_weights collection in trades database")
                    
                    # Insert or update signal weights for each ticker
                    for signal in signals:
                        ticker = signal['ticker']
                        
                        # Extract buy/sell weights
                        buy_weight = signal.get('buy_weight', 0)
                        sell_weight = signal.get('sell_weight', 0)
                        
                        # Convert action to weight adjustment
                        action = signal.get('action', 'HOLD')
                        confidence = signal.get('confidence', 0)
                        
                        # Scale weights based on action and confidence
                        if action == 'STRONG BUY':
                            buy_weight = max(buy_weight, 1000) * (1 + confidence)
                            sell_weight = 0
                        elif action == 'BUY':
                            buy_weight = max(buy_weight, 500) * (1 + confidence)
                            sell_weight = 0
                        elif action == 'STRONG SELL':
                            sell_weight = max(sell_weight, 1000) * (1 + confidence)
                            buy_weight = 0
                        elif action == 'SELL':
                            sell_weight = max(sell_weight, 500) * (1 + confidence)
                            buy_weight = 0
                        
                        # Insert or update in signal_weights collection
                        trading_db.signal_weights.update_one(
                            {"ticker": ticker},
                            {
                                "$set": {
                                    "buy_weight": buy_weight,
                                    "sell_weight": sell_weight,
                                    "action": action,
                                    "confidence": confidence,
                                    "timestamp": datetime.now().isoformat(),
                                    "price": signal.get('price', 0),
                                    "market_regime": signal.get('market_regime', {}).get('description', 'Unknown')
                                }
                            },
                            upsert=True
                        )
                        
                        logger.info(f"Inserted signal for {ticker} into trading system: Buy={buy_weight}, Sell={sell_weight}, Action={action}")
                except Exception as e:
                    logger.error(f"Error storing signals in trading database: {e}")
                    logger.error(traceback.format_exc())
            
            # Always log the signals for demonstration purposes
            for signal in signals:
                ticker = signal['ticker']
                action = signal.get('action', 'HOLD')
                prob = signal.get('probability', 0.5)
                conf = signal.get('confidence', 0)
                
                logger.info(f"Signal for {ticker}: {action} (prob={prob:.4f}, conf={conf:.2f})")
            
            logger.info(f"Successfully processed {len(signals)} signals")
            return True
        except Exception as e:
            logger.error(f"Error inserting signals into trading system: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def close(self):
        """
        Close connections and stop threads.
        """
        # Stop monitoring thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_monitoring()
        
        # Close MongoDB connection
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("Closed MongoDB connection")


# Integration functions to work with existing trading and ranking clients

def integrate_with_trading_client(ticker, data, mongo_client=None):
    """
    Function to integrate with trading_client.py
    
    This function should be called from trading_client.py when processing a ticker
    to generate a signal and adjust buy/sell weights.
    
    Args:
        ticker: Stock ticker symbol
        data: DataFrame with stock data
        mongo_client: Optional MongoDB client
        
    Returns:
        tuple: (buy_weight, sell_weight) adjusted by signal
    """
    try:
        # Create signal integrator (lazy loaded - will only be created once)
        if not hasattr(integrate_with_trading_client, 'integrator'):
            integrate_with_trading_client.integrator = AmplifySignalIntegrator(mongo_url=mongo_client)
            # Start monitoring thread
            integrate_with_trading_client.integrator.start_monitoring()
        
        # Generate signal
        signal = integrate_with_trading_client.integrator.process_historical_data(data, ticker)
        
        # Return buy/sell weights
        return signal.get('buy_weight', 0), signal.get('sell_weight', 0)
    except Exception as e:
        logger.error(f"Error in integrate_with_trading_client: {e}")
        return 0, 0

def integrate_with_ranking_client(ticker, data, mongo_client=None):
    """
    Function to integrate with ranking_client.py
    
    This function should be called from ranking_client.py when processing a ticker
    to generate and store a signal for later use.
    
    Args:
        ticker: Stock ticker symbol
        data: DataFrame with stock data
        mongo_client: Optional MongoDB client
        
    Returns:
        dict: Signal result
    """
    try:
        # Create signal integrator (lazy loaded - will only be created once)
        if not hasattr(integrate_with_ranking_client, 'integrator'):
            integrate_with_ranking_client.integrator = AmplifySignalIntegrator(mongo_url=mongo_client)
            # Start monitoring thread
            integrate_with_ranking_client.integrator.start_monitoring()
        
        # Generate and store signal
        return integrate_with_ranking_client.integrator.process_historical_data(data, ticker)
    except Exception as e:
        logger.error(f"Error in integrate_with_ranking_client: {e}")
        return {"error": str(e)}


# If run as a standalone script, start monitoring
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AmplifySignalIntegrator - Trading signal integration")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model directory")
    parser.add_argument("--mongo_url", type=str, default=None, help="MongoDB connection URL")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    parser.add_argument("--tickers", type=str, default=None, help="Path to tickers.txt file")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print(" AlphaSynth Advanced Trading Signal Integrator ONLINE ".center(80, '#'))
    print("=" * 80 + "\n")
    
    try:
        # Initialize signal integrator
        integrator = AmplifySignalIntegrator(model_path=args.model_path, mongo_url=args.mongo_url)
        
        # Create tickers.txt file if provided
        if args.tickers:
            tickers_file = os.path.join(args.model_path or os.getcwd(), "tickers.txt")
            with open(tickers_file, 'w') as f:
                for ticker in args.tickers.split(','):
                    f.write(f"{ticker.strip()}\n")
            print(f"Created tickers.txt with {len(args.tickers.split(','))} tickers")
        
        # Start monitoring
        if integrator.start_monitoring(interval=args.interval):
            print(f"Signal monitoring started with {args.interval}s interval")
            
            # Insert signals into trading system
            if integrator.insert_signals_into_trading_system():
                print("Signals inserted into trading system")
            
            print("\nPress Ctrl+C to stop...")
            
            # Keep running until Ctrl+C
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping...")
                integrator.stop_monitoring()
                integrator.close()
                print("Signal integrator stopped")
        else:
            print("Failed to start signal monitoring")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)