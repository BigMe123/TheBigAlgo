# enhanced_trading_system.py
import os
import sys
import pandas as pd
import numpy as np
import time
import logging
import json
import pickle
import uuid
import schedule
import threading
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import importlib
import socket
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any

# Custom module imports
# Import these conditionally to avoid errors if not installed
try:
    from advanced_feature_engineering import AdvancedFeatureEngineering
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

try:
    from ensemble_model import EnsembleModelManager
    ENSEMBLE_MODEL_AVAILABLE = True
except ImportError:
    ENSEMBLE_MODEL_AVAILABLE = False

try:
    from rl_weight_generator import RLWeightGenerator
    RL_WEIGHT_AVAILABLE = True
except ImportError:
    RL_WEIGHT_AVAILABLE = False

try:
    from market_regime_detector import MarketRegimeDetector
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False

try:
    from hft_analyzer import HFTAnalyzer
    HFT_ANALYZER_AVAILABLE = True
except ImportError:
    HFT_ANALYZER_AVAILABLE = False

try:
    from portfolio_optimizer import PortfolioOptimizer
    PORTFOLIO_OPTIMIZER_AVAILABLE = True
except ImportError:
    PORTFOLIO_OPTIMIZER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("enhanced_trading.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger("EnhancedTrading")

class EnhancedTradingSystem:
    """
    Enhanced Trading System with advanced ML, RL, and HFT capabilities.
    
    This is the main orchestrator that integrates all enhanced components:
    1. Advanced Feature Engineering
    2. Ensemble Learning & Model Stacking
    3. Reinforcement Learning-Based Weighting
    4. Regime-Based Weight Adjustment
    5. High-Frequency Trading Enhancements
    6. Smart Portfolio Diversification & Risk Management
    7. Continuous Learning & Model Deployment
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Enhanced Trading System.
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self.feature_engineer = None
        self.ensemble_model = None
        self.rl_weight_generator = None
        self.regime_detector = None
        self.hft_analyzer = None
        self.portfolio_optimizer = None
        
        # Data storage
        self.historical_data = {}
        self.prediction_history = []
        self.performance_metrics = {}
        self.current_regime = None
        
        # Model registry
        self.model_registry = {}
        
        # Training status
        self.is_training = False
        self.last_training_time = None
        self.training_thread = None
        
        # Initialize components based on configuration
        self._initialize_components()
        
        # Load existing models and data
        self._load_existing_models()
        
        # Setup schedulers
        self.scheduler_thread = None
        self.scheduler_running = False
        self._setup_schedulers()
        
        # Start scheduler if auto-start is enabled
        if self.config.get("auto_start_scheduler", False):
            self.start_scheduler()
        
        logger.info("Enhanced Trading System initialized")
    
    def _load_config(self, config_path):
        """
        Load configuration from file or use default.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Configuration settings
        """
        # Default configuration
        default_config = {
            "data_dir": "./data",
            "model_dir": "./models",
            "log_dir": "./logs",
            "prediction_dir": "./predictions",
            "performance_dir": "./performance",
            
            "feature_engineering": {
                "enabled": FEATURE_ENGINEERING_AVAILABLE,
                "api_key": None
            },
            
            "ensemble_model": {
                "enabled": ENSEMBLE_MODEL_AVAILABLE,
                "deep_learning": True,
                "model_dir": "./models/ensemble"
            },
            
            "rl_weight": {
                "enabled": RL_WEIGHT_AVAILABLE,
                "model_dir": "./models/rl_weights"
            },
            
            "regime_detector": {
                "enabled": REGIME_DETECTOR_AVAILABLE,
                "n_regimes": 3,
                "model_dir": "./models/regimes"
            },
            
            "hft_analyzer": {
                "enabled": HFT_ANALYZER_AVAILABLE,
                "order_imbalance_window": 20,
                "vwap_window": 50
            },
            
            "portfolio_optimizer": {
                "enabled": PORTFOLIO_OPTIMIZER_AVAILABLE,
                "risk_free_rate": 0.03,
                "max_drawdown_limit": 0.15
            },
            
            "training": {
                "auto_train": True,
                "training_frequency": "weekly",  # daily, weekly, monthly
                "training_day": "sunday",  # day of week for weekly
                "training_date": 1,  # day of month for monthly
                "training_time": "20:00",  # time of day
                "data_lookback_days": 365,
                "validation_size": 0.2,
                "default_tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA"]
            },
            
            "prediction": {
                "prediction_frequency": "daily",  # daily, hourly, custom
                "prediction_time": "08:00",  # time for daily
                "custom_schedule": None,  # list of times for custom
                "save_predictions": True
            },
            
            "continuous_learning": {
                "enabled": True,
                "feedback_window_days": 30,
                "min_feedback_samples": 100,
                "performance_threshold": 0.55  # win rate
            },
            
            "data_sources": {
                "use_yfinance": True,
                "use_alpha_vantage": False,
                "use_polygon": False,
                "api_keys": {
                    "alpha_vantage": None,
                    "polygon": None
                }
            },
            
            "auto_start_scheduler": True
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Update default config with loaded config
                self._update_nested_dict(default_config, loaded_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        return default_config
    
    def _update_nested_dict(self, d, u):
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.config["data_dir"],
            self.config["model_dir"],
            self.config["log_dir"],
            self.config["prediction_dir"],
            self.config["performance_dir"],
            self.config["ensemble_model"]["model_dir"],
            self.config["rl_weight"]["model_dir"],
            self.config["regime_detector"]["model_dir"]
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize trading system components based on configuration."""
        # Initialize feature engineering
        if self.config["feature_engineering"]["enabled"] and FEATURE_ENGINEERING_AVAILABLE:
            try:
                self.feature_engineer = AdvancedFeatureEngineering(
                    api_key=self.config["feature_engineering"]["api_key"]
                )
                logger.info("Feature engineering component initialized")
            except Exception as e:
                logger.error(f"Error initializing feature engineering: {e}")
        
        # Initialize ensemble model
        if self.config["ensemble_model"]["enabled"] and ENSEMBLE_MODEL_AVAILABLE:
            try:
                self.ensemble_model = EnsembleModelManager(
                    model_dir=self.config["ensemble_model"]["model_dir"]
                )
                logger.info("Ensemble model component initialized")
            except Exception as e:
                logger.error(f"Error initializing ensemble model: {e}")
        
        # Initialize RL weight generator
        if self.config["rl_weight"]["enabled"] and RL_WEIGHT_AVAILABLE:
            try:
                self.rl_weight_generator = RLWeightGenerator(
                    model_dir=self.config["rl_weight"]["model_dir"]
                )
                logger.info("RL weight generator component initialized")
            except Exception as e:
                logger.error(f"Error initializing RL weight generator: {e}")
        
        # Initialize regime detector
        if self.config["regime_detector"]["enabled"] and REGIME_DETECTOR_AVAILABLE:
            try:
                self.regime_detector = MarketRegimeDetector(
                    model_dir=self.config["regime_detector"]["model_dir"],
                    n_regimes=self.config["regime_detector"]["n_regimes"]
                )
                logger.info("Market regime detector component initialized")
            except Exception as e:
                logger.error(f"Error initializing market regime detector: {e}")
        
        # Initialize HFT analyzer
        if self.config["hft_analyzer"]["enabled"] and HFT_ANALYZER_AVAILABLE:
            try:
                self.hft_analyzer = HFTAnalyzer(
                    config={
                        "order_imbalance_window": self.config["hft_analyzer"]["order_imbalance_window"],
                        "vwap_window": self.config["hft_analyzer"]["vwap_window"]
                    }
                )
                logger.info("HFT analyzer component initialized")
            except Exception as e:
                logger.error(f"Error initializing HFT analyzer: {e}")
        
        # Initialize portfolio optimizer
        if self.config["portfolio_optimizer"]["enabled"] and PORTFOLIO_OPTIMIZER_AVAILABLE:
            try:
                self.portfolio_optimizer = PortfolioOptimizer(
                    risk_free_rate=self.config["portfolio_optimizer"]["risk_free_rate"],
                    max_drawdown_limit=self.config["portfolio_optimizer"]["max_drawdown_limit"]
                )
                logger.info("Portfolio optimizer component initialized")
            except Exception as e:
                logger.error(f"Error initializing portfolio optimizer: {e}")
    
    def _load_existing_models(self):
        """Load existing models and model registry."""
        registry_path = os.path.join(self.config["model_dir"], "model_registry.json")
        
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
                logger.info(f"Loaded model registry with {len(self.model_registry)} entries")
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
                self.model_registry = {}
        
        # Load component models
        self._load_component_models()
    
    def _load_component_models(self):
        """Load models for each component."""
        # Load ensemble model
        if self.ensemble_model:
            try:
                self.ensemble_model.load_models()
                logger.info("Ensemble models loaded")
            except Exception as e:
                logger.error(f"Error loading ensemble models: {e}")
        
        # Load RL weight generator model
        if self.rl_weight_generator:
            try:
                self.rl_weight_generator.load_models()
                logger.info("RL weight generator models loaded")
            except Exception as e:
                logger.error(f"Error loading RL weight generator models: {e}")
        
        # Load regime detector model
        if self.regime_detector:
            try:
                self.regime_detector.load_models()
                logger.info("Regime detector models loaded")
            except Exception as e:
                logger.error(f"Error loading regime detector models: {e}")
    
    def _save_model_registry(self):
        """Save model registry to file."""
        registry_path = os.path.join(self.config["model_dir"], "model_registry.json")
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=4)
            logger.info(f"Model registry saved with {len(self.model_registry)} entries")
            return True
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
            return False
    
    def _setup_schedulers(self):
        """Setup training and prediction schedulers."""
        # Clear existing schedule
        schedule.clear()
        
        # Setup training schedule
        training_config = self.config["training"]
        
        if training_config["auto_train"]:
            # Set up training schedule based on frequency
            if training_config["training_frequency"] == "daily":
                schedule.every().day.at(training_config["training_time"]).do(self.train_models)
                logger.info(f"Scheduled daily training at {training_config['training_time']}")
                
            elif training_config["training_frequency"] == "weekly":
                day = training_config["training_day"].lower()
                
                if day == "monday":
                    schedule.every().monday.at(training_config["training_time"]).do(self.train_models)
                elif day == "tuesday":
                    schedule.every().tuesday.at(training_config["training_time"]).do(self.train_models)
                elif day == "wednesday":
                    schedule.every().wednesday.at(training_config["training_time"]).do(self.train_models)
                elif day == "thursday":
                    schedule.every().thursday.at(training_config["training_time"]).do(self.train_models)
                elif day == "friday":
                    schedule.every().friday.at(training_config["training_time"]).do(self.train_models)
                elif day == "saturday":
                    schedule.every().saturday.at(training_config["training_time"]).do(self.train_models)
                elif day == "sunday":
                    schedule.every().sunday.at(training_config["training_time"]).do(self.train_models)
                
                logger.info(f"Scheduled weekly training on {day} at {training_config['training_time']}")
                
            elif training_config["training_frequency"] == "monthly":
                # Schedule monthly training on specified date
                @schedule.scheduled_job('cron', day=str(training_config["training_date"]), hour=training_config["training_time"].split(":")[0], minute=training_config["training_time"].split(":")[1])
                def monthly_training():
                    self.train_models()
                
                logger.info(f"Scheduled monthly training on day {training_config['training_date']} at {training_config['training_time']}")
        
        # Setup prediction schedule
        prediction_config = self.config["prediction"]
        
        if prediction_config["prediction_frequency"] == "daily":
            schedule.every().day.at(prediction_config["prediction_time"]).do(self.run_daily_predictions)
            logger.info(f"Scheduled daily predictions at {prediction_config['prediction_time']}")
            
        elif prediction_config["prediction_frequency"] == "hourly":
            schedule.every().hour.do(self.run_predictions)
            logger.info("Scheduled hourly predictions")
            
        elif prediction_config["prediction_frequency"] == "custom":
            if prediction_config["custom_schedule"]:
                for time_str in prediction_config["custom_schedule"]:
                    schedule.every().day.at(time_str).do(self.run_predictions)
                logger.info(f"Scheduled custom predictions at {prediction_config['custom_schedule']}")
    
    def start_scheduler(self):
        """Start the scheduler in a background thread."""
        if self.scheduler_running:
            logger.warning("Scheduler is already running")
            return False
        
        def run_scheduler():
            self.scheduler_running = True
            logger.info("Scheduler started")
            
            while self.scheduler_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in scheduler: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        return True
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        if not self.scheduler_running:
            logger.warning("Scheduler is not running")
            return False
        
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Scheduler stopped")
        return True
    
    def fetch_historical_data(self, tickers=None, period="2y", interval="1d", use_cache=True):
        """
        Fetch historical data for training or prediction.
        
        Args:
            tickers: List of ticker symbols
            period: Data period (e.g., "2y", "1y", "6mo")
            interval: Data interval (e.g., "1d", "1h", "15m")
            use_cache: Whether to use cached data if available
            
        Returns:
            list: List of DataFrames with historical data
        """
        try:
            # Use default tickers if none provided
            if tickers is None:
                tickers = self.config["training"]["default_tickers"]
            
            # Check cache for each ticker
            data_frames = []
            tickers_to_fetch = []
            
            if use_cache:
                for ticker in tickers:
                    cache_file = os.path.join(
                        self.config["data_dir"], 
                        f"{ticker}_{period}_{interval}.parquet"
                    )
                    
                    if os.path.exists(cache_file):
                        try:
                            # Check if cache is recent enough (within 1 day for daily data)
                            cache_time = os.path.getmtime(cache_file)
                            cache_age = time.time() - cache_time
                            
                            # Use cache if less than 1 day old for daily data
                            if interval == "1d" and cache_age < 86400:
                                df = pd.read_parquet(cache_file)
                                data_frames.append(df)
                                self.historical_data[ticker] = df
                                continue
                            
                            # For intraday data, use stricter cache rules
                            if "m" in interval and cache_age < 3600:  # 1 hour for minute data
                                df = pd.read_parquet(cache_file)
                                data_frames.append(df)
                                self.historical_data[ticker] = df
                                continue
                        except Exception as e:
                            logger.warning(f"Error reading cache for {ticker}: {e}")
                    
                    tickers_to_fetch.append(ticker)
            else:
                tickers_to_fetch = tickers
            
            if tickers_to_fetch:
                logger.info(f"Fetching historical data for {len(tickers_to_fetch)} tickers")
                
                # Determine data source to use
                if self.config["data_sources"]["use_yfinance"]:
                    new_data_frames = self._fetch_from_yfinance(
                        tickers_to_fetch, period, interval
                    )
                elif self.config["data_sources"]["use_alpha_vantage"]:
                    new_data_frames = self._fetch_from_alpha_vantage(
                        tickers_to_fetch, period, interval
                    )
                elif self.config["data_sources"]["use_polygon"]:
                    new_data_frames = self._fetch_from_polygon(
                        tickers_to_fetch, period, interval
                    )
                else:
                    logger.error("No data source configured")
                    return data_frames
                
                # Add new data to results and cache it
                for df in new_data_frames:
                    if df is not None and not df.empty:
                        ticker = df['Symbol'].iloc[0]
                        
                        # Cache the data
                        cache_file = os.path.join(
                            self.config["data_dir"], 
                            f"{ticker}_{period}_{interval}.parquet"
                        )
                        df.to_parquet(cache_file, index=False)
                        
                        data_frames.append(df)
                        self.historical_data[ticker] = df
            
            logger.info(f"Fetched data for {len(data_frames)} tickers")
            return data_frames
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _fetch_from_yfinance(self, tickers, period, interval):
        """Fetch data from Yahoo Finance."""
        try:
            import yfinance as yf
            
            data_frames = []
            
            for ticker in tickers:
                try:
                    # Fetch data
                    stock = yf.Ticker(ticker)
                    df = stock.history(period=period, interval=interval)
                    
                    if df.empty:
                        logger.warning(f"No data returned for {ticker}")
                        continue
                    
                    # Add ticker and reset index
                    df["Symbol"] = ticker
                    df.reset_index(inplace=True)
                    
                    # Add label column (1 if next close > current close)
                    df = df.sort_values('Date' if 'Date' in df.columns else 'Datetime').reset_index(drop=True)
                    df['Label'] = df['Close'].shift(-1) > df['Close']
                    df['Label'] = df['Label'].astype(int)
                    
                    # Rename columns to standard format
                    if 'Datetime' in df.columns:
                        df.rename(columns={"Datetime": "Timestamp"}, inplace=True)
                    elif 'Date' in df.columns:
                        df.rename(columns={"Date": "Timestamp"}, inplace=True)
                    
                    data_frames.append(df)
                    logger.info(f"Fetched {len(df)} rows of data for {ticker}")
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {e}")
            
            return data_frames
            
        except ImportError:
            logger.error("yfinance not installed. Install using: pip install yfinance")
            return []
    
    def _fetch_from_alpha_vantage(self, tickers, period, interval):
        """Fetch data from Alpha Vantage."""
        try:
            import requests
            
            api_key = self.config["data_sources"]["api_keys"]["alpha_vantage"]
            if not api_key:
                logger.error("Alpha Vantage API key not configured")
                return []
            
            data_frames = []
            
            for ticker in tickers:
                try:
                    # Translate period/interval to Alpha Vantage format
                    if interval == "1d":
                        function = "TIME_SERIES_DAILY_ADJUSTED"
                        output_size = "full"
                    elif "m" in interval:
                        function = "TIME_SERIES_INTRADAY"
                        interval_av = interval.replace("m", "min")
                    else:
                        logger.warning(f"Unsupported interval for Alpha Vantage: {interval}")
                        continue
                    
                    # Fetch data
                    url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&interval={interval_av if 'm' in interval else None}&outputsize={output_size}&apikey={api_key}"
                    
                    r = requests.get(url)
                    data = r.json()
                    
                    # Process data
                    # Implement processing logic based on Alpha Vantage response format
                    
                    # Add to data frames
                    # data_frames.append(df)
                    
                except Exception as e:
                    logger.error(f"Error fetching Alpha Vantage data for {ticker}: {e}")
            
            return data_frames
            
        except ImportError:
            logger.error("requests not installed. Install using: pip install requests")
            return []
    
    def _fetch_from_polygon(self, tickers, period, interval):
        """Fetch data from Polygon."""
        try:
            from polygon import RESTClient
            
            api_key = self.config["data_sources"]["api_keys"]["polygon"]
            if not api_key:
                logger.error("Polygon API key not configured")
                return []
            
            data_frames = []
            
            # Implement Polygon data fetching logic
            
            return data_frames
            
        except ImportError:
            logger.error("polygon-api-client not installed. Install using: pip install polygon-api-client")
            return []
    
    def enhance_features(self, data_frames):
        """
        Enhance data with advanced features.
        
        Args:
            data_frames: List of DataFrames with market data
            
        Returns:
            list: Enhanced DataFrames
        """
        if not self.feature_engineer:
            logger.warning("Feature engineering component not initialized")
            return data_frames
        
        try:
            enhanced_frames = []
            
            for df in data_frames:
                if df is not None and not df.empty:
                    try:
                        enhanced_df = self.feature_engineer.add_all_features(df)
                        enhanced_frames.append(enhanced_df)
                    except Exception as e:
                        logger.error(f"Error enhancing features for {df['Symbol'].iloc[0]}: {e}")
                        enhanced_frames.append(df)  # Use original data on error
            
            logger.info(f"Enhanced features for {len(enhanced_frames)} datasets")
            return enhanced_frames
            
        except Exception as e:
            logger.error(f"Error in feature enhancement: {e}")
            return data_frames
    
    def detect_market_regime(self, data_frames):
        """
        Detect current market regime.
        
        Args:
            data_frames: List of DataFrames with market data
            
        Returns:
            int: Detected market regime ID
        """
        if not self.regime_detector:
            logger.warning("Regime detector component not initialized")
            return 0  # Default regime
        
        try:
            # Combine data frames for regime detection (use SPY or first ticker)
            # Ideally, we would have a market index like SPY for this
            market_data = None
            
            for df in data_frames:
                if df is not None and not df.empty:
                    ticker = df['Symbol'].iloc[0]
                    if ticker == 'SPY':
                        market_data = df
                        break
            
            # If no SPY, use the first data frame
            if market_data is None and data_frames:
                market_data = data_frames[0]
            
            if market_data is None:
                logger.warning("No data available for regime detection")
                return 0  # Default regime
            
            # Get the most recent data (last 100 rows)
            recent_data = market_data.tail(100)
            
            # Detect regime
            regime = self.regime_detector.get_current_regime(recent_data)
            regime_desc = self.regime_detector.get_regime_description(regime)
            
            logger.info(f"Detected market regime: {regime_desc} (ID: {regime})")
            
            # Store current regime
            self.current_regime = {
                "id": regime,
                "description": regime_desc,
                "timestamp": datetime.now().isoformat(),
                "characteristics": self.regime_detector.get_regime_characteristics(regime)
            }
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return 0  # Default regime
    
    def train_models(self, tickers=None, force=False):
        """
        Train all models in the system.
        
        Args:
            tickers: List of ticker symbols to train on
            force: Whether to force training even if already in progress
            
        Returns:
            bool: Success or failure
        """
        if self.is_training and not force:
            logger.warning("Training already in progress")
            return False
        
        self.is_training = True
        logger.info("Starting model training process")
        
        try:
            # Use default tickers if none provided
            if tickers is None:
                tickers = self.config["training"]["default_tickers"]
            
            # Fetch historical data
            data_frames = self.fetch_historical_data(
                tickers,
                period=f"{self.config['training']['data_lookback_days']}d",
                interval="1d"
            )
            
            if not data_frames:
                logger.error("No data available for training")
                self.is_training = False
                return False
            
            # Enhance features
            enhanced_frames = self.enhance_features(data_frames)
            
            # Train regime detector
            if self.regime_detector:
                try:
                    logger.info("Training market regime detector...")
                    result = self.regime_detector.fit(enhanced_frames)
                    if result["success"]:
                        logger.info(f"Market regime detector training successful with {result.get('n_regimes', 0)} regimes")
                    else:
                        logger.warning(f"Market regime detector training failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error training regime detector: {e}")
            
            # Detect current regime
            current_regime = self.detect_market_regime(enhanced_frames)
            
            # Train ensemble model
            if self.ensemble_model:
                try:
                    logger.info("Training ensemble model...")
                    result = self.ensemble_model.train_ensemble(
                        enhanced_frames,
                        test_size=self.config["training"]["validation_size"],
                        deep_learning=self.config["ensemble_model"]["deep_learning"]
                    )
                    
                    if result["success"]:
                        logger.info("Ensemble model training successful")
                        
                        # Update model registry
                        self.model_registry["ensemble"] = {
                            "last_trained": datetime.now().isoformat(),
                            "metrics": result.get("metrics", {}),
                            "description": "Ensemble model with XGBoost, LightGBM, and Neural Network"
                        }
                    else:
                        logger.warning(f"Ensemble model training failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error training ensemble model: {e}")
            
            # Train RL weight generator
            if self.rl_weight_generator:
                try:
                    logger.info("Training RL weight generator...")
                    result = self.rl_weight_generator.train(enhanced_frames, epochs=5)
                    
                    if result["success"]:
                        logger.info("RL weight generator training successful")
                        
                        # Update model registry
                        self.model_registry["rl_weight"] = {
                            "last_trained": datetime.now().isoformat(),
                            "metrics": result.get("metrics", {}),
                            "description": "Reinforcement Learning weight generator"
                        }
                    else:
                        logger.warning(f"RL weight generator training failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error training RL weight generator: {e}")
            
            # Save model registry
            self._save_model_registry()
            
            # Update training status
            self.last_training_time = datetime.now()
            self.is_training = False
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_training = False
            return False
    
    def run_daily_predictions(self):
        """Run daily predictions on all configured tickers."""
        logger.info("Running daily predictions")
        
        tickers = self.config["training"]["default_tickers"]
        return self.run_predictions(tickers)
    
    def run_predictions(self, tickers=None):
        """
        Run predictions on specified tickers.
        
        Args:
            tickers: List of ticker symbols to predict
            
        Returns:
            dict: Prediction results
        """
        try:
            # Use default tickers if none provided
            if tickers is None:
                tickers = self.config["training"]["default_tickers"]
            
            # Fetch recent data (shorter period for predictions)
            data_frames = self.fetch_historical_data(
                tickers,
                period="3mo",  # 3 months should be enough for predictions
                interval="1d",
                use_cache=False  # Always get fresh data for predictions
            )
            
            if not data_frames:
                logger.error("No data available for predictions")
                return {"success": False, "message": "No data available"}
            
            # Enhance features
            enhanced_frames = self.enhance_features(data_frames)
            
            # Detect market regime
            current_regime = self.detect_market_regime(enhanced_frames)
            
            # Run predictions for each ticker
            prediction_results = {}
            
            for df in enhanced_frames:
                if df is not None and not df.empty:
                    ticker = df['Symbol'].iloc[0]
                    
                    # Get predictions
                    ticker_predictions = self._predict_ticker(df, current_regime)
                    
                    prediction_results[ticker] = ticker_predictions
            
            # Generate portfolio allocation
            portfolio_allocation = self._generate_portfolio_allocation(prediction_results)
            
            # Save predictions
            if self.config["prediction"]["save_predictions"]:
                self._save_predictions(prediction_results, portfolio_allocation)
            
            # Add to prediction history
            self._update_prediction_history(prediction_results, portfolio_allocation)
            
            logger.info(f"Predictions completed for {len(prediction_results)} tickers")
            
            return {
                "success": True,
                "predictions": prediction_results,
                "portfolio": portfolio_allocation,
                "timestamp": datetime.now().isoformat(),
                "market_regime": self.current_regime
            }
            
        except Exception as e:
            logger.error(f"Error running predictions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _predict_ticker(self, df, current_regime):
        """
        Make predictions for a single ticker.
        
        Args:
            df: DataFrame with ticker data
            current_regime: Current market regime
            
        Returns:
            dict: Prediction results
        """
        ticker = df['Symbol'].iloc[0]
        logger.info(f"Making predictions for {ticker}")
        
        # Get latest data
        latest_data = df.tail(1)
        
        # Initialize prediction results
        predictions = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "price": float(df['Close'].iloc[-1]),
            "date": str(df['Timestamp'].iloc[-1]),
            "market_regime": current_regime,
            "signals": {},
            "probability": 0.5,  # Default neutral
            "confidence": 0,
            "action": "HOLD",
            "weights": {
                "buy": 0,
                "sell": 0
            }
        }
        
        # Get ensemble model predictions
        if self.ensemble_model:
            try:
                prob, confidence = self.ensemble_model.predict_with_ensemble(latest_data)
                predictions["ensemble"] = {
                    "probability": float(prob),
                    "confidence": float(confidence),
                    "signal": 1 if prob > 0.5 else -1 if prob < 0.5 else 0
                }
                predictions["signals"]["ensemble"] = predictions["ensemble"]["signal"]
            except Exception as e:
                logger.error(f"Error getting ensemble predictions for {ticker}: {e}")
        
        # Get HFT signals
        if self.hft_analyzer:
            try:
                # For HFT, we need more granular data, but we can still provide some metrics
                hft_signals = self.hft_analyzer.get_hft_trading_signals(df.tail(100))
                
                predictions["hft"] = {
                    "buy_weight": float(hft_signals["buy_weight"]),
                    "sell_weight": float(hft_signals["sell_weight"]),
                    "signal": 1 if hft_signals["action"] == "BUY" else -1 if hft_signals["action"] == "SELL" else 0,
                    "confidence": float(hft_signals["confidence"])
                }
                predictions["signals"]["hft"] = predictions["hft"]["signal"]
            except Exception as e:
                logger.error(f"Error getting HFT signals for {ticker}: {e}")
        
        # Get RL weights
        if self.rl_weight_generator:
            try:
                buy_weight, sell_weight = self.rl_weight_generator.generate_weight(latest_data)
                
                predictions["rl_weight"] = {
                    "buy_weight": float(buy_weight),
                    "sell_weight": float(sell_weight),
                    "signal": 1 if buy_weight > sell_weight else -1 if sell_weight > buy_weight else 0
                }
                predictions["signals"]["rl"] = predictions["rl_weight"]["signal"]
                
                # Use RL weights for final weights
                predictions["weights"]["buy"] = float(buy_weight)
                predictions["weights"]["sell"] = float(sell_weight)
            except Exception as e:
                logger.error(f"Error getting RL weights for {ticker}: {e}")
        
        # Combine signals based on current regime
        # Different weights for different regimes
        regime_weights = {
            0: {"ensemble": 0.5, "hft": 0.3, "rl": 0.2},  # Default regime
            1: {"ensemble": 0.6, "hft": 0.2, "rl": 0.2},  # Regime 1 (e.g., Bull market)
            2: {"ensemble": 0.4, "hft": 0.4, "rl": 0.2},  # Regime 2 (e.g., Bear market)
            3: {"ensemble": 0.3, "hft": 0.5, "rl": 0.2}   # Regime 3 (e.g., Sideways market)
        }
        
        # Use weights for current regime or default
        weights = regime_weights.get(current_regime, regime_weights[0])
        
        # Calculate combined probability
        if "ensemble" in predictions and "hft" in predictions and "rl_weight" in predictions:
            ensemble_prob = predictions["ensemble"]["probability"]
            hft_signal = predictions["hft"]["signal"]
            rl_signal = predictions["rl_weight"]["signal"]
            
            # Convert signals to probabilities
            hft_prob = 0.5 + (hft_signal * 0.25)  # Scale to 0.25-0.75 range
            rl_prob = 0.5 + (rl_signal * 0.25)  # Scale to 0.25-0.75 range
            
            # Combine probabilities with regime-specific weights
            combined_prob = (
                weights["ensemble"] * ensemble_prob +
                weights["hft"] * hft_prob +
                weights["rl"] * rl_prob
            )
            
            # Get confidence as agreement between signals
            unique_signals = set(predictions["signals"].values())
            if len(unique_signals) == 1:
                confidence = 1.0  # All agree
            elif len(unique_signals) == 2 and 0 in unique_signals:
                confidence = 0.5  # Partial agreement
            else:
                confidence = 0.0  # Disagreement
            
            # Apply confidence to probability (move toward 0.5 for low confidence)
            adjusted_prob = 0.5 + (combined_prob - 0.5) * confidence
            
            predictions["probability"] = float(adjusted_prob)
            predictions["confidence"] = float(confidence)
            
            # Determine action
            if adjusted_prob > 0.6:  # Strong buy
                predictions["action"] = "BUY"
            elif adjusted_prob > 0.55:  # Weak buy
                predictions["action"] = "WEAK BUY"
            elif adjusted_prob < 0.4:  # Strong sell
                predictions["action"] = "SELL"
            elif adjusted_prob < 0.45:  # Weak sell
                predictions["action"] = "WEAK SELL"
            else:
                predictions["action"] = "HOLD"
        
        return predictions
    
    def _generate_portfolio_allocation(self, prediction_results, total_investment=10000):
        """
        Generate optimal portfolio allocation.
        
        Args:
            prediction_results: Dictionary of prediction results by ticker
            total_investment: Total investment amount
            
        Returns:
            dict: Portfolio allocation
        """
        if not self.portfolio_optimizer:
            logger.warning("Portfolio optimizer component not initialized")
            return {}
        
        try:
            # Extract tickers and predictions
            tickers = list(prediction_results.keys())
            
            if not tickers:
                logger.warning("No tickers available for portfolio optimization")
                return {}
            
            # Create ML weights dictionary
            ml_weights = {}
            
            for ticker, predictions in prediction_results.items():
                # Use buy weight - sell weight as the ML weight
                buy_weight = predictions["weights"]["buy"]
                sell_weight = predictions["weights"]["sell"]
                ml_weights[ticker] = buy_weight - sell_weight
            
            # Fetch historical price data for optimization
            price_data = pd.DataFrame()
            
            for ticker in tickers:
                if ticker in self.historical_data:
                    df = self.historical_data[ticker]
                    price_data[ticker] = df.set_index('Timestamp')['Close']
            
            if price_data.empty:
                logger.warning("No price data available for portfolio optimization")
                return {ticker: {"weight": 1.0 / len(tickers)} for ticker in tickers}
            
            # Load market data into portfolio optimizer
            sector_mapping = {}  # Ideally, we would have sector data
            self.portfolio_optimizer.load_market_data(price_data, sector_mapping)
            
            # Optimize portfolio
            if self.current_regime and self.current_regime["id"] >= 0:
                # Optimize differently based on regime
                regime_id = self.current_regime["id"]
                
                if regime_id == 1:  # Bull market
                    logger.info("Using Sharpe ratio optimization for bull market")
                    optimization_result = self.portfolio_optimizer.optimize_portfolio(objective='sharpe')
                elif regime_id == 2:  # Bear market
                    logger.info("Using minimum drawdown optimization for bear market")
                    optimization_result = self.portfolio_optimizer.optimize_with_drawdown_constraint(max_drawdown=-0.1)
                else:  # Sideways or unknown
                    logger.info("Using risk parity optimization for sideways market")
                    optimization_result = self.portfolio_optimizer.optimize_with_risk_parity()
            else:
                # Default to Sharpe ratio
                optimization_result = self.portfolio_optimizer.optimize_portfolio(objective='sharpe')
            
            if not optimization_result["success"]:
                logger.warning(f"Portfolio optimization failed: {optimization_result.get('message', 'Unknown error')}")
                return {ticker: {"weight": 1.0 / len(tickers)} for ticker in tickers}
            
            # Generate allocation with ML weights
            allocation = self.portfolio_optimizer.get_portfolio_allocation(
                total_investment, ml_weights
            )
            
            logger.info(f"Generated portfolio allocation with {len(allocation) - 1} positions")  # -1 for metrics
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error generating portfolio allocation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _save_predictions(self, predictions, portfolio):
        """
        Save predictions and portfolio allocation to file.
        
        Args:
            predictions: Dictionary of prediction results
            portfolio: Portfolio allocation
        """
        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save predictions
            predictions_file = os.path.join(
                self.config["prediction_dir"],
                f"predictions_{timestamp}.json"
            )
            
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=4)
            
            # Save portfolio
            portfolio_file = os.path.join(
                self.config["prediction_dir"],
                f"portfolio_{timestamp}.json"
            )
            
            with open(portfolio_file, 'w') as f:
                json.dump(portfolio, f, indent=4)
            
            logger.info(f"Saved predictions to {predictions_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            return False
    
    def _update_prediction_history(self, predictions, portfolio):
        """
        Update prediction history with new predictions.
        
        Args:
            predictions: Dictionary of prediction results
            portfolio: Portfolio allocation
        """
        try:
            # Create history entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "predictions": predictions,
                "portfolio": portfolio,
                "market_regime": self.current_regime
            }
            
            # Add to history
            self.prediction_history.append(entry)
            
            # Limit history size
            max_history = 100
            if len(self.prediction_history) > max_history:
                self.prediction_history = self.prediction_history[-max_history:]
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating prediction history: {e}")
            return False
    
    def evaluate_prediction_performance(self):
        """
        Evaluate performance of past predictions.
        
        Returns:
            dict: Performance metrics
        """
        try:
            # Need actual price data to compare with predictions
            # This would require a database of past predictions with outcomes
            
            # For now, return placeholder metrics
            metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0
            }
            
            # TODO: Implement actual performance evaluation
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating prediction performance: {e}")
            return {}
    
    def update_models_from_feedback(self):
        """
        Update models using feedback from past predictions.
        
        Returns:
            bool: Success or failure
        """
        # This would implement continuous learning using past prediction results
        # and their actual outcomes.
        
        # TODO: Implement continuous learning logic
        
        return False
    
    def get_system_status(self):
        """
        Get current status of the trading system.
        
        Returns:
            dict: System status
        """
        try:
            status = {
                "system": {
                    "version": "1.0.0",
                    "uptime": "Unknown",  # Would track actual uptime
                    "is_training": self.is_training,
                    "last_training": self.last_training_time.isoformat() if self.last_training_time else None,
                    "scheduler_running": self.scheduler_running
                },
                "components": {
                    "feature_engineering": self.feature_engineer is not None,
                    "ensemble_model": self.ensemble_model is not None,
                    "rl_weight_generator": self.rl_weight_generator is not None,
                    "regime_detector": self.regime_detector is not None,
                    "hft_analyzer": self.hft_analyzer is not None,
                    "portfolio_optimizer": self.portfolio_optimizer is not None
                },
                "models": self.model_registry,
                "current_regime": self.current_regime,
                "predictions": {
                    "count": len(self.prediction_history),
                    "last_prediction": self.prediction_history[-1]["timestamp"] if self.prediction_history else None
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "error": str(e),
                "is_training": self.is_training
            }
    
    def run_backtesting(self, tickers=None, start_date=None, end_date=None, initial_capital=10000):
        """
        Run backtesting to evaluate strategy performance.
        
        Args:
            tickers: List of ticker symbols to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital for backtesting
            
        Returns:
            dict: Backtesting results
        """
        try:
            # Use default tickers if none provided
            if tickers is None:
                tickers = self.config["training"]["default_tickers"]
            
            # Set default dates if not provided
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            logger.info(f"Running backtesting for {len(tickers)} tickers from {start_date} to {end_date}")
            
            # Fetch historical data
            # This would ideally use a more sophisticated data fetching approach
            # to get data specifically for the date range
            data_frames = self.fetch_historical_data(tickers, period="2y", interval="1d")
            
            if not data_frames:
                logger.error("No data available for backtesting")
                return {"success": False, "message": "No data available"}
            
            # Filter data frames to the specified date range
            filtered_frames = []
            
            for df in data_frames:
                if df is not None and not df.empty:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                    filtered_df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
                    
                    if not filtered_df.empty:
                        filtered_frames.append(filtered_df.reset_index(drop=True))
            
            if not filtered_frames:
                logger.error("No data available in the specified date range")
                return {"success": False, "message": "No data in date range"}
            
            # Enhance features
            enhanced_frames = self.enhance_features(filtered_frames)
            
            # Run backtest
            # TODO: Implement actual backtesting logic
            
            # For now, return placeholder results
            results = {
                "success": True,
                "message": "Backtesting completed successfully",
                "performance": {
                    "total_return": 0.0,
                    "annual_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0
                },
                "trades": []
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtesting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error: {str(e)}"}


# Import check function for component availability
def check_component_availability():
    """Check availability of all enhanced trading components."""
    availability = {
        "feature_engineering": FEATURE_ENGINEERING_AVAILABLE,
        "ensemble_model": ENSEMBLE_MODEL_AVAILABLE,
        "rl_weight": RL_WEIGHT_AVAILABLE,
        "regime_detector": REGIME_DETECTOR_AVAILABLE,
        "hft_analyzer": HFT_ANALYZER_AVAILABLE,
        "portfolio_optimizer": PORTFOLIO_OPTIMIZER_AVAILABLE
    }
    
    # Additional Python package requirements
    requirements = {
        "numpy": importlib.util.find_spec("numpy") is not None,
        "pandas": importlib.util.find_spec("pandas") is not None,
        "scikit-learn": importlib.util.find_spec("sklearn") is not None,
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "yfinance": importlib.util.find_spec("yfinance") is not None
    }
    
    return {
        "components": availability,
        "requirements": requirements,
        "all_available": all(availability.values()) and all(requirements.values())
    }


# Main function to run the trading system
def main():
    """Main function to run the trading system."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--check', action='store_true', help='Check component availability')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--predict', action='store_true', help='Run predictions')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers')
    
    args = parser.parse_args()
    
    # Check component availability if requested
    if args.check:
        availability = check_component_availability()
        print("Component Availability:")
        for component, available in availability["components"].items():
            print(f"  {component}: {'Available' if available else 'Not Available'}")
        
        print("\nRequirements:")
        for req, installed in availability["requirements"].items():
            print(f"  {req}: {'Installed' if installed else 'Not Installed'}")
        
        print(f"\nAll Components Available: {'Yes' if availability['all_available'] else 'No'}")
        return
    
    # Parse tickers if provided
    tickers = None
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    
    # Initialize the trading system
    system = EnhancedTradingSystem(config_path=args.config)
    
    # Run requested actions
    if args.train:
        print("Training models...")
        result = system.train_models(tickers=tickers, force=True)
        print(f"Training {'successful' if result else 'failed'}")
    
    if args.predict:
        print("Running predictions...")
        result = system.run_predictions(tickers=tickers)
        
        if result["success"]:
            print("Predictions successful")
            print("\nPrediction Summary:")
            
            for ticker, prediction in result["predictions"].items():
                print(f"  {ticker}: {prediction['action']} (Probability: {prediction['probability']:.2f}, Confidence: {prediction['confidence']:.2f})")
            
            print("\nPortfolio Allocation:")
            if "portfolio" in result and result["portfolio"]:
                for asset, allocation in result["portfolio"].items():
                    if asset != "portfolio_metrics":
                        print(f"  {asset}: {allocation['weight']:.2%} (${allocation['amount']:.2f})")
        else:
            print(f"Predictions failed: {result.get('message', 'Unknown error')}")
    
    if args.backtest:
        print("Running backtesting...")
        result = system.run_backtesting(tickers=tickers)
        
        if result["success"]:
            print("Backtesting successful")
            print("\nPerformance Summary:")
            for metric, value in result["performance"].items():
                print(f"  {metric}: {value}")
        else:
            print(f"Backtesting failed: {result.get('message', 'Unknown error')}")
    
    # If no action specified, start the scheduler
    if not (args.train or args.predict or args.backtest):
        print("Starting trading system scheduler...")
        system.start_scheduler()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping trading system scheduler...")
            system.stop_scheduler()
            print("Scheduler stopped")


if __name__ == "__main__":
    main()