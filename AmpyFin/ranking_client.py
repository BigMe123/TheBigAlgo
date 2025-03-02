#!/usr/bin/env python
# Enhanced Ranking Client - Integrates all components including ML signal integrator and sentiment analysis
import os
import sys
import logging
import pandas as pd
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
import math
import json
import traceback
import heapq
import certifi
from pymongo import MongoClient
from collections import Counter
import yfinance as yf
import requests

# Config imports
try:
    from config import (
        POLYGON_API_KEY, FINANCIAL_PREP_API_KEY, MONGO_DB_USER, MONGO_DB_PASS, 
        API_KEY, API_SECRET, BASE_URL, mongo_url
    )
except ImportError:
    # Default values if config not available
    POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY')
    FINANCIAL_PREP_API_KEY = os.environ.get('FINANCIAL_PREP_API_KEY')
    API_KEY = os.environ.get('ALPACA_API_KEY')
    API_SECRET = os.environ.get('ALPACA_API_SECRET')
    BASE_URL = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    mongo_url = os.environ.get('MONGO_URL')

# Import trading helper functions
try:
    from helper_files.client_helper import strategies, get_latest_price, get_ndaq_tickers, dynamic_period_selector
except ImportError:
    logging.warning("Could not import helper_files.client_helper. Using dummy functions.")
    # Dummy functions if imports fail
    strategies = []
    def get_latest_price(ticker):
        """Dummy function to get price from yfinance"""
        try:
            ticker_data = yf.Ticker(ticker)
            return ticker_data.history(period='1d').iloc[-1]['Close']
        except:
            return None
            
    def get_ndaq_tickers(mongo_client, api_key):
        """Dummy function to get NASDAQ tickers"""
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
        
    def dynamic_period_selector(ticker, mongo_client):
        """Dummy function for period selection"""
        return "1y"

# Import signal integrator from deployed model
try:
    from deployed_model.signal_integrator import integrate_with_ranking_client
except ImportError:
    def integrate_with_ranking_client(ticker, data, mongo_client=None):
        """Stub function if signal integrator not available"""
        logging.warning(f"Signal integrator not available for {ticker}")
        return {"action": "HOLD", "confidence": 0, "buy_weight": 0, "sell_weight": 0}

# Import sentiment analyzer
try:
    from news_analyzer import integrate_with_signal_integrator
except ImportError:
    def integrate_with_signal_integrator(ticker, signal_integrator=None, mongo_url=None):
        """Stub function if sentiment analyzer not available"""
        logging.warning(f"Sentiment analyzer not available for {ticker}")
        return 0, 0

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('enhanced_ranking.log'),
        logging.StreamHandler()
    ]
)

class EnhancedRankingClient:
    """
    Enhanced Ranking Client that integrates all components:
    - Strategy-based ranking
    - Machine learning signal integrator
    - News sentiment analyzer
    - Market regime detection
    - Dynamic simulation and evaluation
    """
    
    def __init__(self, mongo_url=None, max_workers=8):
        """
        Initialize the enhanced ranking client.
        
        Args:
            mongo_url: MongoDB connection URL
            max_workers: Maximum number of worker threads
        """
        self.mongo_url = mongo_url
        self.mongo_client = None
        self.max_workers = max_workers
        
        # Component influence weights
        self.component_weights = {
            "strategies": 0.5,    # Base strategy algorithms
            "ml_signals": 0.3,    # Machine learning signals
            "sentiment": 0.2      # News and social sentiment
        }
        
        # Connect to MongoDB
        self._connect_mongodb()
        
        # Fetch and store NASDAQ tickers
        self.tickers = []
        
        # Strategy coefficients cache
        self.strategy_coefficients = {}
        
        # Market status
        self.market_status = "closed"
        
        # Performance statistics
        self.performance_stats = {
            "tickers_processed": 0,
            "success_rate": 0,
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "neutral_trades": 0,
            "avg_processing_time": 0
        }
        
        # Threading
        self.stop_event = threading.Event()
        self.main_thread = None
        
        logging.info("Enhanced Ranking Client initialized")
    
    def _connect_mongodb(self):
        """Connect to MongoDB database."""
        if not self.mongo_url:
            logging.warning("MongoDB URL not available")
            return False
        
        try:
            # Connect to MongoDB
            ca = certifi.where()
            self.mongo_client = MongoClient(self.mongo_url, tlsCAFile=ca)
            
            # Test connection
            self.mongo_client.server_info()
            logging.info("Connected to MongoDB successfully")
            return True
        except Exception as e:
            logging.error(f"Error connecting to MongoDB: {e}")
            self.mongo_client = None
            return False
    
    def get_historical_data(self, ticker, period='1y'):
        """
        Get historical data for a ticker.
        First checks cache in MongoDB, then fetches from yfinance if not cached.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period to fetch
            
        Returns:
            DataFrame with historical stock data or list of dicts
        """
        try:
            # Check if data is in MongoDB cache
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
                    logging.info(f"Using cached data for {ticker} from MongoDB")
                    return cached_data['data']
            
            # If not in cache or no MongoDB, fetch from yfinance
            logging.info(f"Fetching {ticker} data from yfinance for period {period}")
            ticker_data = yf.Ticker(ticker)
            historical_data = ticker_data.history(period=period)
            
            # Reset index to make date a column
            historical_data = historical_data.reset_index()
            
            # Rename columns if needed
            if 'Date' in historical_data.columns and 'Timestamp' not in historical_data.columns:
                historical_data = historical_data.rename(columns={'Date': 'Timestamp'})
            
            # Add Symbol column if not present
            if 'Symbol' not in historical_data.columns:
                historical_data['Symbol'] = ticker
            
            # Store in MongoDB for future use if available
            if self.mongo_client:
                try:
                    collection.update_one(
                        {"ticker": ticker, "period": period},
                        {"$set": {
                            "data": historical_data.to_dict('records'),
                            "timestamp": datetime.now()
                        }},
                        upsert=True
                    )
                except Exception as e:
                    logging.error(f"Error storing {ticker} data in MongoDB: {e}")
            
            return historical_data.to_dict('records')
        except Exception as e:
            logging.error(f"Error getting historical data for {ticker}: {e}")
            return None
    
    def get_strategy_coefficients(self):
        """
        Get strategy coefficients based on rank.
        Coefficients determine how much weight each strategy has.
        
        Returns:
            dict: Strategy name to coefficient mapping
        """
        # Return cached coefficients if available
        if self.strategy_coefficients:
            return self.strategy_coefficients
        
        # Default coefficients if MongoDB not available
        default_coefficients = {
            strategy.__name__: 1.0 for strategy in strategies
        }
        
        if not self.mongo_client:
            logging.warning("Using default strategy coefficients (MongoDB not available)")
            self.strategy_coefficients = default_coefficients
            return default_coefficients
        
        try:
            # Get strategy rankings
            sim_db = self.mongo_client.trading_simulator
            rank_collection = sim_db.rank
            r_t_c_collection = sim_db.rank_to_coefficient
            
            coefficients = {}
            for strategy in strategies:
                rank_doc = rank_collection.find_one({'strategy': strategy.__name__})
                if not rank_doc:
                    # Use default value if strategy not found
                    coefficients[strategy.__name__] = 1.0
                    continue
                
                rank = rank_doc['rank']
                coef_doc = r_t_c_collection.find_one({'rank': rank})
                if not coef_doc:
                    # Use default value if coefficient not found
                    coefficients[strategy.__name__] = 1.0
                    continue
                
                coefficients[strategy.__name__] = coef_doc['coefficient']
            
            # Cache and return
            self.strategy_coefficients = coefficients
            logging.info(f"Loaded strategy coefficients for {len(coefficients)} strategies")
            return coefficients
        except Exception as e:
            logging.error(f"Error getting strategy coefficients: {e}")
            self.strategy_coefficients = default_coefficients
            return default_coefficients
    
    def detect_market_status(self):
        """
        Detect current market status.
        
        Returns:
            str: Market status ("open", "early_hours", "closed")
        """
        if not self.mongo_client:
            # Default to open during market hours as fallback
            now = datetime.now()
            if now.weekday() < 5 and 9 <= now.hour < 16:  # Weekday between 9am and 4pm
                return "open"
            return "closed"
        
        try:
            market_db = self.mongo_client.market_data
            if 'market_status' in market_db.list_collection_names():
                status_doc = market_db.market_status.find_one({})
                if status_doc and 'market_status' in status_doc:
                    return status_doc['market_status']
            
            # Default behavior if no status in DB
            return "closed"
        except Exception as e:
            logging.error(f"Error detecting market status: {e}")
            return "closed"
    
    def get_signal_integrator_recommendation(self, ticker, data):
        """
        Get recommendation from ML signal integrator.
        
        Args:
            ticker: Stock ticker symbol
            data: Historical data for the ticker
            
        Returns:
            dict: Signal integrator recommendation
        """
        try:
            # Call signal integrator
            signal_result = integrate_with_ranking_client(ticker, data, self.mongo_client)
            logging.info(f"Signal integrator for {ticker}: {signal_result.get('action', 'HOLD')} (conf={signal_result.get('confidence', 0):.2f})")
            return signal_result
        except Exception as e:
            logging.error(f"Error getting signal integrator recommendation for {ticker}: {e}")
            return {"action": "HOLD", "confidence": 0, "buy_weight": 0, "sell_weight": 0}
    
    def get_sentiment_recommendation(self, ticker):
        """
        Get recommendation from news sentiment analyzer.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            tuple: (buy_weight, sell_weight) from sentiment analysis
        """
        try:
            # Call sentiment analyzer
            buy_weight, sell_weight = integrate_with_signal_integrator(ticker, mongo_url=self.mongo_url)
            if buy_weight > 0 or sell_weight > 0:
                action = "BUY" if buy_weight > sell_weight else "SELL" if sell_weight > buy_weight else "HOLD"
                confidence = abs(buy_weight - sell_weight) / max(buy_weight, sell_weight, 1)
                logging.info(f"Sentiment for {ticker}: {action} (buy={buy_weight:.1f}, sell={sell_weight:.1f}, conf={confidence:.2f})")
            return buy_weight, sell_weight
        except Exception as e:
            logging.error(f"Error getting sentiment recommendation for {ticker}: {e}")
            return 0, 0
    
    def simulate_strategy(self, strategy, ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
        """
        Simulates a trading strategy and returns the decision and quantity.
        
        Args:
            strategy: Strategy function to simulate
            ticker: Stock ticker symbol
            current_price: Current price of the stock
            historical_data: Historical data for the ticker
            account_cash: Available cash in the account
            portfolio_qty: Current quantity of this stock in the portfolio
            total_portfolio_value: Total portfolio value
            
        Returns:
            tuple: (decision, quantity)
        """
        try:
            # Convert historical_data from list to DataFrame if needed
            if isinstance(historical_data, list):
                historical_data = pd.DataFrame(historical_data)
            
            result = strategy(ticker, historical_data, current_price, account_cash, portfolio_qty, total_portfolio_value)
            action = result.get('decision', 'hold').lower()
            quantity = result.get('quantity', 0)
            return action, quantity
        except Exception as e:
            logging.error(f"Error simulating {strategy.__name__} for {ticker}: {e}")
            return 'hold', 0
    
    def process_ticker(self, ticker):
        """
        Process a ticker using all available components.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            dict: Processing result
        """
        start_time = time.time()
        
        try:
            # Initialize result dict
            result = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "current_price": None,
                "decision": "hold",
                "quantity": 0,
                "confidence": 0,
                "buy_weight": 0,
                "sell_weight": 0,
                "processing_time": 0,
                "components": {}
            }
            
            # Get current price
            current_price = None
            retry_count = 0
            while current_price is None and retry_count < 3:
                try:
                    current_price = get_latest_price(ticker)
                except Exception as fetch_error:
                    logging.warning(f"Error fetching price for {ticker}. Retrying... {fetch_error}")
                    time.sleep(5)
                    retry_count += 1
            
            if current_price is None:
                logging.error(f"Failed to get price for {ticker} after {retry_count} attempts")
                result["error"] = "Failed to get current price"
                return result
            
            result["current_price"] = current_price
            logging.info(f"Current price for {ticker}: ${current_price:.2f}")
            
            # Get ML Signal Integrator period preference
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
                logging.warning(f"Error getting ideal period for {ticker}: {e}. Using default 1y.")
            
            # Get historical data
            historical_data = self.get_historical_data(ticker, period)
            if not historical_data:
                logging.error(f"Failed to get historical data for {ticker}")
                result["error"] = "Failed to get historical data"
                return result
            
            # 1. Get ML signal integrator recommendation
            signal_result = self.get_signal_integrator_recommendation(ticker, historical_data)
            
            # 2. Get sentiment recommendation
            sentiment_buy, sentiment_sell = self.get_sentiment_recommendation(ticker)
            
            # 3. Process each strategy
            account_info = {}
            strategy_decisions = []
            
            # Get strategy coefficients
            strategy_coefficients = self.get_strategy_coefficients()
            
            for strategy in strategies:
                try:
                    # Get strategy-specific period if available
                    strategy_period = period
                    try:
                        if self.mongo_client:
                            indicator_collection = self.mongo_client.IndicatorsDatabase.Indicators
                            period_doc = indicator_collection.find_one({'indicator': strategy.__name__})
                            if period_doc and 'ideal_period' in period_doc:
                                strategy_period = period_doc['ideal_period']
                    except Exception as e:
                        pass  # Use default period if error
                    
                    # Get strategy-specific data if needed
                    if strategy_period != period:
                        strategy_data = self.get_historical_data(ticker, strategy_period)
                        if not strategy_data:
                            logging.warning(f"Failed to get {strategy_period} data for {ticker} and {strategy.__name__}, using default data")
                            strategy_data = historical_data
                    else:
                        strategy_data = historical_data
                    
                    # Get account info for this strategy
                    if not account_info:
                        try:
                            if self.mongo_client:
                                db = self.mongo_client.trading_simulator
                                holdings_collection = db.algorithm_holdings
                                strategy_doc = holdings_collection.find_one({"strategy": strategy.__name__})
                                if strategy_doc:
                                    account_cash = strategy_doc["amount_cash"]
                                    total_portfolio_value = strategy_doc["portfolio_value"]
                                    portfolio_qty = strategy_doc["holdings"].get(ticker, {}).get("quantity", 0)
                                    
                                    account_info = {
                                        "account_cash": account_cash,
                                        "total_portfolio_value": total_portfolio_value,
                                        "portfolio_qty": portfolio_qty
                                    }
                        except Exception as e:
                            logging.error(f"Error getting account info for {strategy.__name__}: {e}")
                    
                    if not account_info:
                        # Default values if no account info available
                        account_info = {
                            "account_cash": 1000000,
                            "total_portfolio_value": 2000000,
                            "portfolio_qty": 0
                        }
                    
                    # Simulate strategy
                    decision, quantity = self.simulate_strategy(
                        strategy,
                        ticker,
                        current_price,
                        strategy_data,
                        account_info["account_cash"],
                        account_info["portfolio_qty"],
                        account_info["total_portfolio_value"]
                    )
                    
                    # Get coefficient for this strategy
                    coefficient = strategy_coefficients.get(strategy.__name__, 1.0)
                    
                    # Add to strategy decisions
                    strategy_decisions.append({
                        "strategy": strategy.__name__,
                        "decision": decision,
                        "quantity": quantity,
                        "coefficient": coefficient
                    })
                    
                    logging.info(f"Strategy {strategy.__name__} for {ticker}: {decision.upper()} {quantity} (coef={coefficient:.2f})")
                    
                except Exception as e:
                    logging.error(f"Error processing {strategy.__name__} for {ticker}: {e}")
            
            # 4. Integrate all components using weighted decision system
            combined_result = self.integrate_components(
                ticker,
                current_price,
                strategy_decisions,
                signal_result,
                sentiment_buy,
                sentiment_sell,
                account_info.get("total_portfolio_value", 2000000)
            )
            
            # Update result with combined decision
            result.update(combined_result)
            
            # 5. Store in MongoDB if available
            if self.mongo_client:
                try:
                    db = self.mongo_client.ranking_decisions
                    if 'decisions' not in db.list_collection_names():
                        db.create_collection('decisions')
                    
                    db.decisions.update_one(
                        {"ticker": ticker},
                        {"$set": result},
                        upsert=True
                    )
                    
                    # Also store in signal_weights collection for trading client
                    trading_db = self.mongo_client.trades
                    if 'ranking_weights' not in trading_db.list_collection_names():
                        trading_db.create_collection('ranking_weights')
                    
                    trading_db.ranking_weights.update_one(
                        {"ticker": ticker},
                        {"$set": {
                            "ticker": ticker,
                            "buy_weight": result["buy_weight"],
                            "sell_weight": result["sell_weight"],
                            "action": result["decision"].upper(),
                            "confidence": result["confidence"],
                            "price": current_price,
                            "timestamp": datetime.now().isoformat()
                        }},
                        upsert=True
                    )
                except Exception as e:
                    logging.error(f"Error storing decision for {ticker} in MongoDB: {e}")
            
            # Calculate processing time
            result["processing_time"] = time.time() - start_time
            
            # Update performance statistics
            with threading.Lock():
                self.performance_stats["tickers_processed"] += 1
                self.performance_stats["avg_processing_time"] = (
                    (self.performance_stats["avg_processing_time"] * (self.performance_stats["tickers_processed"] - 1) + 
                     result["processing_time"]) / self.performance_stats["tickers_processed"]
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Error in process_ticker for {ticker}: {e}")
            logging.error(traceback.format_exc())
            
            # Return error result
            result = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "decision": "hold",
                "quantity": 0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            return result
    
    def integrate_components(self, ticker, current_price, strategy_decisions, signal_result, sentiment_buy, sentiment_sell, portfolio_value):
        """
        Integrate all components to make a final decision.
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current price of the stock
            strategy_decisions: List of strategy decisions
            signal_result: ML signal integrator result
            sentiment_buy: Sentiment buy weight
            sentiment_sell: Sentiment sell weight
            portfolio_value: Total portfolio value
            
        Returns:
            dict: Integrated decision
        """
        # Default neutral result
        result = {
            "decision": "hold",
            "quantity": 0,
            "confidence": 0,
            "buy_weight": 0,
            "sell_weight": 0,
            "components": {
                "strategies": {"decision": "hold", "weight": 0, "confidence": 0},
                "ml_signals": {"decision": "hold", "weight": 0, "confidence": 0},
                "sentiment": {"decision": "hold", "weight": 0, "confidence": 0}
            }
        }
        
        try:
            # 1. Process strategy decisions
            strategy_buy_weight = 0
            strategy_sell_weight = 0
            strategy_hold_weight = 0
            weighted_buy_quantities = []
            weighted_sell_quantities = []
            
            for decision in strategy_decisions:
                if decision["decision"].lower() in ["buy", "strong buy"]:
                    weighted_buy_quantities.append(decision["quantity"])
                    strategy_buy_weight += decision["coefficient"]
                elif decision["decision"].lower() in ["sell", "strong sell"]:
                    weighted_sell_quantities.append(decision["quantity"])
                    strategy_sell_weight += decision["coefficient"]
                else:
                    strategy_hold_weight += decision["coefficient"]
            
            total_strategy_weight = strategy_buy_weight + strategy_sell_weight + strategy_hold_weight
            if total_strategy_weight > 0:
                # Normalize weights
                strategy_buy_weight /= total_strategy_weight
                strategy_sell_weight /= total_strategy_weight
                strategy_hold_weight /= total_strategy_weight
            
            # Determine strategy component decision
            if strategy_buy_weight > strategy_sell_weight and strategy_buy_weight > strategy_hold_weight:
                strategy_decision = "buy"
                strategy_quantity = int(np.median(weighted_buy_quantities)) if weighted_buy_quantities else 0
                strategy_confidence = strategy_buy_weight
            elif strategy_sell_weight > strategy_buy_weight and strategy_sell_weight > strategy_hold_weight:
                strategy_decision = "sell"
                strategy_quantity = int(np.median(weighted_sell_quantities)) if weighted_sell_quantities else 0
                strategy_confidence = strategy_sell_weight
            else:
                strategy_decision = "hold"
                strategy_quantity = 0
                strategy_confidence = strategy_hold_weight
            
            # Update strategy component
            result["components"]["strategies"] = {
                "decision": strategy_decision,
                "quantity": strategy_quantity,
                "confidence": float(strategy_confidence),
                "buy_weight": float(strategy_buy_weight * 1000),  # Scale to 0-1000 range
                "sell_weight": float(strategy_sell_weight * 1000)
            }
            
            # 2. Process ML signal component
            ml_decision = signal_result.get("action", "HOLD").lower()
            ml_confidence = signal_result.get("confidence", 0)
            ml_buy_weight = signal_result.get("buy_weight", 0)
            ml_sell_weight = signal_result.get("sell_weight", 0)
            
            # Scale ML weights and determine ML quantity
            max_ml_investment = portfolio_value * 0.05  # Max 5% per position
            if ml_buy_weight > ml_sell_weight:
                ml_quantity = min(
                    int(max_ml_investment / current_price),
                    int(ml_buy_weight / current_price)
                )
            elif ml_sell_weight > ml_buy_weight:
                ml_quantity = min(
                    int(max_ml_investment / current_price),
                    int(ml_sell_weight / current_price)
                )
            else:
                ml_quantity = 0
            
            # Update ML component
            result["components"]["ml_signals"] = {
                "decision": ml_decision,
                "quantity": ml_quantity,
                "confidence": float(ml_confidence),
                "buy_weight": float(ml_buy_weight),
                "sell_weight": float(ml_sell_weight)
            }
            
            # 3. Process sentiment component
            if sentiment_buy > sentiment_sell:
                sentiment_decision = "buy"
                sentiment_confidence = min(1.0, sentiment_buy / 1000)
            elif sentiment_sell > sentiment_buy:
                sentiment_decision = "sell"
                sentiment_confidence = min(1.0, sentiment_sell / 1000)
            else:
                sentiment_decision = "hold"
                sentiment_confidence = 0
            
            # Determine sentiment quantity
            max_sentiment_investment = portfolio_value * 0.03  # Max 3% per position from sentiment
            if sentiment_decision == "buy":
                sentiment_quantity = min(
                    int(max_sentiment_investment / current_price),
                    int(sentiment_buy / current_price)
                )
            elif sentiment_decision == "sell":
                sentiment_quantity = min(
                    int(max_sentiment_investment / current_price),
                    int(sentiment_sell / current_price)
                )
            else:
                sentiment_quantity = 0
            
            # Update sentiment component
            result["components"]["sentiment"] = {
                "decision": sentiment_decision,
                "quantity": sentiment_quantity,
                "confidence": float(sentiment_confidence),
                "buy_weight": float(sentiment_buy),
                "sell_weight": float(sentiment_sell)
            }
            
            # 4. Combine all components using weighted voting
            buy_weight = (
                self.component_weights["strategies"] * result["components"]["strategies"]["buy_weight"] +
                self.component_weights["ml_signals"] * result["components"]["ml_signals"]["buy_weight"] +
                self.component_weights["sentiment"] * result["components"]["sentiment"]["buy_weight"]
            )
            
            sell_weight = (
                self.component_weights["strategies"] * result["components"]["strategies"]["sell_weight"] +
                self.component_weights["ml_signals"] * result["components"]["ml_signals"]["sell_weight"] +
                self.component_weights["sentiment"] * result["components"]["sentiment"]["sell_weight"]
            )
            
            # Determine confidence based on agreement between components
            decisions = [
                result["components"]["strategies"]["decision"],
                result["components"]["ml_signals"]["decision"],
                result["components"]["sentiment"]["decision"]
            ]
            decision_counts = Counter(decisions)
            most_common_decision, most_common_count = decision_counts.most_common(1)[0]
            agreement_confidence = most_common_count / len(decisions)
            
            # Determine final decision
            if buy_weight > sell_weight * 1.2:  # 20% threshold to reduce oscillation
                if buy_weight > 700 and agreement_confidence > 0.66:
                    final_decision = "strong buy"
                else:
                    final_decision = "buy"
            elif sell_weight > buy_weight * 1.2:  # 20% threshold to reduce oscillation
                if sell_weight > 700 and agreement_confidence > 0.66:
                    final_decision = "strong sell"
                else:
                    final_decision = "sell"
            else:
                final_decision = "hold"
            
            # Calculate final confidence
            final_confidence = agreement_confidence * 0.5 + (
                abs(buy_weight - sell_weight) / max(buy_weight + sell_weight, 1) * 0.5
            )
            
            # Calculate final quantity
            if final_decision in ["buy", "strong buy"]:
                # Average quantities from components weighted by confidence
                quantities = [
                    result["components"]["strategies"]["quantity"] * result["components"]["strategies"]["confidence"],
                    result["components"]["ml_signals"]["quantity"] * result["components"]["ml_signals"]["confidence"],
                    result["components"]["sentiment"]["quantity"] * result["components"]["sentiment"]["confidence"]
                ]
                confidences = [
                    result["components"]["strategies"]["confidence"],
                    result["components"]["ml_signals"]["confidence"],
                    result["components"]["sentiment"]["confidence"]
                ]
                
                if sum(confidences) > 0:
                    final_quantity = int(sum(quantities) / sum(confidences))
                else:
                    # Default quantity based on 2% of portfolio
                    final_quantity = int((portfolio_value * 0.02) / current_price)
                
                # Adjust for "strong buy"
                if final_decision == "strong buy":
                    final_quantity = int(final_quantity * 1.5)
            elif final_decision in ["sell", "strong sell"]:
                # For sell, calculate a percentage of position to sell
                sell_percentage = final_confidence
                if final_decision == "strong sell":
                    sell_percentage = min(1.0, sell_percentage * 1.5)  # Up to 100% for strong sell
                
                # Need to determine current position size - use strategy with highest coefficient
                position_size = 0
                if strategy_decisions:
                    best_strategy = max(strategy_decisions, key=lambda x: x["coefficient"])
                    portfolio_qty = best_strategy.get("portfolio_qty", 0)
                    position_size = portfolio_qty
                
                final_quantity = max(1, int(position_size * sell_percentage))
            else:
                final_quantity = 0
            
            # Adjust weights based on final decision for trading client
            if final_decision == "strong buy":
                buy_weight = max(buy_weight, 1000)
                sell_weight = 0
            elif final_decision == "buy":
                buy_weight = max(buy_weight, 500)
                sell_weight = min(sell_weight, 200)
            elif final_decision == "strong sell":
                sell_weight = max(sell_weight, 1000)
                buy_weight = 0
            elif final_decision == "sell":
                sell_weight = max(sell_weight, 500)
                buy_weight = min(buy_weight, 200)
            
            # Update final result
            result.update({
                "decision": final_decision,
                "quantity": final_quantity,
                "confidence": float(final_confidence),
                "buy_weight": float(buy_weight),
                "sell_weight": float(sell_weight)
            })
            
            return result
            
        except Exception as e:
            logging.error(f"Error in integrate_components for {ticker}: {e}")
            logging.error(traceback.format_exc())
            return result