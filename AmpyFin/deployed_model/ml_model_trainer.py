# ml_model_trainer.py
import pandas as pd
import numpy as np
import time
import logging
import os
import schedule
from datetime import datetime, timedelta
from pymongo import MongoClient
import yfinance as yf
from ml_trading_predictor import MLTradingPredictor
import certifi

# Import directly from local_config.py
from local_config import mongo_url

def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with file and console handlers."""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers if they don't exist
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def create_directory_if_not_exists(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")
    return directory_path

# Set up logging
logger = setup_logger('ml_trainer', 'ml_training.log')

class MLModelTrainer:
    def __init__(self, mongo_client, predictor=None):
        """Initialize the ML model trainer."""
        self.mongo_client = mongo_client
        self.predictor = predictor if predictor else MLTradingPredictor()
        self.data_dir = "./ml_training_data"
        create_directory_if_not_exists(self.data_dir)
        
    def fetch_intraday_data(self, tickers, interval="5m", period="5d"):
        """Fetch intraday data for a list of tickers."""
        all_data = []
        
        for ticker in tickers:
            try:
                logger.info(f"Fetching intraday data for {ticker}")
                stock = yf.Ticker(ticker)
                df = stock.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue
                    
                # Add ticker symbol and reset index
                df["Symbol"] = ticker
                df.reset_index(inplace=True)
                
                # Add label column (1 if next close > current close)
                df = df.sort_values('Datetime').reset_index(drop=True)
                df['Label'] = df['Close'].shift(-1) > df['Close']
                df['Label'] = df['Label'].astype(int)
                
                # Rename columns to match expected format
                df.rename(columns={"Datetime": "Timestamp"}, inplace=True)
                
                # Save to csv for backup
                timestamp = datetime.now().strftime("%Y%m%d")
                df.to_csv(f"{self.data_dir}/{ticker}_{timestamp}.csv", index=False)
                
                all_data.append(df)
                logger.info(f"Successfully processed {ticker} data with {len(df)} rows")
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        
        return all_data
    
    def load_historical_data_from_mongodb(self):
        """Load previously stored data from MongoDB for training."""
        try:
            db = self.mongo_client.HistoricalDatabase
            collection = db.HistoricalDatabase
            
            data_frames = []
            
            # Find all unique tickers
            tickers = collection.distinct("Symbol")
            
            for ticker in tickers:
                logger.info(f"Loading historical data for {ticker} from MongoDB")
                
                # Get data for this ticker
                cursor = collection.find({"Symbol": ticker})
                data = list(cursor)
                
                if not data:
                    logger.warning(f"No data found for {ticker} in MongoDB")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Drop MongoDB ID
                if '_id' in df.columns:
                    df = df.drop('_id', axis=1)
                
                # Add label column if not present
                if 'Label' not in df.columns:
                    df = df.sort_values('Timestamp').reset_index(drop=True)
                    df['Label'] = df['Close'].shift(-1) > df['Close']
                    df['Label'] = df['Label'].astype(int)
                    
                # Drop rows with NaN in Label column
                df = df.dropna(subset=['Label'])
                
                data_frames.append(df)
                logger.info(f"Loaded {len(df)} rows for {ticker}")
                
            return data_frames
            
        except Exception as e:
            logger.error(f"Error loading historical data from MongoDB: {e}")
            return []
    
    def get_ndaq_tickers(self, limit=20):
        """Get NASDAQ tickers for training from MongoDB."""
        try:
            db = self.mongo_client.trading_simulator
            holdings_collection = db.algorithm_holdings
            
            # Find tickers with highest trading activity
            tickers = []
            
            for strategy_doc in holdings_collection.find({}):
                for ticker in strategy_doc.get("holdings", {}):
                    tickers.append(ticker)
            
            # Count occurrences of each ticker across strategies
            ticker_counts = {}
            for ticker in tickers:
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            
            # Sort by count (descending)
            sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N tickers
            top_tickers = [ticker for ticker, count in sorted_tickers[:limit]]
            
            if not top_tickers:
                # Fallback to some popular tickers if none found
                top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX"]
            
            return top_tickers
            
        except Exception as e:
            logger.error(f"Error getting NASDAQ tickers: {e}")
            # Return some default tickers
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX"]
    
    def train_and_update_model(self):
        """Train/update the ML model with both historical and new data."""
        try:
            logger.info("Starting model training process")
            
            # Get top tickers for training
            tickers = self.get_ndaq_tickers(limit=20)
            logger.info(f"Selected {len(tickers)} tickers for training: {tickers}")
            
            # Fetch new intraday data
            new_data = self.fetch_intraday_data(tickers)
            logger.info(f"Fetched {len(new_data)} new data frames")
            
            # Load historical data from MongoDB
            historical_data = self.load_historical_data_from_mongodb()
            logger.info(f"Loaded {len(historical_data)} historical data frames")
            
            # Combine data
            all_data = new_data + historical_data
            
            if not all_data:
                logger.warning("No data available for training")
                return False
            
            # Train model
            success = self.predictor.train_model(all_data)
            
            if success:
                logger.info("Model training completed successfully")
            else:
                logger.error("Model training failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in train_and_update_model: {e}")
            return False
    
    def schedule_training(self, time_str="16:30"):
        """Schedule regular model training after market close."""
        logger.info(f"Scheduling daily model training at {time_str}")
        schedule.every().monday.at(time_str).do(self.train_and_update_model)
        schedule.every().tuesday.at(time_str).do(self.train_and_update_model)
        schedule.every().wednesday.at(time_str).do(self.train_and_update_model)
        schedule.every().thursday.at(time_str).do(self.train_and_update_model)
        schedule.every().friday.at(time_str).do(self.train_and_update_model)
        
    def run_scheduler(self):
        """Run the scheduler loop."""
        logger.info("Starting scheduler for ML model training")
        while True:
            schedule.run_pending()
            time.sleep(60)

# Run as standalone module
if __name__ == "__main__":
    try:
        logger.info("Starting ML model trainer")
        
        # Connect to MongoDB
        ca = certifi.where()
        mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
        
        # Initialize predictor and trainer
        predictor = MLTradingPredictor()
        trainer = MLModelTrainer(mongo_client, predictor)
        
        # Train model immediately
        logger.info("Running initial model training")
        trainer.train_and_update_model()
        
        # Schedule future training
        trainer.schedule_training()
        trainer.run_scheduler()
        
    except Exception as e:
        logger.error(f"Error in ML model trainer: {e}")