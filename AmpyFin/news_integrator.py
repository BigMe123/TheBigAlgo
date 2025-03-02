#!/usr/bin/env python
# News Sentiment Integration Module
# This file serves as the bridge between the news sentiment analyzer and the ranking client

import os
import logging
import time
from datetime import datetime, timedelta
import traceback
import certifi
from pymongo import MongoClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SentimentIntegration")

# Try to import NewsSentimentAnalyzer from your news-only analyzer file
try:
    from news_analyzer import NewsSentimentAnalyzer
except ImportError:
    logger.warning("Could not import NewsSentimentAnalyzer, using stub implementation")
    
    # Stub implementation if real analyzer not available
    class NewsSentimentAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        
        def generate_signal(self, ticker):
            return {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "source": "sentiment",
                "sentiment_score": 0,
                "confidence": 0,
                "action": "HOLD",
                "buy_weight": 0,
                "sell_weight": 0
            }

# Singleton instance for the sentiment analyzer
_analyzer_instance = None

def get_sentiment_analyzer(mongo_url=None):
    """
    Get or create a singleton instance of the sentiment analyzer.
    
    Args:
        mongo_url: MongoDB connection URL
    
    Returns:
        NewsSentimentAnalyzer instance
    """
    global _analyzer_instance
    
    if _analyzer_instance is None:
        try:
            # Load required API keys from environment variables
            alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
            newsapi_key = os.environ.get('NEWSAPI_KEY')
            finnhub_key = os.environ.get('FINNHUB_API_KEY')
            
            # Create analyzer instance (no Twitter parameters)
            _analyzer_instance = NewsSentimentAnalyzer(
                alpha_vantage_key=alpha_vantage_key,
                newsapi_key=newsapi_key,
                finnhub_key=finnhub_key,
                mongo_url=mongo_url
            )
            
            # Start monitoring thread with a 1-hour update interval
            _analyzer_instance.start_monitoring(interval=3600)
            logger.info("Started sentiment monitoring thread")
        except Exception as e:
            logger.error(f"Error creating sentiment analyzer: {e}")
            logger.error(traceback.format_exc())
            _analyzer_instance = NewsSentimentAnalyzer()  # Fallback to stub
    
    return _analyzer_instance

def integrate_with_signal_integrator(ticker, signal_integrator=None, mongo_url=None):
    """
    Function to integrate sentiment analysis with the signal integrator.
    
    Args:
        ticker: Stock ticker symbol
        signal_integrator: Optional signal integrator instance (not used, kept for compatibility)
        mongo_url: MongoDB connection URL
        
    Returns:
        tuple: (buy_weight, sell_weight) from sentiment analysis
    """
    try:
        analyzer = get_sentiment_analyzer(mongo_url)
        signal = analyzer.generate_signal(ticker)
        return signal.get('buy_weight', 0), signal.get('sell_weight', 0)
    except Exception as e:
        logger.error(f"Error in integrate_with_signal_integrator for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return 0, 0

def get_cached_sentiment(ticker, mongo_url=None):
    """
    Get cached sentiment data from MongoDB if available.
    
    Args:
        ticker: Stock ticker symbol
        mongo_url: MongoDB connection URL
        
    Returns:
        dict: Sentiment data or None if not available
    """
    if not mongo_url:
        return None
    
    try:
        client = MongoClient(mongo_url, tlsCAFile=certifi.where())
        db = client.market_sentiment
        cutoff_time = datetime.now() - timedelta(hours=12)
        sentiment = db.sentiment.find_one({
            "ticker": ticker,
            "source": "combined",
            "timestamp": {"$gt": cutoff_time.isoformat()}
        })
        client.close()
        return sentiment
    except Exception as e:
        logger.error(f"Error getting cached sentiment for {ticker}: {e}")
        return None

def get_market_sentiment(mongo_url=None):
    """
    Get overall market sentiment.
    
    Args:
        mongo_url: MongoDB connection URL
        
    Returns:
        dict: Market sentiment data
    """
    if not mongo_url:
        return None
    
    try:
        client = MongoClient(mongo_url, tlsCAFile=certifi.where())
        db = client.market_sentiment
        sentiment = db.sentiment.find_one({"source": "market"})
        client.close()
        return sentiment
    except Exception as e:
        logger.error(f"Error getting market sentiment: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="News Sentiment Integration Module")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker to analyze")
    parser.add_argument("--mongo_url", type=str, help="MongoDB connection URL")
    
    args = parser.parse_args()
    
    print(f"Getting sentiment for {args.ticker}")
    buy_weight, sell_weight = integrate_with_signal_integrator(args.ticker, mongo_url=args.mongo_url)
    
    print(f"Results for {args.ticker}:")
    print(f"  Buy weight: {buy_weight}")
    print(f"  Sell weight: {sell_weight}")
    
    if buy_weight > sell_weight * 1.5 and buy_weight > 200:
        print(f"  Recommendation: BUY (confidence: {min(1.0, buy_weight/1000):.2f})")
    elif sell_weight > buy_weight * 1.5 and sell_weight > 200:
        print(f"  Recommendation: SELL (confidence: {min(1.0, sell_weight/1000):.2f})")
    else:
        print("  Recommendation: HOLD")
        
    cached = get_cached_sentiment(args.ticker, args.mongo_url)
    if cached:
        print("\nCached sentiment data:")
        print(f"  Score: {cached.get('sentiment_score', 0):.2f}")
        print(f"  Confidence: {cached.get('confidence', 0):.2f}")
        print(f"  Label: {cached.get('sentiment_label', 'neutral').upper()}")
        print(f"  Last updated: {cached.get('timestamp', 'unknown')}")
