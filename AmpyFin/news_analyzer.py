#!/usr/bin/env python
# News Sentiment Analyzer (News Only) -- Hardcoded API Keys

import os
import sys
import logging
import urllib.parse
import traceback
import time
import threading
from datetime import datetime, timedelta, timezone
import re
import json
from collections import Counter, defaultdict

import certifi
import requests
from pymongo import MongoClient
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import yfinance as yf

# Download required NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------------
# Hardcoded API Keys and Credentials
# ---------------------------------------------------------------------
ALPHA_VANTAGE_API_KEY = "1A4W4OWOQ1KR2J5A"
NEWSAPI_KEY = "027e167533f7488bb9935e9ab1874e72"
FINNHUB_API_KEY = "cv1vs2hr01qngf0b5390cv1vs2hr01qngf0b539g"

# If your Atlas username is actually "shadowguy311@gmail.com", do this:
MONGO_DB_USER = "shadowguy311@gmail.com"  # EXACT username from Atlas
MONGO_DB_PASS = "tanki123"
encoded_user = urllib.parse.quote_plus(MONGO_DB_USER)  # e.g., shadowguy311%40gmail.com
encoded_pass = urllib.parse.quote_plus(MONGO_DB_PASS)
mongo_url = (
    f"mongodb+srv://{encoded_user}:{encoded_pass}@cluster0.yjmg8.mongodb.net/"
    f"?retryWrites=true&w=majority&appName=Cluster0"
)

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("NewsSentiment")


class NewsSentimentAnalyzer:
    """
    News Sentiment Analyzer that uses:
      - Alpha Vantage (with Yahoo Finance fallback)
      - NewsAPI.org
      - Finnhub
    to gather news articles and compute sentiment.
    """
    def __init__(self, alpha_vantage_key, newsapi_key, finnhub_key, mongo_url=None):
        self.alpha_vantage_key = alpha_vantage_key
        self.newsapi_key = newsapi_key
        self.finnhub_key = finnhub_key
        self.mongo_url = mongo_url
        self.mongo_client = None
        self.vader = SentimentIntensityAnalyzer()
        self._connect_mongodb()
        self.sentiment_cache = {}
        self.stop_event = threading.Event()
        self.monitor_thread = None
        logger.info("News Sentiment Analyzer initialized (news sources only)")

    def _connect_mongodb(self):
        if not self.mongo_url:
            logger.info("No MongoDB URL provided; using in-memory storage")
            return False
        try:
            ca = certifi.where()
            self.mongo_client = MongoClient(self.mongo_url, tlsCAFile=ca)
            db = self.mongo_client.market_sentiment
            if 'sentiment' not in db.list_collection_names():
                db.create_collection('sentiment')
            db.sentiment.find_one()  # Test connection
            logger.info("Connected to MongoDB successfully")
            return True
        except Exception as e:
            logger.warning(f"Error connecting to MongoDB: {e}")
            self.mongo_client = None
            return False

    def _parse_time(self, time_str):
        """
        Parse a time string that may be in ISO format or compact (e.g. '20250301T160048').
        Returns a timezone-aware datetime in UTC or None on failure.
        """
        if not time_str:
            return None
        # If no dashes/colons, assume format YYYYMMDDTHHMMSS
        if '-' not in time_str or ':' not in time_str:
            if len(time_str) == 15 and 'T' in time_str:
                time_str = f"{time_str[0:4]}-{time_str[4:6]}-{time_str[6:8]}T{time_str[9:11]}:{time_str[11:13]}:{time_str[13:15]}"
                time_str += "+00:00"
        else:
            time_str = time_str.replace('Z', '+00:00')
        try:
            dt = datetime.fromisoformat(time_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception as e:
            logger.error(f"Error parsing time string '{time_str}': {e}")
            return None

    def get_news_sentiment(self, ticker, days_back=7):
        """
        Get news sentiment using Alpha Vantage (with Yahoo Finance fallback).
        """
        cache_key = f"news_{ticker}_{days_back}"
        if cache_key in self.sentiment_cache:
            entry = self.sentiment_cache[cache_key]
            if datetime.now(timezone.utc) - entry['timestamp'] < timedelta(hours=12):
                return entry['data']
        articles = []
        # Alpha Vantage News
        if self.alpha_vantage_key:
            av_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.alpha_vantage_key}"
            response = requests.get(av_url)
            if response.status_code == 200:
                data = response.json()
                if 'feed' in data:
                    for item in data['feed']:
                        pub_time = self._parse_time(item.get('time_published', ''))
                        if pub_time and (datetime.now(timezone.utc) - pub_time <= timedelta(days=days_back)):
                            articles.append({
                                'title': item.get('title', ''),
                                'summary': item.get('summary', ''),
                                'source': item.get('source', ''),
                                'url': item.get('url', ''),
                                'time': pub_time,
                                'sentiment': item.get('overall_sentiment_score')
                            })
        # Yahoo Finance fallback if < 10 articles
        if len(articles) < 10:
            ticker_data = yf.Ticker(ticker)
            yahoo_news = ticker_data.news
            for item in yahoo_news:
                try:
                    pub_time = datetime.fromtimestamp(item.get('providerPublishTime', 0), tz=timezone.utc)
                except Exception:
                    continue
                if pub_time and (datetime.now(timezone.utc) - pub_time <= timedelta(days=days_back)):
                    articles.append({
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'source': item.get('publisher', 'Yahoo Finance'),
                        'url': item.get('link', ''),
                        'time': pub_time,
                        'sentiment': None
                    })
        # Log how many found
        if articles:
            logger.info(f"Found {len(articles)} articles for {ticker} via Alpha Vantage/ Yahoo Finance")
        else:
            logger.info(f"No articles found for {ticker} via Alpha Vantage/ Yahoo Finance")

        # Compute VADER for articles missing sentiment
        for article in articles:
            if article['sentiment'] is None:
                text = f"{article['title']}. {article['summary']}"
                article['sentiment'] = self.vader.polarity_scores(text)['compound']

        # Aggregate
        scores = [a['sentiment'] for a in articles]
        avg_sent = sum(scores) / len(scores) if scores else 0
        magnitude = sum(abs(s) for s in scores) / len(scores) if scores else 0
        result = {
            "ticker": ticker,
            "source": "news",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sentiment_score": float(avg_sent),
            "sentiment_magnitude": float(magnitude),
            "article_count": len(articles)
        }
        # Cache
        self.sentiment_cache[cache_key] = {"timestamp": datetime.now(timezone.utc), "data": result}

        # Store in MongoDB if connected
        if self.mongo_client:
            try:
                db = self.mongo_client.market_sentiment
                db.sentiment.update_one({"ticker": ticker, "source": "news"}, {"$set": result}, upsert=True)
            except Exception as e:
                logger.error(f"Error storing news sentiment in MongoDB: {e}")

        return result

    def get_newsapi_sentiment(self, ticker, days_back=7):
        """
        Get news sentiment from NewsAPI.org.
        """
        if not self.newsapi_key:
            return None
        cache_key = f"newsapi_{ticker}_{days_back}"
        if cache_key in self.sentiment_cache:
            entry = self.sentiment_cache[cache_key]
            if datetime.now(timezone.utc) - entry['timestamp'] < timedelta(hours=12):
                return entry['data']
        try:
            url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&apiKey={self.newsapi_key}"
            response = requests.get(url)
            articles = []
            if response.status_code == 200:
                data = response.json()
                for item in data.get("articles", []):
                    pub_time = self._parse_time(item.get("publishedAt", ""))
                    if pub_time and (datetime.now(timezone.utc) - pub_time <= timedelta(days=days_back)):
                        articles.append({
                            "title": item.get("title", ""),
                            "summary": item.get("description", ""),
                            "source": item.get("source", {}).get("name", ""),
                            "url": item.get("url", ""),
                            "time": pub_time,
                            "sentiment": None
                        })
            if articles:
                logger.info(f"Found {len(articles)} articles for {ticker} via NewsAPI")
            else:
                logger.info(f"No articles found for {ticker} via NewsAPI")
            for a in articles:
                text = f"{a['title']}. {a['summary']}"
                a["sentiment"] = self.vader.polarity_scores(text)["compound"]
            scores = [a["sentiment"] for a in articles]
            avg_sent = sum(scores) / len(scores) if scores else 0
            magnitude = sum(abs(s) for s in scores) / len(scores) if scores else 0
            result = {
                "ticker": ticker,
                "source": "newsapi",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sentiment_score": float(avg_sent),
                "sentiment_magnitude": float(magnitude),
                "article_count": len(articles)
            }
            self.sentiment_cache[cache_key] = {"timestamp": datetime.now(timezone.utc), "data": result}
            return result
        except Exception as e:
            logger.error(f"Error in get_newsapi_sentiment for {ticker}: {e}")
            return {
                "ticker": ticker,
                "source": "newsapi",
                "sentiment_score": 0,
                "sentiment_magnitude": 0,
                "article_count": 0
            }

    def get_finnhub_sentiment(self, ticker, days_back=7):
        """
        Get news sentiment from Finnhub.
        """
        if not self.finnhub_key:
            return None
        cache_key = f"finnhub_{ticker}_{days_back}"
        if cache_key in self.sentiment_cache:
            entry = self.sentiment_cache[cache_key]
            if datetime.now(timezone.utc) - entry['timestamp'] < timedelta(hours=12):
                return entry['data']
        try:
            from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
            to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={self.finnhub_key}"
            response = requests.get(url)
            articles = []
            if response.status_code == 200:
                articles = response.json()
            if articles:
                logger.info(f"Found {len(articles)} articles for {ticker} via Finnhub")
            else:
                logger.info(f"No articles found for {ticker} via Finnhub")
            for a in articles:
                text = f"{a.get('headline', '')}. {a.get('summary', '')}"
                a["sentiment"] = self.vader.polarity_scores(text)["compound"]
            scores = [a["sentiment"] for a in articles]
            avg_sent = sum(scores) / len(scores) if scores else 0
            magnitude = sum(abs(s) for s in scores) / len(scores) if scores else 0
            result = {
                "ticker": ticker,
                "source": "finnhub",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sentiment_score": float(avg_sent),
                "sentiment_magnitude": float(magnitude),
                "article_count": len(articles)
            }
            self.sentiment_cache[cache_key] = {"timestamp": datetime.now(timezone.utc), "data": result}
            return result
        except Exception as e:
            logger.error(f"Error in get_finnhub_sentiment for {ticker}: {e}")
            return {
                "ticker": ticker,
                "source": "finnhub",
                "sentiment_score": 0,
                "sentiment_magnitude": 0,
                "article_count": 0
            }

    def get_combined_sentiment(self, ticker):
        """
        Combine sentiment from available news sources: 
          - Alpha Vantage / Yahoo Finance
          - NewsAPI.org
          - Finnhub
        Prints how many articles & sentiment from each source for debugging.
        """
        sources = []
        news = self.get_news_sentiment(ticker)
        if news:
            sources.append(("news", news))
        newsapi = self.get_newsapi_sentiment(ticker)
        if newsapi:
            sources.append(("newsapi", newsapi))
        finnhub = self.get_finnhub_sentiment(ticker)
        if finnhub:
            sources.append(("finnhub", finnhub))

        # Print out the source breakdown
        for name, data in sources:
            logger.info(
                f"[{ticker}] Source={name}, "
                f"Articles={data.get('article_count', 0)}, "
                f"Score={data.get('sentiment_score', 0)}, "
                f"Magnitude={data.get('sentiment_magnitude', 0)}"
            )

        result = {
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sentiment_score": 0,
            "sentiment_magnitude": 0,
            "source_weights": {},
        }
        if not sources:
            logger.warning(f"No news sentiment data available for {ticker}")
            return result

        total_articles = sum(s[1].get("article_count", 0) for s in sources)
        if total_articles == 0:
            # If no articles at all
            weights = {s[0]: 1.0 / len(sources) for s in sources}
        else:
            weights = {s[0]: s[1].get("article_count", 0) / total_articles for s in sources}

        weighted_sent = sum(s[1].get("sentiment_score", 0) * weights[s[0]] for s in sources)
        weighted_mag = sum(s[1].get("sentiment_magnitude", 0) * weights[s[0]] for s in sources)

        result.update({
            "sentiment_score": float(weighted_sent),
            "sentiment_magnitude": float(weighted_mag),
            "source_weights": weights,
        })

        # If connected, store combined result
        if self.mongo_client:
            try:
                db = self.mongo_client.market_sentiment
                db.sentiment.update_one({"ticker": ticker, "source": "combined"}, {"$set": result}, upsert=True)
            except Exception as e:
                logger.error(f"Error storing combined sentiment in MongoDB: {e}")

        return result

    def get_market_sentiment(self, tickers=None):
        """
        Get overall market sentiment for a list of tickers.
        """
        if not tickers:
            tickers = ["SPY", "QQQ", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
        results = []
        for t in tickers:
            res = self.get_combined_sentiment(t)
            results.append(res)
        scores = [r["sentiment_score"] for r in results]
        avg_market = sum(scores) / len(scores) if scores else 0
        return {"market_sentiment": avg_market, "details": results}

    def generate_signal(self, ticker):
        """
        Generate and print a trading signal based on the combined sentiment.
        """
        combined = self.get_combined_sentiment(ticker)
        print(f"Sentiment for {ticker}: "
              f"Score={combined.get('sentiment_score'):.4f}, "
              f"Magnitude={combined.get('sentiment_magnitude'):.4f}, "
              f"Articles Weights={combined.get('source_weights')}")
        return combined

    def start_monitoring(self, interval=3600, tickers=None):
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitor thread already running")
            return False
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval, tickers))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info(f"Started monitoring thread (interval: {interval}s)")
        return True

    def stop_monitoring(self):
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            logger.warning("Monitor thread not running")
            return False
        self.stop_event.set()
        self.monitor_thread.join(timeout=5)
        logger.info("Stopped monitoring thread")
        return True

    def _monitoring_loop(self, interval, fixed_tickers=None):
        while not self.stop_event.is_set():
            try:
                tickers = fixed_tickers if fixed_tickers else ["SPY", "QQQ", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
                logger.info(f"Monitoring sentiment for {len(tickers)} tickers")
                for t in tickers:
                    if self.stop_event.is_set():
                        break
                    self.get_combined_sentiment(t)
                    time.sleep(2)
                self.stop_event.wait(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                logger.error(traceback.format_exc())
                self.stop_event.wait(60)
        logger.info("Monitoring loop stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="News Sentiment Analyzer (News Only)")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers")
    parser.add_argument("--interval", type=int, default=3600, help="Update interval in seconds")
    parser.add_argument("--mongo_url", type=str, help="MongoDB connection URL")
    parser.add_argument("--alpha_vantage_key", type=str, help="Alpha Vantage API key")
    parser.add_argument("--newsapi_key", type=str, help="NewsAPI.org API key")
    parser.add_argument("--finnhub_key", type=str, help="Finnhub API key")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None

    # Instantiate the analyzer using hardcoded keys (or command-line arguments if provided)
    analyzer = NewsSentimentAnalyzer(
        alpha_vantage_key=args.alpha_vantage_key or ALPHA_VANTAGE_API_KEY,
        newsapi_key=args.newsapi_key or NEWSAPI_KEY,
        finnhub_key=args.finnhub_key or FINNHUB_API_KEY,
        mongo_url=args.mongo_url or mongo_url
    )
    
    print("=" * 80)
    print(" Multi-Source News Sentiment Analyzer (News Only) ".center(80, "#"))
    print("=" * 80)
    
    # Print out the generated signal for each ticker if provided
    if tickers:
        for t in tickers:
            analyzer.generate_signal(t)
    
    try:
        if analyzer.start_monitoring(interval=args.interval, tickers=tickers):
            print(f"Monitoring started (interval: {args.interval}s). Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
        else:
            print("Failed to start monitoring.")
    except KeyboardInterrupt:
        print("\nStopping...")
        analyzer.stop_monitoring()
        print("Analyzer stopped.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
