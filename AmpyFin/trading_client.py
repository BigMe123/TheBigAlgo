from polygon import RESTClient
from config import POLYGON_API_KEY, FINANCIAL_PREP_API_KEY, MONGO_DB_USER, MONGO_DB_PASS, API_KEY, API_SECRET, BASE_URL, mongo_url
import json
import certifi
from urllib.request import urlopen
from zoneinfo import ZoneInfo
from pymongo import MongoClient
import time
from datetime import datetime, timedelta
from helper_files.client_helper import place_order, get_ndaq_tickers, market_status, strategies, get_latest_price, dynamic_period_selector
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from strategies.archived_strategies.trading_strategies_v1 import get_historical_data
import yfinance as yf
import logging
from collections import Counter
from statistics import median, mode
import statistics
import heapq
import requests
from strategies.talib_indicators import *

# Import the signal integrator functions
from deployed_model.signal_integrator import integrate_with_trading_client

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('system.log'),  # Log messages to a file
        logging.StreamHandler()             # Log messages to the console
    ]
)

# Configuration for signal integrator influence
SIGNAL_WEIGHT = 0.5  # Adjust between 0.0 and 1.0 to control influence

def weighted_majority_decision_with_signals(decisions_and_quantities, signal_data=None):  
    """  
    Enhanced weighted majority decision that incorporates signal integrator recommendations.
    Groups 'strong buy' with 'buy' and 'strong sell' with 'sell'.
    Applies weights to quantities based on strategy coefficients.  
    
    Args:
        decisions_and_quantities: List of (decision, quantity, weight) tuples from strategies
        signal_data: Dictionary containing signal integrator recommendations
        
    Returns:
        Tuple of (decision, quantity, buy_weight, sell_weight, hold_weight)
    """
    buy_decisions = ['buy', 'strong buy']  
    sell_decisions = ['sell', 'strong sell']  

    weighted_buy_quantities = []
    weighted_sell_quantities = []
    buy_weight = 0
    sell_weight = 0
    hold_weight = 0
    
    # Process decisions with weights from strategies
    for decision, quantity, weight in decisions_and_quantities:
        if decision.lower() in buy_decisions:
            weighted_buy_quantities.extend([quantity])
            buy_weight += weight
        elif decision.lower() in sell_decisions:
            weighted_sell_quantities.extend([quantity])
            sell_weight += weight
        elif decision.lower() == 'hold':
            hold_weight += weight
    
    # Add signal integrator recommendation if available
    if signal_data and SIGNAL_WEIGHT > 0:
        signal_action = signal_data.get('action', 'HOLD')
        signal_confidence = signal_data.get('confidence', 0)
        
        # Apply the signal's buy/sell weights
        signal_buy_weight = signal_data.get('buy_weight', 0) * SIGNAL_WEIGHT
        signal_sell_weight = signal_data.get('sell_weight', 0) * SIGNAL_WEIGHT
        
        if signal_action in ['STRONG BUY', 'BUY']:
            buy_weight += signal_buy_weight * (1 + signal_confidence)
            # Calculate a reasonable quantity based on the buy weight
            signal_quantity = max(1, int(signal_buy_weight / signal_data.get('price', 100)))
            weighted_buy_quantities.append(signal_quantity)
            logging.info(f"Signal integrator suggests BUY with quantity {signal_quantity}")
        elif signal_action in ['STRONG SELL', 'SELL']:
            sell_weight += signal_sell_weight * (1 + signal_confidence)
            # Calculate a reasonable quantity based on the sell weight
            signal_quantity = max(1, int(signal_sell_weight / signal_data.get('price', 100)))
            weighted_sell_quantities.append(signal_quantity)
            logging.info(f"Signal integrator suggests SELL with quantity {signal_quantity}")
        else:
            hold_weight += SIGNAL_WEIGHT
            logging.info("Signal integrator suggests HOLD")
    
    # Determine the majority decision based on the highest accumulated weight
    if buy_weight > sell_weight and buy_weight > hold_weight:
        return 'buy', median(weighted_buy_quantities) if weighted_buy_quantities else 0, buy_weight, sell_weight, hold_weight
    elif sell_weight > buy_weight and sell_weight > hold_weight:
        return 'sell', median(weighted_sell_quantities) if weighted_sell_quantities else 0, buy_weight, sell_weight, hold_weight
    else:
        return 'hold', 0, buy_weight, sell_weight, hold_weight

def get_data(ticker, mongo_client, period='1y'):
    """
    Get historical data for a ticker.
    First checks cache in MongoDB, then fetches from yfinance if not cached.
    """
    try:
        # Check if data is in MongoDB cache
        db = mongo_client.HistoricalDatabase
        collection = db.HistoricalDatabase
        
        # Look for recent data in cache
        cached_data = collection.find_one({
            "ticker": ticker,
            "period": period,
            "timestamp": {"$gt": datetime.now() - timedelta(hours=12)}
        })
        
        if cached_data and 'data' in cached_data:
            return cached_data['data']
        
        # If not in cache, fetch from yfinance
        ticker_data = yf.Ticker(ticker)
        historical_data = ticker_data.history(period=period)
        
        # Store in MongoDB for future use
        collection.update_one(
            {"ticker": ticker, "period": period},
            {"$set": {
                "data": historical_data.to_dict('records'),
                "timestamp": datetime.now()
            }},
            upsert=True
        )
        
        return historical_data.to_dict('records')
    except Exception as e:
        logging.error(f"Error getting data for {ticker}: {e}")
        return None

def simulate_strategy(strategy, ticker, current_price, historical_data, buying_power, portfolio_qty, portfolio_value):
    """
    Simulates a trading strategy and returns the decision and quantity.
    """
    try:
        result = strategy(ticker, historical_data, current_price, buying_power, portfolio_qty, portfolio_value)
        decision = result.get('decision', 'hold').lower()
        quantity = result.get('quantity', 0)
        return decision, quantity
    except Exception as e:
        logging.error(f"Error simulating {strategy.__name__} for {ticker}: {e}")
        return 'hold', 0

def main():
    """
    Main function to control the workflow based on the market's status.
    """
    ndaq_tickers = []
    early_hour_first_iteration = True
    post_hour_first_iteration = True
    client = RESTClient(api_key=POLYGON_API_KEY)
    trading_client = TradingClient(API_KEY, API_SECRET)
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    mongo_client = MongoClient(mongo_url, tlsCAFile=certifi.where())
    db = mongo_client.trades
    asset_collection = db.assets_quantities
    limits_collection = db.assets_limit
    strategy_to_coefficient = {}

    while True:
        client = RESTClient(api_key=POLYGON_API_KEY)
        trading_client = TradingClient(API_KEY, API_SECRET)
        data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
        status = market_status(client)  # Use the helper function for market status
        db = mongo_client.trades
        asset_collection = db.assets_quantities
        limits_collection = db.assets_limit
        market_db = mongo_client.market_data
        market_collection = market_db.market_status
        indicator_tb = mongo_client.IndicatorsDatabase
        indicator_collection = indicator_tb.Indicators
        
        market_collection.update_one({}, {"$set": {"market_status": status}}, upsert=True)
        
        if status == "open":
            if not ndaq_tickers:
                logging.info("Market is open")
                ndaq_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
                sim_db = mongo_client.trading_simulator
                rank_collection = sim_db.rank
                r_t_c_collection = sim_db.rank_to_coefficient
                for strategy in strategies:
                    
                    rank = rank_collection.find_one({'strategy': strategy.__name__})['rank']
                    coefficient = r_t_c_collection.find_one({'rank': rank})['coefficient']
                    strategy_to_coefficient[strategy.__name__] = coefficient
                    early_hour_first_iteration = False
                    post_hour_first_iteration = True
                    
            account = trading_client.get_account()
            qqq_latest = get_latest_price('QQQ')
            spy_latest = get_latest_price('SPY')
            buy_heap = []
            suggestion_heap = []

            for ticker in ndaq_tickers:
                decisions_and_quantities = []
                try:
                    trading_client = TradingClient(API_KEY, API_SECRET)
                    account = trading_client.get_account()
                    buying_power = float(account.cash)
                    portfolio_value = float(account.portfolio_value)
                    cash_to_portfolio_ratio = buying_power / portfolio_value
                    trades_db = mongo_client.trades
                    portfolio_collection = trades_db.portfolio_values
                    
                    portfolio_collection.update_one({"name": "portfolio_percentage"}, 
                                                   {"$set": {"portfolio_value": (portfolio_value-50491.13)/50491.13}}, 
                                                   upsert=True)
                    portfolio_collection.update_one({"name": "ndaq_percentage"}, 
                                                   {"$set": {"portfolio_value": (qqq_latest-518.58)/518.58}}, 
                                                   upsert=True)
                    portfolio_collection.update_one({"name": "spy_percentage"}, 
                                                   {"$set": {"portfolio_value": (spy_latest-591.95)/591.95}}, 
                                                   upsert=True)
                    
                    current_price = None
                    while current_price is None:
                        try:
                            current_price = get_latest_price(ticker)
                        except:
                            print(f"Error fetching price for {ticker}. Retrying...")
                            time.sleep(10)
                    print(f"Current price of {ticker}: {current_price}")

                    asset_info = asset_collection.find_one({'symbol': ticker})
                    portfolio_qty = asset_info['quantity'] if asset_info else 0.0
                    print(f"Portfolio quantity for {ticker}: {portfolio_qty}")
                    
                    limit_info = limits_collection.find_one({'symbol': ticker})
                    if limit_info:
                        stop_loss_price = limit_info['stop_loss_price']
                        take_profit_price = limit_info['take_profit_price']
                        if current_price <= stop_loss_price or current_price >= take_profit_price:
                            print(f"Executing SELL order for {ticker} due to stop-loss or take-profit condition")
                            quantity = portfolio_qty
                            order = place_order(trading_client, symbol=ticker, side=OrderSide.SELL, quantity=quantity, mongo_client=mongo_client)
                            logging.info(f"Executed SELL order for {ticker}: {order}")
                            continue
                    
                    # Get signal integrator recommendation
                    signal_data = None
                    try:
                        # Get data for the signal integrator
                        period = indicator_collection.find_one({'indicator': 'AmplifySignalIntegrator'})
                        if not period:
                            # Use a default period if not found
                            period = {'ideal_period': '1y'}
                            # Store it for future use
                            indicator_collection.insert_one({'indicator': 'AmplifySignalIntegrator', 'ideal_period': '1y'})
                        
                        historical_data = get_data(ticker, mongo_client, period['ideal_period'])
                        
                        # Get signal integrator recommendations
                        signal_buy_weight, signal_sell_weight = integrate_with_trading_client(ticker, historical_data, mongo_client)
                        
                        # If we got weights, create a signal data structure
                        if signal_buy_weight > 0 or signal_sell_weight > 0:
                            signal_data = {
                                'buy_weight': signal_buy_weight,
                                'sell_weight': signal_sell_weight,
                                'price': current_price,
                                'action': 'BUY' if signal_buy_weight > signal_sell_weight else 'SELL' if signal_sell_weight > signal_buy_weight else 'HOLD',
                                'confidence': abs(signal_buy_weight - signal_sell_weight) / max(signal_buy_weight, signal_sell_weight, 1) if max(signal_buy_weight, signal_sell_weight) > 0 else 0
                            }
                            logging.info(f"Signal integrator for {ticker}: Buy weight={signal_buy_weight}, Sell weight={signal_sell_weight}")
                    except Exception as e:
                        logging.error(f"Error getting signal integrator data for {ticker}: {e}")
                        signal_data = None
                                        
                    for strategy in strategies:
                        historical_data = None
                        while historical_data is None:
                            try:
                                period = indicator_collection.find_one({'indicator': strategy.__name__})
                                historical_data = get_data(ticker, mongo_client, period['ideal_period'])
                            except:
                                print(f"Error fetching data for {ticker}. Retrying...")
                                time.sleep(10)
                        
                        decision, quantity = simulate_strategy(strategy, ticker, current_price, historical_data, buying_power, portfolio_qty, portfolio_value)
                        weight = strategy_to_coefficient[strategy.__name__]
                        decisions_and_quantities.append((decision, quantity, weight))
                    
                    # Use the enhanced weighted majority function that incorporates signal data
                    decision, quantity, buy_weight, sell_weight, hold_weight = weighted_majority_decision_with_signals(
                        decisions_and_quantities, signal_data
                    )
                    
                    print(f"Ticker: {ticker}, Decision: {decision}, Quantity: {quantity}, Weights: Buy: {buy_weight}, Sell: {sell_weight}, Hold: {hold_weight}")
                    
                    print(f"Cash: {account.cash}")

                    if decision == "buy" and float(account.cash) > 15000 and (((quantity + portfolio_qty) * current_price) / portfolio_value) < 0.1:
                        heapq.heappush(buy_heap, (-(buy_weight-(sell_weight + (hold_weight * 0.5))), quantity, ticker))
                    elif (decision == "sell") and portfolio_qty > 0:
                        print(f"Executing SELL order for {ticker}")
                        print(f"Executing quantity of {quantity} for {ticker}")
                        quantity = max(quantity, 1)
                        order = place_order(trading_client, symbol=ticker, side=OrderSide.SELL, quantity=quantity, mongo_client=mongo_client)
                        logging.info(f"Executed SELL order for {ticker}: {order}")
                    elif portfolio_qty == 0.0 and buy_weight > sell_weight and (((quantity + portfolio_qty) * current_price) / portfolio_value) < 0.1 and float(account.cash) > 15000:
                        max_investment = portfolio_value * 0.10
                        buy_quantity = min(int(max_investment // current_price), int(buying_power // current_price))
                        if buy_weight > 2050000:
                            buy_quantity = max(buy_quantity, 2)
                            buy_quantity = buy_quantity // 2
                            print(f"Suggestions for buying for {ticker} with a weight of {buy_weight} and quantity of {buy_quantity}")

                            heapq.heappush(suggestion_heap, (-(buy_weight - sell_weight), buy_quantity, ticker))
                        else:
                            logging.info(f"Holding for {ticker}, no action taken.")
                    else:
                        logging.info(f"Holding for {ticker}, no action taken.")
                    
                except Exception as e:
                    logging.error(f"Error processing {ticker}: {e}")
            trading_client = TradingClient(API_KEY, API_SECRET)
            account = trading_client.get_account()
            while (buy_heap or suggestion_heap) and float(account.cash) > 15000:
                try:
                    trading_client = TradingClient(API_KEY, API_SECRET)
                    account = trading_client.get_account()
                    print(f"Cash: {account.cash}")
                    if buy_heap and float(account.cash) > 15000:
                        
                        _, quantity, ticker = heapq.heappop(buy_heap)
                        print(f"Executing BUY order for {ticker} of quantity {quantity}")
                        
                        order = place_order(trading_client, symbol=ticker, side=OrderSide.BUY, quantity=quantity, mongo_client=mongo_client)
                        logging.info(f"Executed BUY order for {ticker}: {order}")
                        
                    elif suggestion_heap and float(account.cash) > 15000:
                        
                        _, quantity, ticker = heapq.heappop(suggestion_heap)
                        print(f"Executing BUY order for {ticker} of quantity {quantity}")
                        
                        order = place_order(trading_client, symbol=ticker, side=OrderSide.BUY, quantity=quantity, mongo_client=mongo_client)
                        logging.info(f"Executed BUY order for {ticker}: {order}")
                        
                    time.sleep(5)
                    """
                    This is here so order will propage through and we will have an accurate cash balance recorded
                    """
                except Exception as e:
                    print(f"Error occurred while executing buy order: {e}. Continuing...")
                    break
            
            print("Sleeping for 60 seconds...")
            time.sleep(60)

        elif status == "early_hours":
            if early_hour_first_iteration:
                ndaq_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
                sim_db = mongo_client.trading_simulator
                rank_collection = sim_db.rank
                r_t_c_collection = sim_db.rank_to_coefficient
                for strategy in strategies:
                    rank = rank_collection.find_one({'strategy': strategy.__name__})['rank']
                    coefficient = r_t_c_collection.find_one({'rank': rank})['coefficient']
                    strategy_to_coefficient[strategy.__name__] = coefficient
                    early_hour_first_iteration = False
                    post_hour_first_iteration = True
                logging.info("Market is in early hours. Waiting for 60 seconds.")
            time.sleep(30)

        elif status == "closed":
            if post_hour_first_iteration:
                early_hour_first_iteration = True
                post_hour_first_iteration = False
                logging.info("Market is closed. Performing post-market operations.")
            time.sleep(30)
        else:
            logging.error("An error occurred while checking market status.")
            time.sleep(60)

if __name__ == "__main__":
    main()