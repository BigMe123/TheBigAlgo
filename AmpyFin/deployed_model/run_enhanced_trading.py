# run_enhanced_trading.py
import os
import sys
import pandas as pd
import numpy as np
import logging
import json
import time
import traceback
from datetime import datetime, timedelta
import argparse
import importlib.util

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EnhancedTrading")

# Import check function for component availability
def check_component_availability():
    """Check availability of all enhanced trading components."""
    availability = {
        "feature_engineering": importlib.util.find_spec("advanced_feature_engineering") is not None,
        "ensemble_model": importlib.util.find_spec("ensemble_model") is not None,
        "rl_weight": importlib.util.find_spec("rl_weight_generator") is not None,
        "regime_detector": importlib.util.find_spec("market_regime_detector") is not None,
        "hft_analyzer": importlib.util.find_spec("hft_analyzer") is not None,
        "portfolio_optimizer": importlib.util.find_spec("portfolio_optimizer") is not None,
        "trading_system": importlib.util.find_spec("enhanced_trading_system") is not None
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

# Conditionally import components
try:
    from advanced_feature_engineering import AdvancedFeatureEngineering
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    logger.warning("Advanced Feature Engineering module not available")

try:
    from ensemble_model import EnsembleModelManager
    ENSEMBLE_MODEL_AVAILABLE = True
except ImportError:
    ENSEMBLE_MODEL_AVAILABLE = False
    logger.warning("Ensemble Model module not available")

try:
    from rl_weight_generator import RLWeightGenerator
    RL_WEIGHT_AVAILABLE = True
except ImportError:
    RL_WEIGHT_AVAILABLE = False
    logger.warning("RL Weight Generator module not available")

try:
    from market_regime_detector import MarketRegimeDetector
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False
    logger.warning("Market Regime Detector module not available")

try:
    from hft_analyzer import HFTAnalyzer
    HFT_ANALYZER_AVAILABLE = True
except ImportError:
    HFT_ANALYZER_AVAILABLE = False
    logger.warning("HFT Analyzer module not available")

try:
    from portfolio_optimizer import PortfolioOptimizer
    PORTFOLIO_OPTIMIZER_AVAILABLE = True
except ImportError:
    PORTFOLIO_OPTIMIZER_AVAILABLE = False
    logger.warning("Portfolio Optimizer module not available")

try:
    from enhanced_trading_system import EnhancedTradingSystem
    TRADING_SYSTEM_AVAILABLE = True
except ImportError:
    TRADING_SYSTEM_AVAILABLE = False
    logger.warning("Enhanced Trading System module not available")

# Import yfinance for data fetching
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available, install with: pip install yfinance")

# Default tickers for demonstration
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "NVDA", "TSLA", "ADBE", "NFLX", "PYPL",
    "INTC", "CSCO", "AMD", "QCOM", "CRM"
]

def fetch_data(tickers, period="2y", interval="1d"):
    """Fetch historical data from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance not available")
        return []
    
    data_frames = []
    
    for ticker in tickers:
        try:
            logger.info(f"Fetching {ticker} data")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                continue
                
            # Add ticker symbol and reset index
            df["Symbol"] = ticker
            df.reset_index(inplace=True)
            
            # Add label (1 if next close > current close)
            df = df.sort_values('Date').reset_index(drop=True)
            df['Label'] = df['Close'].shift(-1) > df['Close']
            df['Label'] = df['Label'].astype(int)
            
            # Rename Date to Timestamp for consistent column naming
            df.rename(columns={"Date": "Timestamp"}, inplace=True)
            
            data_frames.append(df)
            logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching {ticker} data: {e}")
    
    return data_frames

def run_integrated_system(args):
    """Run the integrated trading system."""
    # Check if the system is available
    if not TRADING_SYSTEM_AVAILABLE:
        logger.error("Enhanced Trading System not available")
        return
    
    # Parse tickers if provided
    tickers = None
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    
    # Initialize the trading system
    system = EnhancedTradingSystem(config_path=args.config)
    
    # Run requested actions
    if args.train:
        logger.info("Training models...")
        result = system.train_models(tickers=tickers, force=True)
        logger.info(f"Training {'successful' if result else 'failed'}")
    
    if args.predict:
        logger.info("Running predictions...")
        result = system.run_predictions(tickers=tickers)
        
        if result["success"]:
            logger.info("Predictions successful")
            
            # Save predictions to file
            try:
                prediction_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(prediction_file, 'w') as f:
                    json.dump(result, f, indent=4, default=str)
                logger.info(f"Predictions saved to {prediction_file}")
            except Exception as e:
                logger.error(f"Error saving predictions: {e}")
                
            # Display summary
            print("\n" + "=" * 80)
            print("PREDICTION SUMMARY")
            print("=" * 80)
            
            print("\nStock Predictions:")
            print("-" * 80)
            print(f"{'Symbol':<6} {'Action':<10} {'Probability':<12} {'Confidence':<12} {'Price':<10}")
            print("-" * 80)
            
            for ticker, prediction in result["predictions"].items():
                print(f"{ticker:<6} {prediction['action']:<10} {prediction['probability']:.2f} ({prediction['probability']*100:.1f}%) "
                      f"{prediction['confidence']:.2f} ({prediction['confidence']*100:.1f}%) ${prediction['price']:<9.2f}")
            
            # Display portfolio
            if "portfolio" in result and result["portfolio"]:
                print("\nRecommended Portfolio Allocation (Investment: $10,000):")
                print("-" * 80)
                
                portfolio = result["portfolio"]
                metrics = portfolio.pop("portfolio_metrics", None)
                
                print(f"{'Rank':<4} {'Symbol':<6} {'Weight':<10} {'Amount ($)':<12} {'ML Adjustment':<12}")
                print("-" * 80)
                
                # Sort by weight
                sorted_portfolio = sorted(portfolio.items(), key=lambda x: x[1].get('weight', 0), reverse=True)
                
                for i, (ticker, data) in enumerate(sorted_portfolio, 1):
                    if isinstance(data, dict) and 'weight' in data:
                        weight = data.get('weight', 0)
                        amount = data.get('amount', 0)
                        ml_adj = data.get('ml_adjustment', 0)
                        
                        print(f"{i:<4} {ticker:<6} {weight:.2%} ${amount:<11.2f} {ml_adj:<12.2f}")
                
                # Display portfolio metrics
                if metrics:
                    print("\nPortfolio Metrics:")
                    print("-" * 80)
                    
                    if "expected_return" in metrics:
                        print(f"Expected Return: {metrics['expected_return']:.2%}")
                    if "annual_volatility" in metrics:
                        print(f"Volatility: {metrics['annual_volatility']:.2%}")
                    if "sharpe_ratio" in metrics:
                        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    if "max_drawdown" in metrics:
                        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
            
            # Display market regime
            if "market_regime" in result and result["market_regime"]:
                regime = result["market_regime"]
                print("\nCurrent Market Regime:")
                print("-" * 80)
                print(f"Regime: {regime.get('description', 'Unknown')}")
                print(f"ID: {regime.get('id', 'Unknown')}")
        else:
            logger.error(f"Predictions failed: {result.get('message', 'Unknown error')}")
    
    if args.backtest:
        logger.info("Running backtesting...")
        result = system.run_backtesting(tickers=tickers)
        
        if result["success"]:
            logger.info("Backtesting successful")
            
            # Display performance
            print("\n" + "=" * 80)
            print("BACKTESTING RESULTS")
            print("=" * 80)
            
            print("\nPerformance Summary:")
            print("-" * 80)
            
            for metric, value in result["performance"].items():
                if "return" in metric or "drawdown" in metric:
                    print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
                else:
                    print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
        else:
            logger.error(f"Backtesting failed: {result.get('message', 'Unknown error')}")
    
    # If no action specified, start the scheduler
    if not (args.train or args.predict or args.backtest):
        logger.info("Starting trading system scheduler...")
        system.start_scheduler()
        
        try:
            print("\n" + "=" * 80)
            print("ENHANCED TRADING SYSTEM RUNNING")
            print("=" * 80)
            print("\nPress Ctrl+C to stop the system")
            
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping trading system scheduler...")
            system.stop_scheduler()
            logger.info("Scheduler stopped")

def run_standalone_components(args):
    """Run standalone components when integrated system is not available."""
    # Parse tickers
    tickers = args.tickers.split(",") if args.tickers else DEFAULT_TICKERS
    
    # Fetch data
    period = "2y" if args.train else "3mo"
    data_frames = fetch_data(tickers, period=period)
    
    if not data_frames:
        logger.error("Failed to fetch data")
        return
    
    # Add features if feature engineering is available
    if FEATURE_ENGINEERING_AVAILABLE:
        logger.info("Adding advanced features")
        fe = AdvancedFeatureEngineering()
        enhanced_frames = []
        
        for df in data_frames:
            enhanced_df = fe.add_all_features(df)
            enhanced_frames.append(enhanced_df)
    else:
        enhanced_frames = data_frames
    
    # Detect market regime if available
    current_regime = 0
    regime_description = "Unknown"
    
    if REGIME_DETECTOR_AVAILABLE:
        try:
            logger.info("Detecting market regime")
            regime_detector = MarketRegimeDetector(n_regimes=3)
            
            # Train the detector if not already trained
            if not os.path.exists("./regime_models/metadata.joblib"):
                regime_detector.fit(enhanced_frames)
            else:
                regime_detector.load_models()
            
            # Get current regime from first dataframe (ideally SPY)
            current_regime = regime_detector.get_current_regime(enhanced_frames[0])
            regime_description = regime_detector.get_regime_description(current_regime)
            
            logger.info(f"Current market regime: {regime_description} (ID: {current_regime})")
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
    
    # Train/run ensemble model if available
    if ENSEMBLE_MODEL_AVAILABLE and args.train:
        try:
            logger.info("Training ensemble model")
            ensemble = EnsembleModelManager(model_dir="./ensemble_models")
            result = ensemble.train_ensemble(enhanced_frames)
            
            if result["success"]:
                logger.info("Ensemble model training successful")
            else:
                logger.error(f"Ensemble model training failed: {result.get('message', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
    
    # Train RL weight generator if available
    if RL_WEIGHT_AVAILABLE and args.train:
        try:
            logger.info("Training RL weight generator")
            rl_generator = RLWeightGenerator(model_dir="./rl_weights")
            result = rl_generator.train(enhanced_frames)
            
            if result["success"]:
                logger.info("RL weight generator training successful")
            else:
                logger.error(f"RL weight generator training failed: {result.get('message', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error training RL weight generator: {e}")
    
    # Run predictions if requested
    if args.predict:
        try:
            logger.info("Running predictions")
            
            # Initialize prediction results
            predictions = {}
            
            # Use ensemble model if available
            if ENSEMBLE_MODEL_AVAILABLE:
                ensemble = EnsembleModelManager(model_dir="./ensemble_models")
                ensemble.load_models()
            else:
                ensemble = None
            
            # Use RL weight generator if available
            if RL_WEIGHT_AVAILABLE:
                rl_generator = RLWeightGenerator(model_dir="./rl_weights")
                rl_generator.load_models()
            else:
                rl_generator = None
            
            # Use HFT analyzer if available
            if HFT_ANALYZER_AVAILABLE:
                hft_analyzer = HFTAnalyzer()
            else:
                hft_analyzer = None
            
            # Process each ticker
            for df in enhanced_frames:
                ticker = df['Symbol'].iloc[0]
                latest_data = df.tail(1)
                
                # Initialize prediction
                prediction = {
                    "ticker": ticker,
                    "price": float(df['Close'].iloc[-1]),
                    "date": str(df['Timestamp'].iloc[-1]),
                    "signals": {},
                    "probability": 0.5,  # Default neutral
                    "confidence": 0,
                    "action": "HOLD",
                    "weights": {
                        "buy": 0,
                        "sell": 0
                    }
                }
                
                # Get ensemble predictions
                if ensemble:
                    try:
                        prob, confidence = ensemble.predict_with_ensemble(latest_data)
                        prediction["ensemble"] = {
                            "probability": float(prob),
                            "confidence": float(confidence),
                            "signal": 1 if prob > 0.5 else -1 if prob < 0.5 else 0
                        }
                        prediction["signals"]["ensemble"] = prediction["ensemble"]["signal"]
                    except Exception as e:
                        logger.error(f"Error getting ensemble predictions for {ticker}: {e}")
                
                # Get HFT signals
                if hft_analyzer:
                    try:
                        hft_signals = hft_analyzer.get_hft_trading_signals(df.tail(100))
                        
                        prediction["hft"] = {
                            "buy_weight": float(hft_signals["buy_weight"]),
                            "sell_weight": float(hft_signals["sell_weight"]),
                            "signal": 1 if hft_signals["action"] == "BUY" else -1 if hft_signals["action"] == "SELL" else 0,
                            "confidence": float(hft_signals["confidence"])
                        }
                        prediction["signals"]["hft"] = prediction["hft"]["signal"]
                    except Exception as e:
                        logger.error(f"Error getting HFT signals for {ticker}: {e}")
                
                # Get RL weights
                if rl_generator:
                    try:
                        buy_weight, sell_weight = rl_generator.generate_weight(latest_data)
                        
                        prediction["rl_weight"] = {
                            "buy_weight": float(buy_weight),
                            "sell_weight": float(sell_weight),
                            "signal": 1 if buy_weight > sell_weight else -1 if sell_weight > buy_weight else 0
                        }
                        prediction["signals"]["rl"] = prediction["rl_weight"]["signal"]
                        
                        # Use RL weights for final weights
                        prediction["weights"]["buy"] = float(buy_weight)
                        prediction["weights"]["sell"] = float(sell_weight)
                    except Exception as e:
                        logger.error(f"Error getting RL weights for {ticker}: {e}")
                
                # Combine signals (simple average)
                if prediction["signals"]:
                    signals = list(prediction["signals"].values())
                    avg_signal = sum(signals) / len(signals)
                    
                    # Convert to probability
                    prediction["probability"] = 0.5 + (avg_signal * 0.25)  # Scale to 0.25-0.75 range
                    
                    # Get confidence as agreement between signals
                    unique_signals = set(signals)
                    if len(unique_signals) == 1:
                        prediction["confidence"] = 1.0  # All agree
                    elif len(unique_signals) == 2 and 0 in unique_signals:
                        prediction["confidence"] = 0.5  # Partial agreement
                    else:
                        prediction["confidence"] = 0.0  # Disagreement
                    
                    # Determine action
                    prob = prediction["probability"]
                    if prob > 0.6:  # Strong buy
                        prediction["action"] = "BUY"
                    elif prob > 0.55:  # Weak buy
                        prediction["action"] = "WEAK BUY"
                    elif prob < 0.4:  # Strong sell
                        prediction["action"] = "SELL"
                    elif prob < 0.45:  # Weak sell
                        prediction["action"] = "WEAK SELL"
                    else:
                        prediction["action"] = "HOLD"
                
                predictions[ticker] = prediction
            
            # Generate portfolio allocation if optimizer is available
            portfolio = {}
            
            if PORTFOLIO_OPTIMIZER_AVAILABLE:
                try:
                    # Create price data for portfolio optimization
                    price_data = pd.DataFrame()
                    
                    for df in data_frames:
                        ticker = df['Symbol'].iloc[0]
                        price_data[ticker] = df.set_index('Timestamp')['Close']
                    
                    # Initialize optimizer
                    optimizer = PortfolioOptimizer()
                    optimizer.load_market_data(price_data)
                    
                    # Optimize portfolio based on regime
                    if current_regime == 1:  # Bull market
                        optimization_result = optimizer.optimize_portfolio(objective='sharpe')
                    elif current_regime == 2:  # Bear market
                        optimization_result = optimizer.optimize_with_drawdown_constraint()
                    else:  # Sideways or unknown
                        optimization_result = optimizer.optimize_with_risk_parity()
                    
                    if optimization_result["success"]:
                        # Create ML weights based on predictions
                        ml_weights = {}
                        for ticker, prediction in predictions.items():
                            buy_weight = prediction["weights"]["buy"]
                            sell_weight = prediction["weights"]["sell"]
                            ml_weights[ticker] = buy_weight - sell_weight
                        
                        # Get allocation with ML weights
                        portfolio = optimizer.get_portfolio_allocation(10000, ml_weights)
                except Exception as e:
                    logger.error(f"Error generating portfolio allocation: {e}")
            
            # Display results
            print("\n" + "=" * 80)
            print("PREDICTION RESULTS")
            print("=" * 80)
            
            print("\nMarket Regime:", regime_description)
            
            print("\nStock Predictions:")
            print("-" * 80)
            print(f"{'Symbol':<6} {'Action':<10} {'Probability':<12} {'Confidence':<12} {'Price':<10}")
            print("-" * 80)
            
            for ticker, prediction in predictions.items():
                print(f"{ticker:<6} {prediction['action']:<10} {prediction['probability']:.2f} ({prediction['probability']*100:.1f}%) "
                      f"{prediction['confidence']:.2f} ({prediction['confidence']*100:.1f}%) ${prediction['price']:<9.2f}")
            
            # Display portfolio
            if portfolio:
                print("\nRecommended Portfolio Allocation (Investment: $10,000):")
                print("-" * 80)
                
                metrics = portfolio.pop("portfolio_metrics", None)
                
                print(f"{'Rank':<4} {'Symbol':<6} {'Weight':<10} {'Amount ($)':<12}")
                print("-" * 80)
                
                # Sort by amount
                sorted_portfolio = sorted(portfolio.items(), key=lambda x: x[1].get('amount', 0), reverse=True)
                
                for i, (ticker, data) in enumerate(sorted_portfolio, 1):
                    if isinstance(data, dict) and 'weight' in data:
                        weight = data.get('weight', 0)
                        amount = data.get('amount', 0)
                        
                        print(f"{i:<4} {ticker:<6} {weight:.2%} ${amount:<11.2f}")
            
            # Save results to file
            try:
                results = {
                    "predictions": predictions,
                    "portfolio": portfolio,
                    "market_regime": {
                        "id": current_regime,
                        "description": regime_description
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                prediction_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(prediction_file, 'w') as f:
                    json.dump(results, f, indent=4, default=str)
                
                logger.info(f"Predictions saved to {prediction_file}")
            except Exception as e:
                logger.error(f"Error saving predictions: {e}")
        
        except Exception as e:
            logger.error(f"Error running predictions: {e}")
            traceback.print_exc()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--check', action='store_true', help='Check component availability')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--predict', action='store_true', help='Run predictions')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers')
    parser.add_argument('--standalone', action='store_true', help='Run using standalone components instead of integrated system')
    
    args = parser.parse_args()
    
    # Check component availability if requested
    if args.check:
        availability = check_component_availability()
        
        print("\n" + "=" * 80)
        print("ENHANCED TRADING SYSTEM COMPONENT AVAILABILITY")
        print("=" * 80)
        
        print("\nComponent Availability:")
        for component, available in availability["components"].items():
            print(f"  {component}: {'Available' if available else 'Not Available'}")
        
        print("\nRequirements:")
        for req, installed in availability["requirements"].items():
            print(f"  {req}: {'Installed' if installed else 'Not Installed'}")
        
        print(f"\nAll Components Available: {'Yes' if availability['all_available'] else 'No'}")
        
        # Suggest installation commands
        print("\nMissing components? Install with:")
        print("  pip install numpy pandas scikit-learn tensorflow yfinance")
        print("  (ML components require manual installation from provided code)")
        
        return
    
    # Decide whether to use integrated system or standalone components
    if TRADING_SYSTEM_AVAILABLE and not args.standalone:
        run_integrated_system(args)
    else:
        if not args.standalone and not TRADING_SYSTEM_AVAILABLE:
            logger.warning("Enhanced Trading System not available, falling back to standalone components")
        run_standalone_components(args)

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ADVANCED ML TRADING SYSTEM WITH ENHANCED CAPABILITIES")
    print("=" * 80)
    print("\nInitializing system...")
    
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        traceback.print_exc()
        sys.exit(1)