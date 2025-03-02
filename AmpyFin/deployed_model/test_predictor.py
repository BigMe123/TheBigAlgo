# test_predictor.py
import pandas as pd
import yfinance as yf
import logging
from ml_trading_predictor import MLTradingPredictor

# Set up simple logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Initialize the predictor
    print("Initializing ML Trading Predictor...")
    predictor = MLTradingPredictor(model_dir="./models")
    
    # Download some sample stock data
    print("\nDownloading sample data for AAPL...")
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    df = stock.history(period="30d", interval="1d")
    
    # Add Symbol column and reset index to prepare data
    df["Symbol"] = ticker
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Timestamp"}, inplace=True)
    
    print(f"\nSample data shape: {df.shape}")
    print(df.head(3))
    
    # Generate label for training (1 if next close > current close)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['Label'] = df['Close'].shift(-1) > df['Close']
    df['Label'] = df['Label'].astype(int)
    
    # Check if model exists and train if needed
    if predictor.model is None:
        print("\nNo existing model found. Training a new model...")
        # Note: In a real scenario, we'd use more data, but this is just a demonstration
        success = predictor.train_model([df])
        if success:
            print("Model trained successfully!")
        else:
            print("Model training failed.")
            return
    
    # Make a prediction on the most recent data
    print("\nMaking prediction on recent data...")
    prob, confidence = predictor.predict(df)
    
    print(f"\nPrediction Results:")
    print(f"Probability of price increase: {prob:.2f} (50% = neutral)")
    print(f"Confidence: {confidence:.2f} (0 = no confidence, 1 = high confidence)")
    
    direction = "BULLISH" if prob > 0.5 else "BEARISH" if prob < 0.5 else "NEUTRAL"
    print(f"Overall market direction prediction: {direction}")
    
    # Get trading weights
    buy_weight, sell_weight = predictor.get_ml_weight(df)
    print(f"\nTrading recommendation:")
    print(f"Buy weight: {buy_weight:.2f}")
    print(f"Sell weight: {sell_weight:.2f}")
    
    if buy_weight > sell_weight:
        print("Recommendation: BUY")
    elif sell_weight > buy_weight:
        print("Recommendation: SELL")
    else:
        print("Recommendation: HOLD")

if __name__ == "__main__":
    main()