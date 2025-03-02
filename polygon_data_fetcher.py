import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, period="7d", interval="1m"):
    """
    Fetches high-frequency intraday stock data from Yahoo Finance.

    Parameters:
        ticker (str): The stock symbol (e.g., "AAPL").
        period (str): How much history to fetch (default: "7d" for 7 days).
        interval (str): The frequency of data (e.g., "1m" for 1-minute bars).

    Returns:
        pd.DataFrame: DataFrame with timestamped OHLCV intraday data.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Reset index and rename columns
        df.reset_index(inplace=True)
        df.rename(columns={"Datetime": "Timestamp"}, inplace=True)
        
        # Ensure timestamp column is in correct format
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Add ticker symbol to each row
        df["Symbol"] = ticker

        return df
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    ticker = "AAPL"
    df_intraday = fetch_stock_data(ticker, period="7d", interval="1m")  # 1-minute interval
    print(df_intraday.head())
