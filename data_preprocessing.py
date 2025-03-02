import yfinance as yf
import pandas as pd

def get_intraday_data(ticker, interval="5m", period="5d", start_date=None):
    """
    Fetches intraday trading data from Yahoo Finance.
    
    Parameters:
        ticker (str): Stock symbol (e.g., "AAPL").
        interval (str): Timeframe for data (e.g., "1m", "5m", "15m").
        period (str): Lookback period (e.g., "1d", "5d", "1mo").
        start_date (str): Start date for fetching data (optional).
    
    Returns:
        pd.DataFrame: Intraday trading data.
    """
    stock = yf.Ticker(ticker)

    # Use `start_date` if provided, else default to period
    if start_date:
        df = stock.history(start=start_date, interval=interval)
    else:
        df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker, interval, or date range.")

    df["Symbol"] = ticker
    df.reset_index(inplace=True)
    df.rename(columns={"Datetime": "Timestamp"}, inplace=True)

    return df

def add_intraday_label(df):
    """
    Adds a 'Label' column: 1 if next Close > current Close, else 0.
    """
    df = df.sort_values(['Symbol', 'Timestamp']).reset_index(drop=True)
    df['Label'] = df.groupby('Symbol')['Close'].shift(-1) > df['Close']
    df['Label'] = df['Label'].astype(int)
    df.dropna(subset=['Label'], inplace=True)
    return df

def load_and_preprocess_data(ticker, interval="5m", period="5d", start_date=None):
    """
    Fetches and preprocesses intraday trading data.
    """
    df = get_intraday_data(ticker, interval, period, start_date=start_date)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.dropna(subset=['Timestamp'], inplace=True)
    df = add_intraday_label(df)
    return df
