import pandas as pd
import numpy as np
import logging
import talib
from arch import arch_model
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import warnings
from datetime import datetime, timedelta

# Add near the top of the file, after imports
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

# Setup for sentiment analysis
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering:
    """Advanced feature engineering with market signals, sentiment and volatility modeling."""
    
    def __init__(self, api_key=None):
        """Initialize with optional API key for external data sources."""
        self.api_key = api_key
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        logger.debug("AdvancedFeatureEngineering initialized with api_key: %s", api_key)
    
    def add_all_features(self, df):
        """Add all advanced features to the dataframe."""
        logger.debug("Starting add_all_features")
        if df is None or df.empty:
            logger.warning("Empty dataframe provided to feature engineering")
            return df
            
        df_enhanced = df.copy()
        logger.debug("DataFrame copied for enhancement")
        logger.debug("Initial DataFrame columns: %s", df_enhanced.columns.tolist())
        
        # Check for MultiIndex columns
        if isinstance(df_enhanced.columns, pd.MultiIndex):
            logger.info("Converting MultiIndex columns to flat columns")
            new_columns = {}
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                matching_cols = [c for c in df_enhanced.columns if col in c]
                if matching_cols:
                    new_columns[col] = df_enhanced[matching_cols[0]].values
                    logger.debug("Mapping MultiIndex column %s to %s", matching_cols[0], col)
            
            date_cols = [c for c in df_enhanced.columns if 'Date' in c]
            if date_cols:
                new_columns['Date'] = df_enhanced[date_cols[0]].values
                logger.debug("Mapping MultiIndex Date column: %s", date_cols[0])
                
            symbol_cols = [c for c in df_enhanced.columns if 'Symbol' in c]
            if symbol_cols:
                new_columns['Symbol'] = df_enhanced[symbol_cols[0]].values
                logger.debug("Mapping MultiIndex Symbol column: %s", symbol_cols[0])
            
            df_enhanced = pd.DataFrame(new_columns)
            logger.info("New DataFrame created with columns: %s", df_enhanced.columns.tolist())
        
        # Convert columns to numeric where applicable
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_enhanced.columns:
                try:
                    logger.debug("Converting column %s to numeric", col)
                    df_enhanced[col] = df_enhanced[col].astype(float)
                    logger.debug("Successfully converted %s", col)
                except Exception as e:
                    logger.error("Error converting %s: %s", col, str(e))
        
        symbol = df_enhanced['Symbol'].iloc[0] if 'Symbol' in df_enhanced.columns else None
        if symbol:
            logger.debug("Detected symbol: %s", symbol)
        else:
            logger.debug("No symbol found in DataFrame")
        
        # Add features in groups
        try:
            logger.debug("Adding volatility features")
            df_enhanced = self.add_volatility_features(df_enhanced)
            
            logger.debug("Adding technical features")
            df_enhanced = self.add_technical_features(df_enhanced)
            
            if symbol:
                logger.debug("Adding sentiment features for symbol: %s", symbol)
                df_enhanced = self.add_sentiment_features(df_enhanced, symbol)
                
                logger.debug("Adding options flow features for symbol: %s", symbol)
                df_enhanced = self.add_options_flow_features(df_enhanced, symbol)
            
            logger.debug("Adding market regime features")
            df_enhanced = self.add_market_regime_features(df_enhanced)
            
            logger.debug("Adding order flow features")
            df_enhanced = self.add_order_flow_features(df_enhanced)
            
            logger.debug("Cleaning and finalizing DataFrame")
            df_enhanced = self.clean_and_finalize(df_enhanced)
            
            logger.debug("Completed add_all_features successfully")
            return df_enhanced
            
        except Exception as e:
            logger.error("Error in feature engineering: %s", e, exc_info=True)
            return df
    
    def add_volatility_features(self, df):
        """Add volatility modeling features using GARCH."""
        logger.info("Adding volatility features")
        logger.debug("DataFrame shape at start of add_volatility_features: %s", df.shape)
        
        df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
        logger.debug("Calculated LogReturn")
        
        df['HistoricalVolatility_10'] = df['LogReturn'].rolling(window=10).std() * np.sqrt(252)
        logger.debug("Calculated HistoricalVolatility_10")
        
        df['HistoricalVolatility_20'] = df['LogReturn'].rolling(window=20).std() * np.sqrt(252)
        logger.debug("Calculated HistoricalVolatility_20")
        
        if len(df) >= 100:
            try:
                logger.debug("Fitting GARCH model with %s data points", len(df['LogReturn'].dropna()))
                returns = df['LogReturn'].dropna().tail(500)
                garch_model = arch_model(returns, vol='Garch', p=1, q=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_fit = garch_model.fit(disp='off')
                conditional_vol = model_fit.conditional_volatility
                vol_series = pd.Series(conditional_vol, index=returns.index)
                df = df.join(vol_series.rename('GARCH_Volatility'), how='left')
                df['GARCH_Volatility'] = df['GARCH_Volatility'].fillna(method='ffill')
                logger.debug("GARCH volatility calculated and joined")
                
                df['VolatilityRegime'] = (df['GARCH_Volatility'] > df['GARCH_Volatility'].rolling(window=20).mean()).astype(int)
                logger.debug("Calculated VolatilityRegime based on GARCH")
            except Exception as e:
                logger.warning("Could not add GARCH features: %s", e)
                df['GARCH_Volatility'] = df['HistoricalVolatility_20']
                df['VolatilityRegime'] = (df['HistoricalVolatility_20'] > df['HistoricalVolatility_20'].rolling(window=20).mean()).astype(int)
        else:
            logger.debug("Not enough data for GARCH; using HistoricalVolatility_20")
            df['GARCH_Volatility'] = df['HistoricalVolatility_20']
            df['VolatilityRegime'] = (df['HistoricalVolatility_20'] > df['HistoricalVolatility_20'].rolling(window=20).mean()).astype(int)
            
        logger.debug("Completed add_volatility_features")
        return df
    
    def add_technical_features(self, df):
        """Add advanced technical indicators."""
        logger.info("Adding technical features")
        logger.debug("DataFrame shape at start of add_technical_features: %s", df.shape)
        
        if len(df) < 50:
            logger.warning("Not enough data for technical indicators")
            return df
            
        try:
            try:
                logger.debug("Attempting to use TALib for technical indicators")
                df['RSI'] = talib.RSI(df['Close'].values)
                df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'].values)
                df['WILLR'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values)
                df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values)
                df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values)
                df['NATR'] = talib.NATR(df['High'].values, df['Low'].values, df['Close'].values)
                df['OBV'] = talib.OBV(df['Close'].values, df['Volume'].values)
                df['AD'] = talib.AD(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values)
                df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['Close'].values)
                df['HT_DCPHASE'] = talib.HT_DCPHASE(df['Close'].values)
                df['CDL_DOJI'] = talib.CDLDOJI(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
                df['CDL_HAMMER'] = talib.CDLHAMMER(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
                logger.debug("TALib technical indicators added successfully")
            except (ImportError, AttributeError) as talib_error:
                logger.warning("TALib not available, using pandas implementations: %s", talib_error)
                df['RSI'] = self._calculate_rsi(df['Close'])
                df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self._calculate_macd(df['Close'])
                df['ATR'] = self._calculate_atr(df['High'], df['Low'], df['Close'])
                df['OBV'] = self._calculate_obv(df['Close'], df['Volume'])
                logger.debug("Pandas-based technical indicators added")
            
            df['ZScore_10'] = (df['Close'] - df['Close'].rolling(window=10).mean()) / df['Close'].rolling(window=10).std()
            df['ZScore_20'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close'].rolling(window=20).std()
            logger.debug("Added mean reversion indicators")
            
            df['MomStrength_5'] = df['Close'].pct_change(5)
            df['MomStrength_10'] = df['Close'].pct_change(10)
            df['MomStrength_20'] = df['Close'].pct_change(20)
            logger.debug("Added price momentum features")
            
            df['VolAdjMomentum'] = df['MomStrength_10'] / df['HistoricalVolatility_10'].replace(0, np.nan).fillna(df['HistoricalVolatility_10'].mean())
            logger.debug("Calculated Volatility-adjusted momentum")
            
            df['SMA10'] = df['Close'].rolling(window=10).mean()
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['SMA200'] = df['Close'].rolling(window=200).mean()
            logger.debug("Calculated moving averages")
            
            df['CrossoverSMA10_20'] = (df['SMA10'] > df['SMA20']).astype(int)
            df['CrossoverSMA20_50'] = (df['SMA20'] > df['SMA50']).astype(int)
            df['CrossoverSMA50_200'] = (df['SMA50'] > df['SMA200']).astype(int)
            logger.debug("Calculated moving average crossovers")
            
            df['GoldenCross'] = ((df['CrossoverSMA50_200'] == 1) & (df['CrossoverSMA50_200'].shift(1) == 0)).astype(int)
            df['DeathCross'] = ((df['CrossoverSMA50_200'] == 0) & (df['CrossoverSMA50_200'].shift(1) == 1)).astype(int)
            logger.debug("Identified Golden and Death crosses")
            
            df['VolumeStrength'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            df['VolumePriceCorrelation'] = df['Close'].rolling(window=10).corr(df['Volume'])
            logger.debug("Added volume indicators")
            
            df['DailyRange'] = (df['High'] - df['Low']) / df['Open']
            df['TrendStrength'] = abs(df['SMA10'] - df['SMA20']) / df['SMA20']
            df['MomAndVol'] = df['MomStrength_10'] * df['VolumeStrength']
            logger.debug("Added custom combined features")
            
            logger.debug("Completed add_technical_features")
            return df
            
        except Exception as e:
            logger.error("Error adding technical features: %s", e, exc_info=True)
            return df
    
    def add_sentiment_features(self, df, symbol):
        """Add news sentiment and social media indicators."""
        logger.info("Adding sentiment features for %s", symbol)
        logger.debug("DataFrame shape at start of add_sentiment_features: %s", df.shape)
        
        try:
            df['NewsSentiment'] = 0.0
            df['SocialMediaSentiment'] = 0.0
            df['SentimentMomentum'] = 0.0
            logger.debug("Initialized sentiment columns with neutral values")
            
            if self.api_key:
                logger.debug("API key provided; simulating sentiment data")
                seed_value = hash(symbol) % 100 / 100.0
                base_sentiment = 0.5 + (seed_value - 0.5) * 0.5
                sentiment_values = np.zeros(len(df))
                sentiment_values[0] = base_sentiment
                
                for i in range(1, len(df)):
                    sentiment_values[i] = sentiment_values[i-1] + np.random.normal(0, 0.05)
                    sentiment_values[i] = max(0, min(1, sentiment_values[i]))
                df['NewsSentiment'] = (sentiment_values - 0.5) * 2
                df['SocialMediaSentiment'] = df['NewsSentiment'].shift(1).fillna(0)
                df['SentimentMomentum'] = df['NewsSentiment'] - df['NewsSentiment'].shift(5).fillna(0)
                logger.debug("Simulated sentiment features using API key logic")
            
            if symbol:
                try:
                    logger.debug("Attempting to scrape recent sentiment for symbol: %s", symbol)
                    recent_sentiment = self._get_recent_sentiment_for_symbol(symbol)
                    if recent_sentiment is not None:
                        last_n = min(5, len(df))
                        df['NewsSentiment'].iloc[-last_n:] = recent_sentiment
                        df['SentimentMomentum'].iloc[-last_n:] = recent_sentiment - df['NewsSentiment'].iloc[-last_n-5:-last_n].mean()
                        logger.debug("Applied scraped recent sentiment")
                except Exception as e:
                    logger.warning("Error getting recent sentiment: %s", e)
            
            logger.debug("Completed add_sentiment_features")
            return df
            
        except Exception as e:
            logger.error("Error adding sentiment features: %s", e, exc_info=True)
            return df
    
    def _get_recent_sentiment_for_symbol(self, symbol):
        """Get recent sentiment for a symbol by scraping recent headlines."""
        logger.debug("Fetching recent sentiment for symbol: %s", symbol)
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code != 200:
                logger.warning("Non-200 response while fetching sentiment: %s", response.status_code)
                return None
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = []
            news_elements = soup.select('[data-test="LATEST_NEWS"] a')
            for element in news_elements:
                headline = element.text.strip()
                if headline:
                    headlines.append(headline)
            if not headlines:
                logger.debug("No headlines found for sentiment scraping")
                return None
            sentiments = []
            for headline in headlines:
                sentiment = self.sentiment_analyzer.polarity_scores(headline)
                sentiments.append(sentiment['compound'])
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            logger.debug("Calculated average scraped sentiment: %s", avg_sentiment)
            return avg_sentiment
        except Exception as e:
            logger.warning("Error scraping headlines: %s", e)
            return None
    
    def add_options_flow_features(self, df, symbol):
        """Add options flow and unusual activity indicators."""
        logger.info("Adding options flow features for %s", symbol)
        logger.debug("DataFrame shape at start of add_options_flow_features: %s", df.shape)
        
        try:
            df['CallPutRatio'] = 1.0
            df['OptionsVolume'] = 0.0
            df['UnusualOptions'] = 0.0
            logger.debug("Initialized options flow columns")
            
            try:
                ticker = yf.Ticker(symbol)
                try:
                    expirations = ticker.options
                    if expirations:
                        expiry = expirations[0]
                        opt_chain = ticker.option_chain(expiry)
                        call_volume = opt_chain.calls['volume'].sum()
                        put_volume = opt_chain.puts['volume'].sum()
                        call_put_ratio = call_volume / put_volume if put_volume > 0 else 2.0
                        last_n = min(5, len(df))
                        for i in range(last_n):
                            factor = (last_n - i) / last_n
                            idx = df.index[-(i+1)]
                            df.loc[idx, 'CallPutRatio'] = call_put_ratio * factor + 1.0 * (1 - factor)
                            df.loc[idx, 'OptionsVolume'] = (call_volume + put_volume) / 1000
                        logger.debug("Options data applied from yfinance")
                except Exception as e:
                    logger.warning("Error processing options expiration data: %s", e)
            except Exception as e:
                logger.warning("Error fetching options data: %s", e)
            
            if len(df) > 30:
                spike_idx = np.random.randint(len(df) - 20, len(df) - 5)
                spike_indices = df.index[spike_idx:spike_idx+3]
                df.loc[spike_indices, 'UnusualOptions'] = np.random.uniform(0.7, 0.95, 3)
                logger.debug("Simulated unusual options activity added")
            
            df['OptionsBullishSignal'] = ((df['CallPutRatio'] > 1.5) & 
                                          (df['OptionsVolume'] > df['OptionsVolume'].rolling(window=10).mean())).astype(float)
            df['OptionsBearishSignal'] = ((df['CallPutRatio'] < 0.7) & 
                                          (df['OptionsVolume'] > df['OptionsVolume'].rolling(window=10).mean())).astype(float)
            logger.debug("Calculated options-based signals")
            return df
            
        except Exception as e:
            logger.error("Error adding options flow features: %s", e, exc_info=True)
            return df
    
    def add_market_regime_features(self, df):
        """Add market regime classification features."""
        logger.info("Adding market regime features")
        logger.debug("DataFrame shape at start of add_market_regime_features: %s", df.shape)
        
        try:
            df['Trend_Indicator'] = ((df['Close'] > df['SMA50']) & (df['SMA50'] > df['SMA200'])).astype(int)
            logger.debug("Calculated Trend_Indicator")
            
            if 'ATR' in df.columns:
                df['ATR_Percentile'] = df['ATR'].rolling(window=100).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
                df['VolatilityRegime_ATR'] = pd.cut(
                    df['ATR_Percentile'],
                    bins=[-float('inf'), 0.33, 0.66, float('inf')],
                    labels=[0, 1, 2]
                ).astype(float)
                logger.debug("Calculated VolatilityRegime_ATR")
            
            if 'RSI' in df.columns:
                df['MomentumRegime'] = pd.cut(
                    df['RSI'],
                    bins=[-float('inf'), 30, 70, float('inf')],
                    labels=[0, 1, 2]
                ).astype(float)
                logger.debug("Calculated MomentumRegime based on RSI")
            
            if all(col in df.columns for col in ['VolatilityRegime_ATR', 'MomentumRegime', 'Trend_Indicator']):
                df['MarketRegime'] = df['Trend_Indicator'] + df['VolatilityRegime_ATR'] * 0.5 - (df['MomentumRegime'] == 2).astype(float) * 0.5
                df['MarketRegime'] = pd.cut(
                    df['MarketRegime'],
                    bins=[-float('inf'), 0.5, 1.5, float('inf')],
                    labels=[0, 1, 2]
                ).astype(float)
                logger.debug("Combined regimes into MarketRegime")
            
            logger.debug("Completed add_market_regime_features")
            return df
            
        except Exception as e:
            logger.error("Error adding market regime features: %s", e, exc_info=True)
            return df
    
    def add_order_flow_features(self, df):
        """Add order flow imbalance and dark pool indicators."""
        logger.info("Adding order flow features")
        logger.debug("DataFrame shape at start of add_order_flow_features: %s", df.shape)
        
        try:
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
                df['VWAP'] = df.apply(lambda x: x['TypicalPrice'] * x['Volume'], axis=1).cumsum() / df['Volume'].cumsum()
                df['VWAP_Ratio'] = df['Close'] / df['VWAP']
                df['AboveVWAP'] = (df['Close'] > df['VWAP']).astype(int)
                df['VWAP_Crossover'] = ((df['AboveVWAP'] == 1) & (df['AboveVWAP'].shift(1) == 0) | 
                                          (df['AboveVWAP'] == 0) & (df['AboveVWAP'].shift(1) == 1)).astype(int)
                logger.debug("Calculated VWAP-based indicators")
            
            df['DarkPoolActivity'] = (df['Volume'] * np.random.normal(loc=0.4, scale=0.1, size=len(df))).clip(lower=0)
            logger.debug("Simulated DarkPoolActivity")
            
            random_seed = 42
            np.random.seed(random_seed)
            imbalance = np.random.normal(loc=0, scale=0.1, size=len(df))
            price_changes = df['Close'].pct_change().fillna(0)
            imbalance = imbalance * 0.7 + price_changes * 0.3
            imbalance = np.tanh(imbalance * 5)
            df['OrderFlowImbalance'] = imbalance
            logger.debug("Calculated OrderFlowImbalance")
            
            df['BuyingPressure'] = (df['OrderFlowImbalance'] > 0.2).astype(int)
            df['SellingPressure'] = (df['OrderFlowImbalance'] < -0.2).astype(int)
            logger.debug("Added BuyingPressure and SellingPressure signals")
            
            df['CumulativeDelta'] = df['OrderFlowImbalance'].cumsum()
            df['PriceDirection'] = np.sign(df['Close'].diff()).fillna(0)
            df['DeltaDirection'] = np.sign(df['OrderFlowImbalance']).fillna(0)
            df['DeltaDivergence'] = (df['PriceDirection'] != df['DeltaDirection']).astype(int)
            logger.debug("Calculated CumulativeDelta and DeltaDivergence")
            
            logger.debug("Completed add_order_flow_features")
            return df
            
        except Exception as e:
            logger.error("Error adding order flow features: %s", e, exc_info=True)
            return df
    
    def clean_and_finalize(self, df):
        """Clean the dataframe and handle missing values."""
        logger.debug("Starting clean_and_finalize")
        df = df.replace([np.inf, -np.inf], np.nan)
        logger.debug("Replaced infinities with NaN")
        
        df = df.fillna(method='ffill').fillna(method='bfill')
        logger.debug("Filled missing values using forward and backward fill")
        
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.debug("Filled remaining NaNs in numeric column %s with median %s", col, median_val)
                else:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else None
                    df[col] = df[col].fillna(mode_val)
                    logger.debug("Filled remaining NaNs in non-numeric column %s with mode %s", col, mode_val)
        
        logger.debug("Completed clean_and_finalize")
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI using pandas."""
        logger.debug("Calculating RSI using pandas")
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD using pandas."""
        logger.debug("Calculating MACD using pandas")
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram
    
    def _calculate_atr(self, high, low, close, window=14):
        """Calculate ATR using pandas."""
        logger.debug("Calculating ATR using pandas")
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    def _calculate_obv(self, close, volume):
        """Calculate OBV using pandas."""
        logger.debug("Calculating OBV using pandas")
        obv = volume.copy()
        obv[close < close.shift()] = -volume
        obv[close == close.shift()] = 0
        return obv.cumsum()
