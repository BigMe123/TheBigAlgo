# hft_analyzer.py
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
import os
import json
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HFTAnalyzer:
    """
    High-Frequency Trading (HFT) analysis module.
    
    Provides microstructure analysis, order flow modeling, and fast signal processing
    for high-frequency trading applications.
    """
    
    def __init__(self, config=None):
        """
        Initialize the HFT analyzer.
        
        Args:
            config: Configuration dictionary for the analyzer
        """
        # Default configuration
        self.default_config = {
            'order_imbalance_window': 10,
            'vwap_window': 20,
            'kalman_process_noise': 0.01,
            'kalman_measurement_noise': 0.1,
            'statistical_arb_threshold': 2.0,
            'price_impact_decay': 0.5,
            'tick_size': 0.01,
            'order_book_levels': 5
        }
        
        # Use provided config or default
        self.config = config or self.default_config
        
        # Kalman filter for price prediction
        self.price_kf = None
        
        # Deques for real-time analysis
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.vwap_history = deque(maxlen=100)
        
        # Order book state (simulated)
        self.bid_book = {}
        self.ask_book = {}
        
        # Initialize components
        self._init_kalman_filter()
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter for price prediction."""
        # Simple constant velocity model
        self.price_kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # State transition matrix (constant velocity model)
        self.price_kf.F = np.array([[1., 1.],
                                  [0., 1.]])
        
        # Measurement matrix (we only observe position)
        self.price_kf.H = np.array([[1., 0.]])
        
        # Covariance matrix
        self.price_kf.P *= 1000.
        
        # Process noise
        q = self.config['kalman_process_noise']
        self.price_kf.Q = np.array([[q, 0],
                                  [0, q]])
        
        # Measurement noise
        r = self.config['kalman_measurement_noise']
        self.price_kf.R = np.array([[r]])
    
    def process_tick_data(self, tick_data):
        """
        Process tick-level data for HFT analysis.
        
        Args:
            tick_data: DataFrame with tick data including price, volume, and timestamp
            
        Returns:
            DataFrame: Enhanced tick data with HFT metrics
        """
        try:
            if tick_data.empty:
                logger.warning("Empty tick data provided")
                return tick_data
            
            # Make a copy to avoid modifying the original
            enhanced_data = tick_data.copy()
            
            # Ensure required columns are present
            required_cols = ['price', 'volume', 'timestamp']
            if not all(col in enhanced_data.columns for col in required_cols):
                # Try to map common column names
                column_map = {
                    'Price': 'price',
                    'Volume': 'volume',
                    'Timestamp': 'timestamp',
                    'Close': 'price',
                    'Size': 'volume',
                    'Time': 'timestamp'
                }
                
                for old_col, new_col in column_map.items():
                    if old_col in enhanced_data.columns and new_col not in enhanced_data.columns:
                        enhanced_data[new_col] = enhanced_data[old_col]
            
            # Check again after mapping
            if not all(col in enhanced_data.columns for col in required_cols):
                logger.warning("Missing required columns in tick data")
                return tick_data
            
            # Sort by timestamp if needed
            if not pd.api.types.is_datetime64_dtype(enhanced_data['timestamp']):
                enhanced_data['timestamp'] = pd.to_datetime(enhanced_data['timestamp'])
            
            enhanced_data = enhanced_data.sort_values('timestamp')
            
            # Add microsecond timestamps for true HFT analysis
            if 'microtime' not in enhanced_data.columns:
                enhanced_data['microtime'] = enhanced_data['timestamp'].astype(int) // 1000
            
            # Process data sequentially to simulate real-time processing
            self._add_realtime_metrics(enhanced_data)
            
            # Add microstructure metrics
            enhanced_data = self._add_microstructure_metrics(enhanced_data)
            
            # Add statistical arbitrage signals
            enhanced_data = self._add_statistical_arbitrage(enhanced_data)
            
            # Add Kalman filtered signals
            enhanced_data = self._add_kalman_signals(enhanced_data)
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return tick_data
    
    def _add_realtime_metrics(self, data):
        """
        Add real-time metrics to tick data, processing it sequentially.
        
        Args:
            data: DataFrame with tick data
            
        Returns:
            None (modifies data in-place)
        """
        # Calculate VWAP
        data['volume_price'] = data['price'] * data['volume']
        data['cum_volume'] = data['volume'].cumsum()
        data['cum_volume_price'] = data['volume_price'].cumsum()
        data['vwap'] = data['cum_volume_price'] / data['cum_volume']
        
        # Calculate rolling VWAP
        window = self.config['vwap_window']
        data['rolling_vwap'] = (data['volume_price']
                              .rolling(window=window, min_periods=1)
                              .sum() /
                              data['volume']
                              .rolling(window=window, min_periods=1)
                              .sum())
        
        # Calculate price changes
        data['price_change'] = data['price'].diff()
        
        # Calculate time between ticks
        data['time_delta'] = data['timestamp'].diff().dt.total_seconds()
        
        # Calculate trade direction (1 for up, -1 for down, 0 for no change)
        data['trade_direction'] = np.sign(data['price_change'])
        
        # Calculate tick speed (ticks per second)
        data['tick_speed'] = 1 / data['time_delta']
        data['tick_speed'] = data['tick_speed'].fillna(0).replace(np.inf, 0)
        
        # Calculate volume imbalance
        up_vol = data.loc[data['trade_direction'] > 0, 'volume']
        down_vol = data.loc[data['trade_direction'] < 0, 'volume']
        
        # Handle first row
        up_vol.iloc[0] = 0 if up_vol.empty else up_vol.iloc[0]
        down_vol.iloc[0] = 0 if down_vol.empty else down_vol.iloc[0]
        
        data['up_volume'] = up_vol
        data['down_volume'] = down_vol
        
        # Fill NaN values
        data['up_volume'] = data['up_volume'].fillna(0)
        data['down_volume'] = data['down_volume'].fillna(0)
        
        # Calculate volume imbalance
        data['volume_imbalance'] = data['up_volume'] - data['down_volume']
        
        # Calculate cumulative volume imbalance
        data['cum_volume_imbalance'] = data['volume_imbalance'].cumsum()
        
        # Calculate order imbalance
        window = self.config['order_imbalance_window']
        data['order_imbalance'] = (data['volume_imbalance']
                                 .rolling(window=window, min_periods=1)
                                 .sum())
        
        # Calculate relative strength
        data['rel_strength'] = data['order_imbalance'] / data['volume'].rolling(window=window, min_periods=1).sum()
        data['rel_strength'] = data['rel_strength'].fillna(0)
    
    def _add_microstructure_metrics(self, data):
        """
        Add market microstructure metrics to tick data.
        
        Args:
            data: DataFrame with tick data
            
        Returns:
            DataFrame: Enhanced tick data with microstructure metrics
        """
        # Calculate effective spread
        if 'bid_price' in data.columns and 'ask_price' in data.columns:
            data['mid_price'] = (data['bid_price'] + data['ask_price']) / 2
            data['effective_spread'] = 2 * abs(data['price'] - data['mid_price'])
            data['relative_spread'] = data['effective_spread'] / data['mid_price']
        else:
            # Estimate mid price using Kalman filter or moving average
            data['mid_price'] = data['price'].rolling(window=3, min_periods=1).mean()
            # Estimate spread using typical tick size
            tick_size = self.config['tick_size']
            data['effective_spread'] = tick_size
            data['relative_spread'] = tick_size / data['price']
        
        # Calculate price impact
        window = self.config['vwap_window']
        data['price_impact'] = (data['price'].diff(window) / 
                              (data['volume'].rolling(window=window, min_periods=1).sum() ** 
                               self.config['price_impact_decay']))
        
        # Calculate realized volatility
        data['returns'] = data['price'].pct_change()
        data['squared_returns'] = data['returns'] ** 2
        data['realized_vol'] = np.sqrt(data['squared_returns']
                                     .rolling(window=window, min_periods=1)
                                     .sum() * (252 * 7 * 24 * 60))  # Annualized for minute data
        
        # Calculate order flow toxicity (VPIN proxy)
        # Volume-synchronized probability of informed trading
        data['abs_volume_imbalance'] = abs(data['volume_imbalance'])
        data['vpin'] = (data['abs_volume_imbalance']
                      .rolling(window=window, min_periods=1)
                      .sum() / 
                      data['volume']
                      .rolling(window=window, min_periods=1)
                      .sum())
        
        # Calculate market flow pressure
        data['buy_pressure'] = (data['up_volume']
                             .rolling(window=window, min_periods=1)
                             .sum() / 
                             data['volume']
                             .rolling(window=window, min_periods=1)
                             .sum())
        
        data['sell_pressure'] = (data['down_volume']
                             .rolling(window=window, min_periods=1)
                             .sum() / 
                             data['volume']
                             .rolling(window=window, min_periods=1)
                             .sum())
        
        data['pressure_ratio'] = data['buy_pressure'] / data['sell_pressure']
        data['pressure_ratio'] = data['pressure_ratio'].fillna(1).replace(np.inf, 10)
        
        # Clean up any NaN values
        for col in ['effective_spread', 'relative_spread', 'price_impact', 
                    'realized_vol', 'vpin', 'pressure_ratio']:
            data[col] = data[col].fillna(0)
        
        return data
    
    def _add_statistical_arbitrage(self, data):
        """
        Add statistical arbitrage signals to tick data.
        
        Args:
            data: DataFrame with tick data
            
        Returns:
            DataFrame: Enhanced tick data with statistical arbitrage metrics
        """
        # Deviation from VWAP
        data['vwap_deviation'] = (data['price'] - data['vwap']) / data['vwap']
        
        # Z-score of price relative to short-term moving average
        window = self.config['vwap_window']
        data['sma'] = data['price'].rolling(window=window, min_periods=1).mean()
        data['sma_std'] = data['price'].rolling(window=window, min_periods=1).std()
        data['price_zscore'] = (data['price'] - data['sma']) / data['sma_std'].replace(0, 1)
        
        # Smooth z-score with Savitzky-Golay filter for noise reduction
        try:
            if len(data) >= window:
                data['smooth_zscore'] = savgol_filter(
                    data['price_zscore'].fillna(0).values, 
                    min(window, len(data) // 2 * 2 + 1),  # Window must be odd and not larger than data
                    3  # Polynomial order
                )
            else:
                data['smooth_zscore'] = data['price_zscore']
        except Exception as e:
            logger.warning(f"Error applying Savitzky-Golay filter: {e}")
            data['smooth_zscore'] = data['price_zscore']
        
        # Mean reversion signal
        threshold = self.config['statistical_arb_threshold']
        data['mean_reversion_signal'] = np.where(
            data['smooth_zscore'] > threshold, -1,  # Sell when too high
            np.where(data['smooth_zscore'] < -threshold, 1, 0)  # Buy when too low
        )
        
        # Momentum signal
        data['momentum_signal'] = np.sign(data['order_imbalance'])
        
        # Combined signal (mean reversion + momentum)
        data['combined_signal'] = (0.6 * data['mean_reversion_signal'] + 
                                 0.4 * data['momentum_signal'])
        
        # Calculate statistical arbitrage signal strength
        data['arb_strength'] = abs(data['smooth_zscore'] / threshold) * np.sign(data['combined_signal'])
        
        return data
    
    def _add_kalman_signals(self, data):
        """
        Add Kalman filtered signals to tick data.
        
        Args:
            data: DataFrame with tick data
            
        Returns:
            DataFrame: Enhanced tick data with Kalman filtered signals
        """
        # Reset Kalman filter
        self._init_kalman_filter()
        
        # Arrays to store filtered values
        n = len(data)
        filtered_prices = np.zeros(n)
        price_velocities = np.zeros(n)
        prediction_errors = np.zeros(n)
        
        # Process sequentially to simulate real-time filtering
        for i, (_, row) in enumerate(data.iterrows()):
            # Get price
            price = row['price']
            
            # Predict next state
            self.price_kf.predict()
            
            # Update with measurement
            self.price_kf.update(price)
            
            # Store filtered values
            filtered_prices[i] = self.price_kf.x[0]  # Position
            price_velocities[i] = self.price_kf.x[1]  # Velocity
            prediction_errors[i] = price - self.price_kf.x[0]  # Innovation
        
        # Add to dataframe
        data['kalman_price'] = filtered_prices
        data['price_velocity'] = price_velocities
        data['prediction_error'] = prediction_errors
        
        # Add Kalman-based trading signals
        data['kalman_signal'] = np.sign(data['price_velocity'])
        
        # Add prediction-based signal
        data['prediction_signal'] = np.where(
            abs(data['prediction_error']) > self.config['tick_size'] * 2,
            -np.sign(data['prediction_error']),  # Trade against large errors (mean reversion)
            np.sign(data['price_velocity'])  # Trade with the trend for small errors
        )
        
        return data
    
    def simulate_order_book(self, base_price, spread=0.01, depth=5, liquidity=1000):
        """
        Simulate a basic order book for testing.
        
        Args:
            base_price: Current mid price
            spread: Bid-ask spread
            depth: Number of price levels to simulate
            liquidity: Base liquidity (quantity) at each level
            
        Returns:
            tuple: (bid book, ask book) as dictionaries of price -> quantity
        """
        # Reset order books
        self.bid_book = {}
        self.ask_book = {}
        
        # Calculate basic bid and ask prices
        bid_price = base_price - spread / 2
        ask_price = base_price + spread / 2
        
        # Generate bid side (descending prices)
        for i in range(depth):
            price = round(bid_price - i * spread, 4)
            # Increase liquidity as we move away from the mid price
            quantity = liquidity * (1 + i * 0.2)
            self.bid_book[price] = quantity
        
        # Generate ask side (ascending prices)
        for i in range(depth):
            price = round(ask_price + i * spread, 4)
            # Increase liquidity as we move away from the mid price
            quantity = liquidity * (1 + i * 0.2)
            self.ask_book[price] = quantity
        
        return self.bid_book, self.ask_book
    
    def apply_market_order(self, side, quantity):
        """
        Simulate the impact of a market order on the order book.
        
        Args:
            side: 'buy' or 'sell'
            quantity: Order quantity
            
        Returns:
            tuple: (executed price, executed quantity, remaining quantity)
        """
        executed_quantity = 0
        remaining_quantity = quantity
        executed_value = 0
        
        if side.lower() == 'buy':
            # Market buy order takes liquidity from ask book
            # Sort asks by price (ascending)
            sorted_asks = sorted(self.ask_book.items())
            
            for price, available_qty in sorted_asks:
                if remaining_quantity <= 0:
                    break
                
                # Calculate how much we can execute at this level
                exec_qty = min(remaining_quantity, available_qty)
                
                # Update order book
                self.ask_book[price] -= exec_qty
                if self.ask_book[price] <= 0:
                    del self.ask_book[price]
                
                # Update execution tracking
                executed_quantity += exec_qty
                executed_value += exec_qty * price
                remaining_quantity -= exec_qty
        
        elif side.lower() == 'sell':
            # Market sell order takes liquidity from bid book
            # Sort bids by price (descending)
            sorted_bids = sorted(self.bid_book.items(), reverse=True)
            
            for price, available_qty in sorted_bids:
                if remaining_quantity <= 0:
                    break
                
                # Calculate how much we can execute at this level
                exec_qty = min(remaining_quantity, available_qty)
                
                # Update order book
                self.bid_book[price] -= exec_qty
                if self.bid_book[price] <= 0:
                    del self.bid_book[price]
                
                # Update execution tracking
                executed_quantity += exec_qty
                executed_value += exec_qty * price
                remaining_quantity -= exec_qty
        
        # Calculate average execution price
        if executed_quantity > 0:
            avg_price = executed_value / executed_quantity
        else:
            avg_price = None
        
        return avg_price, executed_quantity, remaining_quantity
    
    def get_optimal_execution_strategy(self, side, quantity, max_time_minutes=10):
        """
        Calculate an optimal execution strategy to minimize market impact.
        
        Args:
            side: 'buy' or 'sell'
            quantity: Total quantity to execute
            max_time_minutes: Maximum time for execution in minutes
            
        Returns:
            dict: Execution strategy
        """
        # Initialize strategy
        strategy = {
            'side': side,
            'total_quantity': quantity,
            'max_time_minutes': max_time_minutes,
            'twap_quantity': quantity / max_time_minutes,
            'schedule': []
        }
        
        # Estimate market impact coefficient based on order book depth
        if side.lower() == 'buy':
            book = self.ask_book
        else:
            book = self.bid_book
        
        # Calculate total liquidity available
        total_liquidity = sum(book.values()) if book else 1000
        
        # Market impact coefficient (higher means more impact)
        impact_coef = min(1.0, quantity / (total_liquidity * 2))
        
        # Generate execution schedule (U-shaped - more at beginning and end)
        # Implements a simplified Almgren-Chriss model
        time_points = np.linspace(0, max_time_minutes, 10)
        
        total_allocated = 0
        schedule = []
        
        for i, t in enumerate(time_points):
            # Higher execution rate at beginning and end
            if i == 0 or i == len(time_points) - 1:
                rate = 0.15  # 15% at start and end
            else:
                # U-shaped curve
                x = (t / max_time_minutes - 0.5) * 2  # Normalize to [-1, 1]
                rate = 0.05 + 0.05 * x * x  # Minimum 5% in middle, higher at edges
            
            # Adjust for remaining quantity
            if i == len(time_points) - 1:
                # Last chunk gets whatever is left
                chunk_quantity = quantity - total_allocated
            else:
                chunk_quantity = rate * quantity
                total_allocated += chunk_quantity
            
            # Add to schedule
            schedule.append({
                'time_point': t,
                'quantity': chunk_quantity,
                'expected_impact': impact_coef * chunk_quantity * 0.01  # Estimated price impact
            })
        
        strategy['schedule'] = schedule
        strategy['estimated_total_impact'] = sum(s['expected_impact'] for s in schedule)
        
        return strategy
    
    def get_order_imbalance(self):
        """
        Calculate the order imbalance from the current order book.
        
        Returns:
            float: Order imbalance ratio (-1 to 1)
        """
        # Calculate total bid and ask quantities
        total_bid_qty = sum(self.bid_book.values())
        total_ask_qty = sum(self.ask_book.values())
        
        # Calculate imbalance
        total_qty = total_bid_qty + total_ask_qty
        if total_qty > 0:
            imbalance = (total_bid_qty - total_ask_qty) / total_qty
        else:
            imbalance = 0
        
        return imbalance
    
    def analyze_order_flow(self, tick_data, window=100):
        """
        Analyze order flow and provide a directional bias based on tick data.
        
        Args:
            tick_data: DataFrame with tick data
            window: Window size for analysis
            
        Returns:
            dict: Order flow analysis
        """
        # Process tick data if it hasn't been processed
        if 'order_imbalance' not in tick_data.columns:
            tick_data = self.process_tick_data(tick_data)
        
        # Get latest data
        recent_data = tick_data.tail(window)
        
        # Calculate key metrics
        vwap = recent_data['vwap'].iloc[-1]
        current_price = recent_data['price'].iloc[-1]
        order_imbalance = recent_data['order_imbalance'].iloc[-1]
        buy_pressure = recent_data['buy_pressure'].iloc[-1]
        sell_pressure = recent_data['sell_pressure'].iloc[-1]
        price_velocity = recent_data['price_velocity'].iloc[-1]
        arb_strength = recent_data['arb_strength'].iloc[-1]
        
        # Calculate directional bias
        bias_factors = [
            np.sign(current_price - vwap) * 0.2,  # Price vs VWAP
            np.sign(order_imbalance) * 0.3,       # Order imbalance
            np.sign(buy_pressure - sell_pressure) * 0.2,  # Pressure difference
            np.sign(price_velocity) * 0.2,        # Price velocity
            np.sign(arb_strength) * 0.1           # Statistical arb signal
        ]
        
        directional_bias = sum(bias_factors)
        
        # Classification
        if directional_bias > 0.3:
            bias_class = "Strongly Bullish"
        elif directional_bias > 0.1:
            bias_class = "Bullish"
        elif directional_bias < -0.3:
            bias_class = "Strongly Bearish"
        elif directional_bias < -0.1:
            bias_class = "Bearish"
        else:
            bias_class = "Neutral"
        
        # Confidence based on signal alignment
        aligned_signals = sum(1 for f in bias_factors if np.sign(f) == np.sign(directional_bias))
        confidence = aligned_signals / len(bias_factors)
        
        return {
            'directional_bias': directional_bias,
            'classification': bias_class,
            'confidence': confidence,
            'vwap': vwap,
            'current_price': current_price,
            'order_imbalance': order_imbalance,
            'price_velocity': price_velocity,
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'arb_strength': arb_strength,
            'book_imbalance': self.get_order_imbalance()
        }
    
    def create_order_flow_features(self, data):
        """
        Create order flow features for machine learning models.
        
        Args:
            data: DataFrame with processed tick data
            
        Returns:
            DataFrame: Feature matrix for ML
        """
        # Process data if necessary
        if 'order_imbalance' not in data.columns:
            data = self.process_tick_data(data)
        
        # Select relevant features
        features = pd.DataFrame()
        
        # Price-based features
        features['price_vs_vwap'] = (data['price'] - data['vwap']) / data['vwap']
        features['price_vs_kalman'] = (data['price'] - data['kalman_price']) / data['kalman_price']
        features['price_velocity'] = data['price_velocity']
        
        # Order flow features
        features['order_imbalance'] = data['order_imbalance']
        features['rel_strength'] = data['rel_strength']
        features['vpin'] = data['vpin']
        features['pressure_ratio'] = data['pressure_ratio']
        features['buy_pressure'] = data['buy_pressure']
        features['sell_pressure'] = data['sell_pressure']
        
        # Microstructure features
        features['effective_spread'] = data['effective_spread']
        features['price_impact'] = data['price_impact']
        features['realized_vol'] = data['realized_vol']
        
        # Statistical arbitrage features
        features['price_zscore'] = data['price_zscore']
        features['smooth_zscore'] = data['smooth_zscore']
        features['arb_strength'] = data['arb_strength']
        
        # Signal features
        features['mean_reversion_signal'] = data['mean_reversion_signal']
        features['momentum_signal'] = data['momentum_signal']
        features['kalman_signal'] = data['kalman_signal']
        features['prediction_signal'] = data['prediction_signal']
        features['combined_signal'] = data['combined_signal']
        
        # Clean up any NaN or inf values
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return features
    
    def get_hft_trading_signals(self, recent_data, threshold=0.2):
        """
        Generate HFT trading signals from recent market data.
        
        Args:
            recent_data: DataFrame with processed tick data (recent)
            threshold: Signal threshold for trading decisions
            
        Returns:
            dict: Trading signals and metrics
        """
        # Process data if necessary
        if 'order_imbalance' not in recent_data.columns:
            recent_data = self.process_tick_data(recent_data)
        
        # Get last row for current signals
        latest = recent_data.iloc[-1]
        
        # Calculate combined signal
        signal_weights = {
            'order_imbalance': 0.20,
            'price_velocity': 0.15,
            'arb_strength': 0.15,
            'mean_reversion_signal': 0.15,
            'momentum_signal': 0.10,
            'kalman_signal': 0.15,
            'prediction_signal': 0.10
        }
        
        combined_signal = 0
        for feature, weight in signal_weights.items():
            if feature in latest:
                # Normalize to [-1, 1] if needed
                value = latest[feature]
                if feature == 'order_imbalance':
                    # Normalize based on typical range
                    norm_value = np.clip(value / 1000, -1, 1)
                elif feature == 'arb_strength':
                    # Already normalized
                    norm_value = value
                elif feature in ['mean_reversion_signal', 'momentum_signal', 'kalman_signal', 'prediction_signal']:
                    # Already in [-1, 1]
                    norm_value = value
                else:
                    # Default normalization
                    norm_value = np.clip(value, -1, 1)
                
                combined_signal += weight * norm_value
        
        # Determine trading action
        if combined_signal > threshold:
            action = "BUY"
        elif combined_signal < -threshold:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Calculate signal strength (for position sizing)
        signal_strength = abs(combined_signal)
        
        # Generate weights for ML model
        buy_weight = max(0, combined_signal) * 1000
        sell_weight = max(0, -combined_signal) * 1000
        
        # Calculate confidence based on signal alignment
        component_signals = [
            np.sign(latest['order_imbalance']) if abs(latest['order_imbalance']) > 100 else 0,
            np.sign(latest['price_velocity']),
            latest['mean_reversion_signal'],
            latest['momentum_signal'],
            latest['kalman_signal'],
            latest['prediction_signal']
        ]
        
        # Count aligned signals (same sign as combined_signal)
        aligned_count = sum(1 for s in component_signals if np.sign(s) == np.sign(combined_signal) and s != 0)
        alignment_pct = aligned_count / sum(1 for s in component_signals if s != 0) if sum(1 for s in component_signals if s != 0) > 0 else 0
        
        # Create trading signal dictionary
        signals = {
            'time': latest['timestamp'] if 'timestamp' in latest else pd.Timestamp.now(),
            'price': latest['price'],
            'combined_signal': combined_signal,
            'action': action,
            'signal_strength': signal_strength,
            'buy_weight': buy_weight,
            'sell_weight': sell_weight,
            'confidence': alignment_pct,
            'component_signals': {
                'order_imbalance': latest['order_imbalance'],
                'price_velocity': latest['price_velocity'],
                'arb_strength': latest['arb_strength'],
                'mean_reversion': latest['mean_reversion_signal'],
                'momentum': latest['momentum_signal'],
                'kalman': latest['kalman_signal'],
                'prediction': latest['prediction_signal']
            },
            'vwap': latest['vwap']
        }
        
        return signals


# Demo usage
if __name__ == "__main__":
    # Create sample tick data
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create synthetic tick data
    np.random.seed(42)
    n_ticks = 1000
    
    # Generate timestamps with random microsecond intervals
    base_time = datetime.now() - timedelta(hours=1)
    timestamps = [base_time + timedelta(microseconds=int(i * 1e5 + np.random.randint(0, 1e5))) 
                 for i in range(n_ticks)]
    
    # Generate price series with random walk
    price_changes = np.random.normal(0, 0.01, n_ticks)
    price_changes[0] = 0
    prices = 100 + np.cumsum(price_changes)
    
    # Generate volumes with occasional spikes
    volumes = np.random.lognormal(3, 1, n_ticks)
    # Add occasional volume spikes
    volume_spikes = np.random.randint(0, n_ticks, 20)
    volumes[volume_spikes] *= 5
    
    # Create DataFrame
    tick_data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes
    })
    
    # Initialize HFT analyzer
    hft = HFTAnalyzer()
    
    # Process tick data
    print("Processing tick data...")
    processed_data = hft.process_tick_data(tick_data)
    
    # Simulate order book
    print("\nSimulating order book...")
    current_price = processed_data['price'].iloc[-1]
    bid_book, ask_book = hft.simulate_order_book(current_price)
    print(f"Bid book: {bid_book}")
    print(f"Ask book: {ask_book}")
    
    # Analyze order flow
    print("\nAnalyzing order flow...")
    flow_analysis = hft.analyze_order_flow(processed_data)
    print(f"Directional bias: {flow_analysis['directional_bias']:.4f} ({flow_analysis['classification']})")
    print(f"Confidence: {flow_analysis['confidence']:.2f}")
    
    # Get optimal execution strategy
    print("\nCalculating optimal execution strategy...")
    strategy = hft.get_optimal_execution_strategy('buy', 5000)
    print(f"TWAP quantity per minute: {strategy['twap_quantity']:.2f}")
    print(f"Estimated market impact: {strategy['estimated_total_impact']:.4f}")
    
    # Get HFT trading signals
    print("\nGenerating HFT trading signals...")
    signals = hft.get_hft_trading_signals(processed_data.tail(100))
    print(f"Action: {signals['action']}")
    print(f"Signal strength: {signals['signal_strength']:.4f}")
    print(f"Confidence: {signals['confidence']:.2f}")
    print(f"Buy weight: {signals['buy_weight']:.2f}")
    print(f"Sell weight: {signals['sell_weight']:.2f}")