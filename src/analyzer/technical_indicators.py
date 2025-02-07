import pandas as pd
import numpy as np

class TechnicalAnalysis:
    @staticmethod
    def calculate_sma(data, period=20):
        """Calculate Simple Moving Average"""
        return data['close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data, period=20):
        """Calculate Exponential Moving Average"""
        return data['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data, period=14):
        """Calculate Relative Strength Index"""
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        # Calculate EMAs
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        macd_histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': macd_histogram
        })
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'Middle': sma,
            'Upper': upper_band,
            'Lower': lower_band
        })
    
    @staticmethod
    def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        # Calculate %K
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'K': k,
            'D': d
        })

    def calculate_all_indicators(self, data):
        """Calculate all technical indicators"""
        results = data.copy()
        
        # Add all indicators
        results['SMA_20'] = self.calculate_sma(data, 20)
        results['SMA_50'] = self.calculate_sma(data, 50)
        results['EMA_20'] = self.calculate_ema(data, 20)
        results['RSI'] = self.calculate_rsi(data)
        
        # Add MACD
        macd_data = self.calculate_macd(data)
        results['MACD'] = macd_data['MACD']
        results['MACD_Signal'] = macd_data['Signal']
        results['MACD_Histogram'] = macd_data['Histogram']
        
        # Add Bollinger Bands
        bb_data = self.calculate_bollinger_bands(data)
        results['BB_Middle'] = bb_data['Middle']
        results['BB_Upper'] = bb_data['Upper']
        results['BB_Lower'] = bb_data['Lower']
        
        # Add Stochastic Oscillator
        stoch_data = self.calculate_stochastic_oscillator(data)
        results['Stoch_K'] = stoch_data['K']
        results['Stoch_D'] = stoch_data['D']
        
        return results