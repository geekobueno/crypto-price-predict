import pandas as pd
import numpy as np

class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataframe"""
        tech_df = df.copy()
        
        # Price-based indicators
        tech_df['daily_return'] = tech_df['close'].pct_change()
        tech_df['log_return'] = np.log(tech_df['close']/tech_df['close'].shift(1))
        
        # Moving averages
        for window in [7, 14, 30, 50, 200]:
            tech_df[f'MA_{window}'] = tech_df['close'].rolling(window=window).mean()
            tech_df[f'volume_MA_{window}'] = tech_df['volume'].rolling(window=window).mean()
        
        # Exponential moving averages
        for window in [12, 26]:
            tech_df[f'EMA_{window}'] = tech_df['close'].ewm(span=window).mean()
        
        # MACD
        tech_df['MACD'] = tech_df['EMA_12'] - tech_df['EMA_26']
        tech_df['MACD_signal'] = tech_df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        for window in [20]:
            mid_band = tech_df['close'].rolling(window=window).mean()
            std_dev = tech_df['close'].rolling(window=window).std()
            tech_df[f'BB_upper_{window}'] = mid_band + (std_dev * 2)
            tech_df[f'BB_lower_{window}'] = mid_band - (std_dev * 2)
        
        # RSI
        delta = tech_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        tech_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        for window in [7, 14, 30]:
            tech_df[f'volatility_{window}d'] = tech_df['daily_return'].rolling(window=window).std()
        
        # Price momentum
        for window in [7, 14, 30]:
            tech_df[f'momentum_{window}d'] = tech_df['close'].pct_change(window)
        
        return tech_df
