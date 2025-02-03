import pandas as pd
import numpy as np
from typing import List

class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, ma_periods: List[int], rsi_period: int, 
                          macd_params: dict) -> pd.DataFrame:
        """Add all technical indicators to the dataframe"""
        df = df.copy()
        
        # Ensure data is sorted by date
        df = df.sort_values('date')
        
        # Add Moving Averages
        for period in ma_periods:
            df[f'ma_{period}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window=period, min_periods=1).mean()
            )
        
        # Add RSI
        delta = df.groupby('symbol')['close'].transform('diff')
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = df.groupby('symbol').transform(
            lambda x: x.rolling(window=rsi_period, min_periods=1).mean()
        )(gain)
        avg_loss = df.groupby('symbol').transform(
            lambda x: x.rolling(window=rsi_period, min_periods=1).mean()
        )(loss)
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Add MACD
        exp1 = df.groupby('symbol')['close'].transform(
            lambda x: x.ewm(span=macd_params['fast'], adjust=False).mean()
        )
        exp2 = df.groupby('symbol')['close'].transform(
            lambda x: x.ewm(span=macd_params['slow'], adjust=False).mean()
        )
        macd = exp1 - exp2
        signal = macd.transform(
            lambda x: x.ewm(span=macd_params['signal'], adjust=False).mean()
        )
        
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = macd - signal
        
        # Add Bollinger Bands
        df['bb_middle'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )
        df['bb_std'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=20, min_periods=1).std()
        )
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Add momentum indicators
        df['momentum'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(periods=10)
        )
        
        return df
