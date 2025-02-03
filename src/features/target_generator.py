import pandas as pd
import numpy as np
from typing import List

class TargetGenerator:
    @staticmethod
    def add_targets(df: pd.DataFrame, prediction_horizons: List[int]) -> pd.DataFrame:
        """Generate target variables for different prediction horizons"""
        df = df.copy()
        
        for horizon in prediction_horizons:
            # Future price
            df[f'future_price_{horizon}d'] = df.groupby('symbol')['close'].transform(
                lambda x: x.shift(-horizon)
            )
            
            # Return
            df[f'return_{horizon}d'] = (
                (df[f'future_price_{horizon}d'] - df['close']) / df['close']
            )
            
            # Binary direction (1 if price goes up, 0 if down)
            df[f'direction_{horizon}d'] = (
                df[f'future_price_{horizon}d'] > df['close']
            ).astype(int)
            
            # Volatility target
            df[f'volatility_{horizon}d'] = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window=horizon).std() / x.rolling(window=horizon).mean()
            )
        
        return df
