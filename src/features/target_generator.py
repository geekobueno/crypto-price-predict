import pandas as pd
from typing import List

class TargetGenerator:
    @staticmethod
    def add_targets(df: pd.DataFrame, horizons: List[int] = [1, 3, 7]) -> pd.DataFrame:
        """Add target variables for different prediction horizons"""
        target_df = df.copy()
        
        for horizon in horizons:
            # Price change
            target_df[f'target_return_{horizon}d'] = target_df['close'].pct_change(horizon).shift(-horizon)
            
            # Binary direction
            target_df[f'target_direction_{horizon}d'] = (target_df[f'target_return_{horizon}d'] > 0).astype(int)
            
            # Volatility target
            target_df[f'target_volatility_{horizon}d'] = target_df['close'].rolling(window=horizon).std().shift(-horizon)
        
        return target_df
