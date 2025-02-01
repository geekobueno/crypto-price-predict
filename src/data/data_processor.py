import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Optional

class DataProcessor:
    def __init__(self):
        self.scalers: Dict[str, MinMaxScaler] = {}
    
    def scale_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Scale numerical features to [0,1] range"""
        scaled_df = df.copy()
        
        # Columns to exclude from scaling
        exclude_columns = ['dates', 'symbol'] + [col for col in df.columns if 'target' in col]
        
        # Columns to scale
        scale_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Initialize scaler for this symbol if not exists
        if symbol not in self.scalers:
            self.scalers[symbol] = MinMaxScaler()
            
        # Scale features
        scaled_data = self.scalers[symbol].fit_transform(df[scale_columns])
        scaled_df[scale_columns] = scaled_data
        
        return scaled_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe by removing NaN values and duplicates"""
        cleaned_df = df.copy()
        cleaned_df.dropna(inplace=True)
        cleaned_df.drop_duplicates(subset=['dates'], inplace=True)
        return cleaned_df
