import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List

class DataProcessor:
    def __init__(self):
        self.scalers = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe by removing invalid entries and handling missing values"""
        df = df.copy()
        
        # Remove rows with invalid prices
        df = df[df['close'] > 0]
        df = df[df['volume'] > 0]
        
        # Forward fill missing values for technical indicators
        technical_columns = [col for col in df.columns if col not in 
                           ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
        df[technical_columns] = df[technical_columns].fillna(method='ffill')
        
        # Remove rows with any remaining NaN values
        df = df.dropna()
        
        return df
    
    def scale_features(self, df: pd.DataFrame, features_to_scale: List[str]) -> pd.DataFrame:
        """Scale numerical features using StandardScaler"""
        df = df.copy()
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            
            # Create or get scaler for this symbol
            if symbol not in self.scalers:
                self.scalers[symbol] = {}
            
            # Scale each feature
            for feature in features_to_scale:
                if feature in df.columns:
                    if feature not in self.scalers[symbol]:
                        self.scalers[symbol][feature] = StandardScaler()
                        
                    values = symbol_data[feature].values.reshape(-1, 1)
                    scaled_values = self.scalers[symbol][feature].fit_transform(values)
                    df.loc[symbol_data.index, f'{feature}_scaled'] = scaled_values
        
        return df
