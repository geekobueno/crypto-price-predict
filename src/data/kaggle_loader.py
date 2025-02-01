import os
import kaggle
import pandas as pd
from typing import Optional

class KaggleDataLoader:
    def __init__(self, dataset_name: str, raw_path: str):
        """
        Initialize Kaggle data loader
        
        Args:
            dataset_name (str): Name of the Kaggle dataset
            raw_path (str): Path to store raw data
        """
        self.dataset_name = dataset_name
        self.raw_path = raw_path
        os.makedirs(self.raw_path, exist_ok=True)
    
    def download_dataset(self) -> None:
        """Download dataset from Kaggle"""
        try:
            kaggle.api.authenticate()
            print("Kaggle authentication successful!")
            
            print(f"Downloading dataset: {self.dataset_name}")
            kaggle.api.dataset_download_files(
                self.dataset_name,
                path=self.raw_path,
                unzip=True
            )
            print("Download completed successfully!")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    def load_crypto_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data for a specific cryptocurrency"""
        try:
            files = os.listdir(self.raw_path)
            crypto_file = [f for f in files if symbol.lower() in f.lower()]
            
            if not crypto_file:
                print(f"No data file found for {symbol}")
                return None
            
            df = pd.read_csv(os.path.join(self.raw_path, crypto_file[0]))
            
            # Ensure consistent column names
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Convert dates to datetime
            df['dates'] = pd.to_datetime(df['dates'])
            
            # Sort by date
            df.sort_values('dates', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return None
