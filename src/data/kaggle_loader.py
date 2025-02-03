import os
import kaggle
import pandas as pd
import glob
from typing import Optional

class KaggleDataLoader:
    def __init__(self, dataset_name: str, raw_path: str):
        self.dataset_name = dataset_name
        self.raw_path = raw_path
        os.makedirs(self.raw_path, exist_ok=True)
    
    def download_dataset(self) -> None:
        try:
            existing_csv_files = glob.glob(os.path.join(self.raw_path, "*.csv"))
            if existing_csv_files:
                print(f"Using existing dataset: {existing_csv_files[0]}")
                return
            
            kaggle.api.authenticate()
            print("Downloading dataset...")
            kaggle.api.dataset_download_files(
                self.dataset_name,
                path=self.raw_path,
                unzip=True
            )
            print("Download completed!")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    def load_data_from_csv(self) -> Optional[pd.DataFrame]:
        try:
            csv_files = glob.glob(os.path.join(self.raw_path, "*.csv"))
            if not csv_files:
                return None
            
            df = pd.read_csv(csv_files[0])
            
            # Standardize column names
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Ensure required columns exist
            required_columns = ['symbol', 'dates', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in dataset")
            
            # Convert date and ensure proper sorting
            df['dates'] = pd.to_datetime(df['dates'])
            df = df.sort_values(['symbol', 'dates'])
            
            # Drop any duplicate entries
            df = df.drop_duplicates(subset=['symbol', 'dates'])
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
