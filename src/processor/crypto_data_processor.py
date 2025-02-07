import pandas as pd
import os
from datetime import datetime

class DataProcessor:
    def __init__(self, output_path="processed_data"):
        """
        Initialize the DataProcessor with an output path for processed data
        
        Args:
            output_path (str): Directory where processed data will be saved
        """
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
    
    def extract_crypto_data(self, df, symbols):
        """
        Extract data for specified cryptocurrencies from the main dataframe
        
        Args:
            df (pandas.DataFrame): The main dataframe containing all crypto data
            symbols (list): List of cryptocurrency symbols to extract
            
        Returns:
            dict: Dictionary of dataframes for each selected cryptocurrency
        """
        extracted_data = {}
        
        for symbol in symbols:
            # Filter data for the specific symbol
            crypto_df = df[df['symbol'] == symbol].copy()
            
            # Convert dates to datetime if they aren't already
            crypto_df['dates'] = pd.to_datetime(crypto_df['dates'])
            
            # Sort by date
            crypto_df = crypto_df.sort_values('dates')
            
            # Store in dictionary
            extracted_data[symbol] = crypto_df
            
        return extracted_data
    
    def save_crypto_data(self, extracted_data):
        """
        Save extracted data for each cryptocurrency to separate CSV files
        
        Args:
            extracted_data (dict): Dictionary of dataframes for each cryptocurrency
            
        Returns:
            dict: Dictionary mapping symbols to their saved file paths
        """
        saved_files = {}
        
        for symbol, df in extracted_data.items():
            # Create filename
            filename = f"{symbol}_data.csv"
            filepath = os.path.join(self.output_path, filename)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            saved_files[symbol] = filepath
            
        return saved_files
    
    def process_data(self, df, symbols):
        """
        Main processing function that handles extraction and saving
        
        Args:
            df (pandas.DataFrame): The main dataframe containing all crypto data
            symbols (list): List of cryptocurrency symbols to process
            
        Returns:
            tuple: (extracted_data, saved_files) containing the processed dataframes
                  and their file locations
        """
        # Extract data for selected cryptocurrencies
        extracted_data = self.extract_crypto_data(df, symbols)
        
        # Save extracted data to CSV files
        saved_files = self.save_crypto_data(extracted_data)
        
        return extracted_data, saved_files