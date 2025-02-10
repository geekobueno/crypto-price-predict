from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data.kaggle_loader import KaggleDataLoader
from src.processor.crypto_data_processor import DataProcessor
from src.data.wikipedia_edit import get_all_wikipedia_edits
from src.utils.groq_caller import get_crypto_name_from_groq
from src.data.google_trend import GoogleTrendsFetcher
from config.config import Config

class CryptoAnalysisPipeline:
    def __init__(self):
        """Initialize the analysis pipeline with config settings."""
        self.config = Config()
        self.setup_logging()
        self.create_directories()
        self.initialize_components()

    def setup_logging(self) -> None:
        """Configure logging with settings from config."""
        log_file = Path(self.config.LOG_PATH) / f'crypto_analysis_{datetime.now():%Y%m%d}.log'
        os.makedirs(self.config.LOG_PATH, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def create_directories(self) -> None:
        """Create all required directories from config."""
        directories = [
            self.config.RAW_DATA_PATH,
            self.config.PROCESSED_DATA_PATH,
            self.config.VISUALIZATIONS_PATH,
            self.config.WIKIPEDIA_PATH,
            self.config.GOOGLE_PATH,
            self.config.LOG_PATH
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {directory}")

    def initialize_components(self) -> None:
        """Initialize all required components."""
        self.kaggle_loader = KaggleDataLoader(
            self.config.KAGGLE_DATASET, 
            self.config.RAW_DATA_PATH
        )
        self.data_processor = DataProcessor(self.config.PROCESSED_DATA_PATH)
        self.google_trends_fetcher = GoogleTrendsFetcher()

    def get_user_selection(self, available_symbols: List[str]) -> List[str]:
        """Get user input for cryptocurrency selection."""
        print("\nAvailable cryptocurrencies:")
        for idx, symbol in enumerate(available_symbols, start=1):
            print(f"{idx}. {symbol}")
        
        while True:
            try:
                selection = input("\nEnter the number(s) of the cryptocurrency to analyze (comma-separated): ")
                selected_indices = [int(i.strip()) - 1 for i in selection.split(',') if i.strip().isdigit()]
                selected_symbols = [available_symbols[i] for i in selected_indices if 0 <= i < len(available_symbols)]
                
                if selected_symbols:
                    return selected_symbols
                print("No valid selection made. Please try again.")
                
            except (ValueError, IndexError):
                print("Invalid input. Please enter valid numbers separated by commas.")

    def process_social_data(self, crypto_name: str) -> None:
        """Process social media data for a cryptocurrency."""
        try:
            # Wikipedia data
            wiki_edit_file = get_all_wikipedia_edits(crypto_name, self.config.WIKIPEDIA_PATH)
            if wiki_edit_file:
                logging.info(f"Saved Wikipedia edit history to {wiki_edit_file}")

            # Google Trends data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
            
            trends_df = self.google_trends_fetcher.fetch_trends(
                crypto_name, 
                start_date, 
                end_date
            )
            
            if trends_df is not None:
                trends_file = self.google_trends_fetcher.save_trends_data(
                    trends_df,
                    crypto_name,
                    self.config.GOOGLE_PATH
                )
                logging.info(f"Saved Google Trends data to {trends_file}")
                
        except Exception as e:
            logging.error(f"Error processing social data for {crypto_name}: {e}")

    def process_cryptocurrency(self, symbol: str, df: pd.DataFrame) -> None:
        """Process all data for a single cryptocurrency."""
        logging.info(f"\nProcessing {symbol}...")
        
        try:
            # Get full name for social media analysis
            crypto_name = get_crypto_name_from_groq(symbol)
            if crypto_name:
                logging.info(f"Cryptocurrency Name: {crypto_name}")
                self.process_social_data(crypto_name)
            else:
                logging.warning(f"Could not fetch crypto name for {symbol}")

            # Save processed market data
            output_file = Path(self.config.PROCESSED_DATA_PATH) / f"{symbol.lower()}_processed.csv"
            df.to_csv(output_file, index=False)
            logging.info(f"Saved processed market data to {output_file}")
            
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
            raise

    def run(self):
        """Execute the complete analysis pipeline."""
        try:
            logging.info("Starting cryptocurrency analysis pipeline...")
            
            # Download and load data
            self.kaggle_loader.download_dataset()
            df_all = self.kaggle_loader.load_data_from_csv()
            
            if df_all is None:
                raise ValueError("Failed to load dataset")
            
            # Get user selection
            available_symbols = sorted(df_all['symbol'].unique())
            selected_symbols = self.get_user_selection(available_symbols)
            
            # Process selected cryptocurrencies
            processed_data, saved_files = self.data_processor.process_data(
                df_all, 
                selected_symbols
            )
            
            logging.info("\nProcessed data files:")
            for symbol, filepath in saved_files.items():
                logging.info(f"{symbol}: {filepath}")
            
            # Process each cryptocurrency
            for symbol, df in processed_data.items():
                self.process_cryptocurrency(symbol, df)
            
            logging.info("Analysis pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Pipeline error: {e}")
            raise

def main():
    pipeline = CryptoAnalysisPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()