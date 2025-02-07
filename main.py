from src.data.kaggle_loader import KaggleDataLoader
from src.processor.crypto_data_processor import DataProcessor
from src.visualizer.history_visualizer import HistoryVisualizer
from src.analyzer.technical_indicators import TechnicalAnalysis
from src.visualizer.technical_visualizer import TechnicalVisualizer
from src.visualizer.wikipedia_visualizer import WikipediaVisualizer
from src.data.wikipedia_edit import get_all_wikipedia_edits
from src.utils.groq_caller import get_crypto_name_from_groq
from src.data.google_trends_fetcher import GoogleTrendsFetcher
from src.visualizer.google_trends_visualizer import GoogleTrendsVisualizer
from config.config import Config
import os
import pandas as pd
from typing import List, Dict
import logging
from datetime import datetime, timedelta

def setup_logging() -> None:
    """Configure logging for the application."""
    os.makedirs(Config.LOG_PATH, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Config.LOG_PATH, f'crypto_analysis_{datetime.now().strftime("%Y%m%d")}.log')),
            logging.StreamHandler()
        ]
    )

def create_directories() -> None:
    """Create necessary directories for the application."""
    directories = [
        Config.RAW_DATA_PATH,
        Config.PROCESSED_DATA_PATH,
        Config.VISUALIZATIONS_PATH,
        Config.LOG_PATH
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def get_user_selection(available_symbols: List[str]) -> List[str]:
    """Get user selection of cryptocurrencies to analyze."""
    print("\nAvailable cryptocurrencies:")
    for idx, symbol in enumerate(available_symbols, start=1):
        print(f"{idx}. {symbol}")
    
    while True:
        try:
            selection = input("\nEnter the number(s) of the cryptocurrency to analyze (comma-separated): ")
            selected_indices = [int(i.strip())-1 for i in selection.split(',') if i.strip().isdigit()]
            selected_symbols = [available_symbols[i] for i in selected_indices if 0 <= i < len(available_symbols)]
            
            if not selected_symbols:
                print("No valid selection made. Please try again.")
                continue
                
            return selected_symbols
            
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid numbers separated by commas.")

def analyze_social_data(crypto_name: str, wiki_visualizer: WikipediaVisualizer) -> None:
    """Analyze Wikipedia edits and Google Trends data."""
    # Fetch and analyze Wikipedia data
    wiki_edit_file = get_all_wikipedia_edits(crypto_name)
    if wiki_edit_file:
        logging.info(f"Saved Wikipedia edit history to {wiki_edit_file}")
        wiki_visualizer.create_all_visualizations(wiki_edit_file, crypto_name)
    
    # Initialize Google Trends components
    trends_fetcher = GoogleTrendsFetcher()
    trends_visualizer = GoogleTrendsVisualizer(Config.VISUALIZATIONS_PATH)
    
    # Set date range for trends analysis
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Fetch Google Trends data
    trends_df = trends_fetcher.fetch_trends(crypto_name, start_date, end_date)
    related_queries = trends_fetcher.fetch_related_queries(crypto_name, start_date, end_date)
    
    if trends_df is not None:
        trends_file = trends_fetcher.save_trends_data(
            trends_df,
            crypto_name,
            Config.PROCESSED_DATA_PATH
        )
        logging.info(f"Saved Google Trends data to {trends_file}")
        trends_visualizer.create_all_visualizations(trends_df, related_queries, crypto_name)

def analyze_market_data(
    symbol: str,
    df: pd.DataFrame,
    history_visualizer: HistoryVisualizer,
    technical_analyzer: TechnicalAnalysis,
    technical_visualizer: TechnicalVisualizer
) -> None:
    """Analyze market data and create visualizations."""
    # Create price history visualizations
    logging.info(f"Creating price history visualizations for {symbol}...")
    history_visualizer.create_all_visualizations(df, symbol)
    
    # Calculate and visualize technical indicators
    logging.info(f"Calculating technical indicators for {symbol}...")
    df_with_indicators = technical_analyzer.calculate_all_indicators(df)
    
    logging.info(f"Creating technical analysis plots for {symbol}...")
    technical_visualizer.create_all_technical_plots(df_with_indicators, symbol)
    
    # Save data with indicators
    indicator_file = os.path.join(Config.PROCESSED_DATA_PATH, f"{symbol}_with_indicators.csv")
    df_with_indicators.to_csv(indicator_file, index=False)
    logging.info(f"Technical indicators data saved to: {indicator_file}")

def process_single_crypto(
    symbol: str,
    df: pd.DataFrame,
    history_visualizer: HistoryVisualizer,
    technical_analyzer: TechnicalAnalysis,
    technical_visualizer: TechnicalVisualizer,
    wiki_visualizer: WikipediaVisualizer
) -> None:
    """Process and analyze a single cryptocurrency."""
    logging.info(f"\nProcessing {symbol}...")
    
    try:
        # Get full cryptocurrency name
        crypto_name = get_crypto_name_from_groq(symbol)
        if crypto_name:
            logging.info(f"Cryptocurrency Name: {crypto_name}")
            
            # Analyze social data (Wikipedia and Google Trends)
            analyze_social_data(crypto_name, wiki_visualizer)
        else:
            logging.warning(f"Could not fetch crypto name for {symbol}. Skipping social data analysis.")
        
        # Analyze market data
        analyze_market_data(
            symbol,
            df,
            history_visualizer,
            technical_analyzer,
            technical_visualizer
        )
        
    except Exception as e:
        logging.error(f"Error processing {symbol}: {e}")
        raise

def main():
    """Main function to run the cryptocurrency analysis pipeline."""
    try:
        # Initial setup
        setup_logging()
        create_directories()
        logging.info("Starting cryptocurrency analysis...")
        
        # Initialize components
        kaggle_loader = KaggleDataLoader(Config.KAGGLE_DATASET, Config.RAW_DATA_PATH)
        data_processor = DataProcessor(Config.PROCESSED_DATA_PATH)
        history_visualizer = HistoryVisualizer(Config.VISUALIZATIONS_PATH)
        technical_visualizer = TechnicalVisualizer(Config.VISUALIZATIONS_PATH)
        technical_analyzer = TechnicalAnalysis()
        wiki_visualizer = WikipediaVisualizer(Config.VISUALIZATIONS_PATH)
        
        # Download and load data
        logging.info("Downloading dataset from Kaggle...")
        kaggle_loader.download_dataset()
        
        logging.info("Loading dataset...")
        df_all = kaggle_loader.load_data_from_csv()
        
        if df_all is None:
            raise ValueError("Failed to load dataset")
        
        # Get user selection and process data
        available_symbols = sorted(df_all['symbol'].unique())
        selected_symbols = get_user_selection(available_symbols)
        
        extracted_data, saved_files = data_processor.process_data(df_all, selected_symbols)
        
        logging.info("\nProcessed data saved to:")
        for symbol, filepath in saved_files.items():
            logging.info(f"{symbol}: {filepath}")
        
        # Process each selected cryptocurrency
        for symbol, df in extracted_data.items():
            process_single_crypto(
                symbol,
                df,
                history_visualizer,
                technical_analyzer,
                technical_visualizer,
                wiki_visualizer
            )
        
        logging.info("Analysis completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()