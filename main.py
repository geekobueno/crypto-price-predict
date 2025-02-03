from src.data.kaggle_loader import KaggleDataLoader
from src.data.data_processor import DataProcessor
from src.features.technical_indicators import TechnicalIndicators
from src.features.target_generator import TargetGenerator
from src.visualization.visualizer import Visualizer
from config.config import Config
import os
import pandas as pd

def main():
    # Initialize components
    kaggle_loader = KaggleDataLoader(Config.KAGGLE_DATASET, Config.RAW_DATA_PATH)
    data_processor = DataProcessor()
    visualizer = Visualizer(Config.VISUALIZATIONS_PATH)
    
    try:
        # Download and load data
        kaggle_loader.download_dataset()
        df_all = kaggle_loader.load_data_from_csv()
        
        if df_all is None:
            raise ValueError("Failed to load dataset")
        
        # Get available symbols
        available_symbols = sorted(df_all['symbol'].unique())
        
        print("\nAvailable cryptocurrencies:")
        for idx, symbol in enumerate(available_symbols, start=1):
            print(f"{idx}. {symbol}")
        
        # Get user selection
        selection = input("\nEnter the number(s) of the cryptocurrency to analyze (comma-separated): ")
        selected_indices = [int(i.strip())-1 for i in selection.split(',') if i.strip().isdigit()]
        selected_symbols = [available_symbols[i] for i in selected_indices if 0 <= i < len(available_symbols)]
        
        if not selected_symbols:
            print("No valid selection made. Exiting...")
            return
        
        # Process each selected cryptocurrency
        for symbol in selected_symbols:
            print(f"\nProcessing {symbol}...")
            
            df = df_all[df_all['symbol'] == symbol].copy()
            
            # Add technical indicators
            df = TechnicalIndicators.add_all_indicators(df)
            
            # Add target variables
            df = TargetGenerator.add_targets(df, Config.PREDICTION_HORIZONS)
            
            # Scale and clean data
            df = data_processor.scale_features(df, symbol)
            df = data_processor.clean_data(df)
            
            # Create processed data directory if it doesn't exist
            os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
            
            # Save processed data
            output_file = os.path.join(Config.PROCESSED_DATA_PATH, f"{symbol}_processed.csv")
            df.to_csv(output_file, index=False)
            print(f"Processed data saved to: {output_file}")
            
            # Generate visualizations
            visualizer.generate_summary_stats(df, symbol)
            
            print(f"Successfully processed {symbol}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()

