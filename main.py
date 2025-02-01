from src.data.kaggle_loader import KaggleDataLoader
from src.data.data_processor import DataProcessor
from src.features.technical_indicators import TechnicalIndicators
from src.features.target_generator import TargetGenerator
from src.visualization.visualizer import Visualizer
from config.config import Config
import os

def main():
    # Initialize components
    kaggle_loader = KaggleDataLoader(Config.KAGGLE_DATASET, Config.RAW_DATA_PATH)
    data_processor = DataProcessor()
    visualizer = Visualizer(Config.VISUALIZATIONS_PATH)
    
    try:
        # Download Kaggle dataset
        kaggle_loader.download_dataset()
        
        # Process each cryptocurrency
        for symbol in Config.SYMBOLS:
            print(f"\nProcessing {symbol}...")
            
            # Load data
            df = kaggle_loader.load_crypto_data(symbol)
            if df is None:
                continue
            
            # Add technical indicators
            df = TechnicalIndicators.add_all_indicators(df)
            
            # Add target variables
            df = TargetGenerator.add_targets(df, Config.PREDICTION_HORIZONS)
            
            # Scale and clean data
            df = data_processor.scale_features(df, symbol)
            df = data_processor.clean_data(df)
            
            # Save processed data
            output_file = os.path.join(Config.PROCESSED_DATA_PATH, f"{symbol}_processed.csv")
            df.to_csv(output_file, index=False)
            print(f"Processed data saved to: {output_file}")
            
            # Generate visualizations
            visualizer.generate_summary_stats(df, symbol)
            
            print(f"Successfully processed {symbol}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
