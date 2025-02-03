class Config:
    # Kaggle dataset configuration
    KAGGLE_DATASET = "ayushkhaire/top-1000-cryptos-historical"
    
    # Directory paths
    RAW_DATA_PATH = "crypto_data/raw"
    PROCESSED_DATA_PATH = "crypto_data/processed"
    VISUALIZATIONS_PATH = "visualizations"
    
    # Technical analysis parameters
    MOVING_AVERAGES = [7, 14, 30, 50, 200]  # Days for different MAs
    RSI_PERIOD = 14
    MACD_PARAMS = {
        'fast': 12,
        'slow': 26,
        'signal': 9
    }
    
    # Target generation parameters
    PREDICTION_HORIZONS = [1, 3, 7, 14, 30]  # Days to predict ahead
    
    # Data processing parameters
    MINIMUM_RECORDS = 250  # Minimum number of records needed for processing
    FEATURES_TO_SCALE = ['open', 'high', 'low', 'close', 'volume']
