import os

class Config:
    # Project paths
    BASE_PATH = "crypto_data"
    RAW_DATA_PATH = os.path.join(BASE_PATH, "raw")
    PROCESSED_DATA_PATH = os.path.join(BASE_PATH, "processed")
    MODELS_PATH = os.path.join(BASE_PATH, "models")
    VISUALIZATIONS_PATH = os.path.join(BASE_PATH, "visualizations")
    
    # Kaggle settings
    KAGGLE_DATASET = "surajshah/crypto-data-top-tokens"
    
    # Cryptocurrency settings
    SYMBOLS = ['BTC', 'ETH', 'BNB']
    
    # Feature settings
    PREDICTION_HORIZONS = [1, 3, 7]
    MA_WINDOWS = [7, 14, 30, 50, 200]
    VOLATILITY_WINDOWS = [7, 14, 30]
