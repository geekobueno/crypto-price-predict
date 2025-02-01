import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Visualizer:
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
    
    def generate_summary_stats(self, df: pd.DataFrame, symbol: str) -> None:
        """Generate and print summary statistics"""
        print(f"\nSummary Statistics for {symbol}:")
        print(f"Date Range: {df['dates'].min()} to {df['dates'].max()}")
        print(f"Total Trading Days: {len(df)}")
        print("\nPrice Statistics:")
        print(df[['close', 'daily_return', 'volatility_30d']].describe())
        
        # Save summary plot
        plt.figure(figsize=(15, 10))
        
        # Price and volume subplot
        plt.subplot(2, 1, 1)
        plt.plot(df['dates'], df['close'], label='Close Price')
        plt.plot(df['dates'], df['MA_50'], label='50-day MA')
        plt.title(f'{symbol} Price History')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(df['dates'], df['volume'], label='Volume')
        plt.title(f'{symbol} Trading Volume')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f"{symbol}_summary.png"))
        plt.close()
