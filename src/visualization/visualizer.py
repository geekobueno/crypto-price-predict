import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class Visualizer:
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
    
    def generate_summary_stats(self, df: pd.DataFrame, symbol: str) -> None:
        """Generate summary visualizations for a cryptocurrency"""
        # Create a figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Price and Volume Plot
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(df['date'], df['close'], label='Close Price')
        ax1_twin = ax1.twinx()
        ax1_twin.bar(df['date'], df['volume'], alpha=0.3, label='Volume')
        ax1.set_title(f'{symbol} Price and Volume')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1_twin.set_ylabel('Volume')
        
        # 2. Technical Indicators
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(df['date'], df['close'], label='Close')
        ax2.plot(df['date'], df['bb_upper'], label='BB Upper')
        ax2.plot(df['date'], df['bb_lower'], label='BB Lower')
        ax2.set_title(f'{symbol} Bollinger Bands')
        ax2.legend()
        
        # 3. RSI Plot
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(df['date'], df['rsi'])
        ax3.axhline(y=70, color='r', linestyle='--')
        ax3.axhline(y=30, color='g', linestyle='--')
        ax3.set_title(f'{symbol} RSI')
        
        # 4. MACD Plot
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(df['date'], df['macd'], label='MACD')
        ax4.plot(df['date'], df['macd_signal'], label='Signal')
        ax4.bar(df['date'], df['macd_hist'], alpha=0.3, label='Histogram')
        ax4.set_title(f'{symbol} MACD')
        ax4.legend()
        
        # 5. Returns Distribution
        ax5 = plt.subplot(3, 2, 5)
        returns = df['close'].pct_change().dropna()
        sns.histplot(returns, bins=50, ax=ax5)
        ax5.set_title(f'{symbol} Daily Returns Distribution')
        
        # 6. Moving Averages
        ax6 = plt.subplot(3, 2, 6)
        ax6.plot(df['date'], df['close'], label='Close')
        for ma in [col for col in df.columns if col.startswith('ma_')]:
            ax6.plot(df['date'], df[ma], label=ma.upper())
        ax6.set_title(f'{symbol} Moving Averages')
        ax6.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{symbol}_summary.png'))
        plt.close()
