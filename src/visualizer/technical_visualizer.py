import matplotlib.pyplot as plt
import os

class TechnicalVisualizer:
    def __init__(self, output_path="visualizations"):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        plt.style.use('ggplot')
    
    def plot_moving_averages(self, df, symbol):
        """Plot price with SMA and EMA"""
        plt.figure(figsize=(15, 7))
        
        plt.plot(df['dates'], df['close'], label='Price', alpha=0.7)
        plt.plot(df['dates'], df['SMA_20'], label='SMA 20', alpha=0.7)
        plt.plot(df['dates'], df['SMA_50'], label='SMA 50', alpha=0.7)
        plt.plot(df['dates'], df['EMA_20'], label='EMA 20', alpha=0.7)
        
        plt.title(f'{symbol} Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        filename = f"{symbol}_moving_averages.png"
        plt.savefig(os.path.join(self.output_path, filename), bbox_inches='tight')
        plt.close()
    
    def plot_rsi(self, df, symbol):
        """Plot RSI indicator"""
        plt.figure(figsize=(15, 7))
        
        plt.plot(df['dates'], df['RSI'], label='RSI', color='purple')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        
        plt.title(f'{symbol} Relative Strength Index')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        filename = f"{symbol}_rsi.png"
        plt.savefig(os.path.join(self.output_path, filename), bbox_inches='tight')
        plt.close()
    
    def plot_macd(self, df, symbol):
        """Plot MACD indicator"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        
        # Price plot
        ax1.plot(df['dates'], df['close'], label='Price', alpha=0.7)
        ax1.set_title(f'{symbol} MACD')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # MACD plot
        ax2.plot(df['dates'], df['MACD'], label='MACD')
        ax2.plot(df['dates'], df['MACD_Signal'], label='Signal')
        ax2.bar(df['dates'], df['MACD_Histogram'], label='Histogram', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f"{symbol}_macd.png"
        plt.savefig(os.path.join(self.output_path, filename), bbox_inches='tight')
        plt.close()
    
    def plot_bollinger_bands(self, df, symbol):
        """Plot Bollinger Bands"""
        plt.figure(figsize=(15, 7))
        
        plt.plot(df['dates'], df['close'], label='Price', alpha=0.7)
        plt.plot(df['dates'], df['BB_Upper'], label='Upper Band', alpha=0.7)
        plt.plot(df['dates'], df['BB_Middle'], label='Middle Band', alpha=0.7)
        plt.plot(df['dates'], df['BB_Lower'], label='Lower Band', alpha=0.7)
        plt.fill_between(df['dates'], df['BB_Upper'], df['BB_Lower'], alpha=0.1)
        
        plt.title(f'{symbol} Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        filename = f"{symbol}_bollinger_bands.png"
        plt.savefig(os.path.join(self.output_path, filename), bbox_inches='tight')
        plt.close()
    
    def plot_stochastic(self, df, symbol):
        """Plot Stochastic Oscillator"""
        plt.figure(figsize=(15, 7))
        
        plt.plot(df['dates'], df['Stoch_K'], label='%K')
        plt.plot(df['dates'], df['Stoch_D'], label='%D')
        plt.axhline(y=80, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=20, color='g', linestyle='--', alpha=0.5)
        
        plt.title(f'{symbol} Stochastic Oscillator')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        filename = f"{symbol}_stochastic.png"
        plt.savefig(os.path.join(self.output_path, filename), bbox_inches='tight')
        plt.close()
    
    def create_all_technical_plots(self, df, symbol):
        """Create all technical analysis plots"""
        print(f"\nCreating technical analysis plots for {symbol}...")
        self.plot_moving_averages(df, symbol)
        self.plot_rsi(df, symbol)
        self.plot_macd(df, symbol)
        self.plot_bollinger_bands(df, symbol)
        self.plot_stochastic(df, symbol)
        print(f"Technical analysis plots saved in {self.output_path}")