import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime

class HistoryVisualizer:
    def __init__(self, output_path="visualizations"):
        """
        Initialize the Visualizer with an output path for saving plots
        
        Args:
            output_path (str): Directory where visualizations will be saved
        """
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Use a built-in style that's guaranteed to work
        plt.style.use('ggplot')
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (15, 7)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def _prepare_data(self, df):
        """
        Prepare and validate data for visualization
        
        Args:
            df (pandas.DataFrame): Input DataFrame
        
        Returns:
            pandas.DataFrame: Processed DataFrame
        """
        # Make a copy to avoid modifying original data
        df = df.copy()
        
        # Convert dates to datetime if they're not already
        if not pd.api.types.is_datetime64_any_dtype(df['dates']):
            df['dates'] = pd.to_datetime(df['dates'])
        
        # Sort by date
        df = df.sort_values('dates')
        
        # Convert numeric columns to float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def plot_price_history(self, df, symbol):
        """
        Create a line plot of cryptocurrency price history
        
        Args:
            df (pandas.DataFrame): DataFrame containing crypto data
            symbol (str): Symbol of the cryptocurrency
        """
        try:
            df = self._prepare_data(df)
            
            fig, ax = plt.subplots(figsize=(15, 7))
            
            # Plot closing price
            ax.plot(df['dates'], df['close'], label='Close Price', 
                   color='#1f77b4', linewidth=2)
            
            # Add high and low prices as light fill
            ax.fill_between(df['dates'], df['high'], df['low'], 
                          alpha=0.2, color='#1f77b4',
                          label='Price Range')
            
            # Formatting
            ax.set_title(f'{symbol} Price History', pad=20)
            ax.set_xlabel('Date', labelpad=10)
            ax.set_ylabel('Price (USD)', labelpad=10)
            ax.legend(loc='upper left')
            
            # Format date axis
            plt.gcf().autofmt_xdate()  # Angle and align the tick labels
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            filename = f"{symbol}_price_history.png"
            plt.savefig(os.path.join(self.output_path, filename), 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Created price history plot for {symbol}")
            
        except Exception as e:
            print(f"Error creating price history plot for {symbol}: {str(e)}")
            plt.close()  # Clean up in case of error
    
    def plot_volume_analysis(self, df, symbol):
        """
        Create a volume analysis plot with price overlay
        
        Args:
            df (pandas.DataFrame): DataFrame containing crypto data
            symbol (str): Symbol of the cryptocurrency
        """
        try:
            df = self._prepare_data(df)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                          height_ratios=[2, 1], 
                                          sharex=True)
            
            # Price plot on top subplot
            ax1.plot(df['dates'], df['close'], label='Close Price', 
                    color='#1f77b4', linewidth=2)
            ax1.set_title(f'{symbol} Price and Volume Analysis', pad=20)
            ax1.set_ylabel('Price (USD)', labelpad=10)
            ax1.legend(loc='upper left')
            
            # Volume plot on bottom subplot
            # Color volume bars based on price change
            df['price_change'] = df['close'].diff()
            colors = np.where(df['price_change'] >= 0, '#2ecc71', '#e74c3c')
            
            ax2.bar(df['dates'], df['volume'], alpha=0.7, 
                   color=colors, label='Volume')
            ax2.set_ylabel('Volume', labelpad=10)
            ax2.set_xlabel('Date', labelpad=10)
            ax2.legend(loc='upper left')
            
            # Format date axis
            plt.gcf().autofmt_xdate()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            filename = f"{symbol}_volume_analysis.png"
            plt.savefig(os.path.join(self.output_path, filename), 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Created volume analysis plot for {symbol}")
            
        except Exception as e:
            print(f"Error creating volume analysis plot for {symbol}: {str(e)}")
            plt.close()
    
    def plot_daily_returns(self, df, symbol):
        """
        Create a plot showing daily price returns/changes
        
        Args:
            df (pandas.DataFrame): DataFrame containing crypto data
            symbol (str): Symbol of the cryptocurrency
        """
        try:
            df = self._prepare_data(df)
            
            # Calculate daily returns
            df['daily_return'] = df['close'].pct_change() * 100
            
            fig, ax = plt.subplots(figsize=(15, 7))
            
            # Plot returns
            ax.fill_between(df['dates'], df['daily_return'], 0,
                          where=(df['daily_return'] >= 0),
                          color='#2ecc71', alpha=0.5, label='Positive Returns')
            ax.fill_between(df['dates'], df['daily_return'], 0,
                          where=(df['daily_return'] < 0),
                          color='#e74c3c', alpha=0.5, label='Negative Returns')
            
            # Add horizontal lines
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=5, color='gray', linestyle='--', alpha=0.3)
            ax.axhline(y=-5, color='gray', linestyle='--', alpha=0.3)
            
            # Formatting
            ax.set_title(f'{symbol} Daily Returns', pad=20)
            ax.set_xlabel('Date', labelpad=10)
            ax.set_ylabel('Daily Return (%)', labelpad=10)
            ax.legend(loc='upper left')
            
            # Format date axis
            plt.gcf().autofmt_xdate()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            filename = f"{symbol}_daily_returns.png"
            plt.savefig(os.path.join(self.output_path, filename), 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Created daily returns plot for {symbol}")
            
        except Exception as e:
            print(f"Error creating daily returns plot for {symbol}: {str(e)}")
            plt.close()
    
    def create_all_visualizations(self, df, symbol):
        """
        Create all available visualizations for a given cryptocurrency
        
        Args:
            df (pandas.DataFrame): DataFrame containing crypto data
            symbol (str): Symbol of the cryptocurrency
        """
        print(f"\nCreating visualizations for {symbol}...")
        
        try:
            self.plot_price_history(df, symbol)
            self.plot_volume_analysis(df, symbol)
            self.plot_daily_returns(df, symbol)
            print(f"All visualizations saved in {self.output_path}")
            
        except Exception as e:
            print(f"Error creating visualizations for {symbol}: {str(e)}")