import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
import logging

class GoogleTrendsVisualizer:
    def __init__(self, output_path: str):
        """
        Initialize the Google Trends visualizer.
        
        Args:
            output_path (str): Directory path where visualizations will be saved
        """
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
    def plot_trends_over_time(self, df: pd.DataFrame, crypto_name: str):
        """
        Create a line plot showing search interest over time for different search terms.
        
        Args:
            df (pd.DataFrame): DataFrame containing trends data
            crypto_name (str): Name of the cryptocurrency
        """
        plt.figure(figsize=(15, 7))
        
        # Plot each search term
        for column in df.columns:
            if column != 'timestamp':
                plt.plot(df['timestamp'], df[column], label=column, linewidth=2)
        
        plt.title(f'Google Search Interest Over Time - {crypto_name}')
        plt.xlabel('Date')
        plt.ylabel('Search Interest')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{crypto_name.lower()}_trends_over_time.png'))
        plt.close()
        
    def plot_trends_heatmap(self, df: pd.DataFrame, crypto_name: str):
        """
        Create a heatmap showing search interest by month and year.
        
        Args:
            df (pd.DataFrame): DataFrame containing trends data
            crypto_name (str): Name of the cryptocurrency
        """
        plt.figure(figsize=(15, 8))
        
        # Create pivot table for the main search term
        main_term_data = df[[crypto_name, 'timestamp']].copy()
        main_term_data['year'] = main_term_data['timestamp'].dt.year
        main_term_data['month'] = main_term_data['timestamp'].dt.month
        
        monthly_activity = main_term_data.pivot_table(
            index='year',
            columns='month',
            values=crypto_name,
            aggfunc='mean'
        )
        
        sns.heatmap(monthly_activity, cmap='YlOrRd', annot=True, fmt='.0f')
        plt.title(f'Monthly Search Interest for {crypto_name}')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{crypto_name.lower()}_trends_heatmap.png'))
        plt.close()
        
    def plot_related_queries(self, related_queries: Dict, crypto_name: str):
        """
        Create bar plots for top rising and top related queries.
        
        Args:
            related_queries (Dict): Dictionary containing related queries data
            crypto_name (str): Name of the cryptocurrency
        """
        if not related_queries or ('rising' not in related_queries and 'top' not in related_queries):
            logging.warning(f"No related queries data available for {crypto_name}")
            return
            
        # Plot rising queries if available
        if 'rising' in related_queries and not related_queries['rising'].empty:
            plt.figure(figsize=(12, 6))
            df_rising = related_queries['rising'].head(10)
            
            sns.barplot(x='value', y='query', data=df_rising)
            plt.title(f'Top Rising Related Queries - {crypto_name}')
            plt.xlabel('Rising Interest')
            plt.ylabel('Query')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, f'{crypto_name.lower()}_rising_queries.png'))
            plt.close()
            
        # Plot top queries if available
        if 'top' in related_queries and not related_queries['top'].empty:
            plt.figure(figsize=(12, 6))
            df_top = related_queries['top'].head(10)
            
            sns.barplot(x='value', y='query', data=df_top)
            plt.title(f'Top Related Queries - {crypto_name}')
            plt.xlabel('Search Interest')
            plt.ylabel('Query')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, f'{crypto_name.lower()}_top_queries.png'))
            plt.close()
            
    def create_all_visualizations(self, trends_df: pd.DataFrame, related_queries: Dict, crypto_name: str):
        """
        Create all available visualizations for Google Trends data.
        
        Args:
            trends_df (pd.DataFrame): DataFrame containing trends data
            related_queries (Dict): Dictionary containing related queries data
            crypto_name (str): Name of the cryptocurrency
        """
        try:
            logging.info(f"Creating Google Trends visualizations for {crypto_name}...")
            
            if trends_df is not None and not trends_df.empty:
                self.plot_trends_over_time(trends_df, crypto_name)
                self.plot_trends_heatmap(trends_df, crypto_name)
            
            if related_queries is not None:
                self.plot_related_queries(related_queries, crypto_name)
                
            logging.info(f"Google Trends visualizations saved to {self.output_path}")
            
        except Exception as e:
            logging.error(f"Error creating Google Trends visualizations: {e}")