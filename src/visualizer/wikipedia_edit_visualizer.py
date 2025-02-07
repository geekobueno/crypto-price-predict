import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class WikipediaVisualizer:
    def __init__(self, output_path):
        """
        Initialize the Wikipedia Visualizer.
        
        Args:
            output_path (str): Directory path where visualizations will be saved
        """
        self.output_path = output_path
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
    def load_wiki_data(self, filepath):
        """
        Load and preprocess Wikipedia edit history data.
        
        Args:
            filepath (str): Path to the Wikipedia edit history CSV file
            
        Returns:
            pd.DataFrame: Processed DataFrame with edit history
        """
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
        
    def plot_edits_over_time(self, df, crypto_name):
        """
        Create a line plot showing edit frequency over time.
        
        Args:
            df (pd.DataFrame): DataFrame containing edit history
            crypto_name (str): Name of the cryptocurrency
        """
        plt.figure(figsize=(15, 7))
        
        # Group by date and count edits
        daily_edits = df.groupby(df['timestamp'].dt.date).size()
        
        plt.plot(daily_edits.index, daily_edits.values, linewidth=2)
        plt.title(f'Wikipedia Edit History for {crypto_name}')
        plt.xlabel('Date')
        plt.ylabel('Number of Edits')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{crypto_name}_edits_over_time.png'))
        plt.close()
        
    def plot_editor_distribution(self, df, crypto_name, top_n=10):
        """
        Create a bar plot showing the most active editors.
        
        Args:
            df (pd.DataFrame): DataFrame containing edit history
            crypto_name (str): Name of the cryptocurrency
            top_n (int): Number of top editors to display
        """
        plt.figure(figsize=(12, 6))
        
        # Get top editors by edit count
        top_editors = df['user'].value_counts().head(top_n)
        
        sns.barplot(x=top_editors.values, y=top_editors.index)
        plt.title(f'Top {top_n} Editors of {crypto_name} Wikipedia Page')
        plt.xlabel('Number of Edits')
        plt.ylabel('Editor Username')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{crypto_name}_top_editors.png'))
        plt.close()
        
    def plot_edit_size_distribution(self, df, crypto_name):
        """
        Create a histogram of edit sizes.
        
        Args:
            df (pd.DataFrame): DataFrame containing edit history
            crypto_name (str): Name of the cryptocurrency
        """
        plt.figure(figsize=(10, 6))
        
        # Plot distribution of edit sizes
        sns.histplot(data=df, x='size', bins=50)
        plt.title(f'Distribution of Edit Sizes for {crypto_name}')
        plt.xlabel('Edit Size (bytes)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{crypto_name}_edit_sizes.png'))
        plt.close()
        
    def plot_monthly_activity_heatmap(self, df, crypto_name):
        """
        Create a heatmap showing edit activity by month and year.
        
        Args:
            df (pd.DataFrame): DataFrame containing edit history
            crypto_name (str): Name of the cryptocurrency
        """
        plt.figure(figsize=(15, 8))
        
        # Extract month and year from timestamp
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        
        # Create pivot table for heatmap
        monthly_activity = df.pivot_table(
            index='year',
            columns='month',
            values='timestamp',
            aggfunc='count',
            fill_value=0
        )
        
        # Create heatmap
        sns.heatmap(monthly_activity, cmap='YlOrRd', annot=True, fmt='g')
        plt.title(f'Monthly Edit Activity for {crypto_name}')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{crypto_name}_monthly_heatmap.png'))
        plt.close()
        
    def create_all_visualizations(self, wiki_data_path, crypto_name):
        """
        Create all available visualizations for the Wikipedia edit history.
        
        Args:
            wiki_data_path (str): Path to the Wikipedia edit history CSV file
            crypto_name (str): Name of the cryptocurrency
        """
        try:
            df = self.load_wiki_data(wiki_data_path)
            
            print(f"\nCreating Wikipedia edit history visualizations for {crypto_name}...")
            
            self.plot_edits_over_time(df, crypto_name)
            self.plot_editor_distribution(df, crypto_name)
            self.plot_edit_size_distribution(df, crypto_name)
            self.plot_monthly_activity_heatmap(df, crypto_name)
            
            print(f"Visualizations saved to {self.output_path}")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")