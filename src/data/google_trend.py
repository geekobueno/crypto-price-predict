import pandas as pd
from pytrends.request import TrendReq
import logging
from typing import Optional, Dict
from datetime import datetime
import os
from pathlib import Path

class GoogleTrendsFetcher:
    def __init__(self):
        """Initialize the Google Trends API client."""
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
            logging.info("Google Trends client initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Google Trends client: {e}")
            raise

    def fetch_trends(self, keyword: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch Google Trends data for a specific keyword and time period.
        
        Args:
            keyword (str): Search term to get trends for
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with trends data or None if failed
        """
        try:
            timeframe = f"{start_date} {end_date}"
            self.pytrends.build_payload([keyword], timeframe=timeframe)
            
            df = self.pytrends.interest_over_time()
            if df.empty:
                logging.warning(f"No trends data found for {keyword}")
                return None
                
            # Clean up the DataFrame
            if 'isPartial' in df.columns:
                df = df.drop('isPartial', axis=1)
                
            df = df.reset_index()
            df.columns = ['date'] + [col for col in df.columns if col != 'date']
            
            logging.info(f"Successfully fetched trends data for {keyword}")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching Google Trends data for {keyword}: {e}")
            return None

    def save_trends_data(self, df: pd.DataFrame, keyword: str, output_path: str) -> Optional[str]:
        """
        Save trends data to CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame containing trends data
            keyword (str): Keyword used for the search
            output_path (str): Directory to save the file
            
        Returns:
            Optional[str]: Path to saved file or None if failed
        """
        try:
            os.makedirs(output_path, exist_ok=True)
            
            safe_name = keyword.lower().replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d')
            filename = f"{safe_name}_trends_{timestamp}.csv"
            filepath = os.path.join(output_path, filename)
            
            df.to_csv(filepath, index=False)
            logging.info(f"Saved trends data to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error saving trends data for {keyword}: {e}")
            return None