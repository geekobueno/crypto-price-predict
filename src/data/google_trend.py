from pytrends.request import TrendReq
import pandas as pd
import time
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List

class GoogleTrendsFetcher:
    def __init__(self):
        """Initialize the Google Trends API client."""
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self.retry_count = 3
        self.sleep_time = 5  # seconds between retries
        
    def fetch_trends(self, crypto_name: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch Google Trends data for a cryptocurrency.
        
        Args:
            crypto_name (str): Name of the cryptocurrency
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing trends data or None if fetch fails
        """
        search_terms = [
            crypto_name,
            f"{crypto_name} crypto",
            f"{crypto_name} price"
        ]
        
        for attempt in range(self.retry_count):
            try:
                # Build payload
                self.pytrends.build_payload(
                    kw_list=search_terms,
                    cat=0,
                    timeframe=f'{start_date} {end_date}',
                    geo='',
                    gprop=''
                )
                
                # Get interest over time
                trends_df = self.pytrends.interest_over_time()
                
                if not trends_df.empty:
                    # Remove isPartial column and reset index
                    trends_df = trends_df.drop('isPartial', axis=1)
                    trends_df.reset_index(inplace=True)
                    
                    # Rename date column
                    trends_df.rename(columns={'date': 'timestamp'}, inplace=True)
                    
                    return trends_df
                    
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.sleep_time)
                    continue
                else:
                    logging.error(f"Failed to fetch Google Trends data for {crypto_name}")
                    return None
        
        return None
    
    def fetch_related_queries(self, crypto_name: str, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Fetch related queries for a cryptocurrency.
        
        Args:
            crypto_name (str): Name of the cryptocurrency
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            Optional[Dict]: Dictionary containing related queries or None if fetch fails
        """
        try:
            self.pytrends.build_payload(
                kw_list=[crypto_name],
                cat=0,
                timeframe=f'{start_date} {end_date}',
                geo='',
                gprop=''
            )
            
            related_queries = self.pytrends.related_queries()
            return related_queries.get(crypto_name, None)
            
        except Exception as e:
            logging.error(f"Failed to fetch related queries: {e}")
            return None
            
    def save_trends_data(self, df: pd.DataFrame, crypto_name: str, output_path: str) -> str:
        """
        Save trends data to CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame containing trends data
            crypto_name (str): Name of the cryptocurrency
            output_path (str): Directory to save the file
            
        Returns:
            str: Path to the saved file
        """
        filename = f"{crypto_name.lower()}_google_trends.csv"
        filepath = os.path.join(output_path, filename)
        df.to_csv(filepath, index=False)
        return filepath