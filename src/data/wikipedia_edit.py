import mwapi
import pandas as pd
import os
import logging
from typing import Optional
from datetime import datetime

class WikipediaEditFetcher:
    def __init__(self):
        """
        Initialize Wikipedia Edit Fetcher using MediaWiki API.
        """
        self.session = mwapi.Session('https://en.wikipedia.org')

    def fetch_edit_history(self, page_title: str) -> Optional[pd.DataFrame]:
        """
        Fetch complete edit history for a Wikipedia page.
        Args:
            page_title (str): Title of the Wikipedia page
        Returns:
            Optional[pd.DataFrame]: DataFrame containing edit history or None if failed
        """
        try:
            revisions = []
            cont = {'rvcontinue': '0'}
            
            while True:
                result = self.session.get(
                    action='query',
                    prop='revisions',
                    titles=page_title,
                    rvprop=['ids', 'timestamp', 'user', 'size', 'comment'],
                    rvlimit='max',
                    formatversion=2,
                    **cont
                )
                
                page = result['query']['pages'][0]
                if 'revisions' not in page:
                    logging.warning(f"No edit history found for {page_title}")
                    return None
                
                revisions.extend(page['revisions'])
                
                if 'continue' not in result:
                    break
                    
                cont = {'rvcontinue': result['continue']['rvcontinue']}

            df = pd.DataFrame(revisions)
            
            if df.empty:
                logging.warning(f"No data retrieved for {page_title}")
                return None
            
            # Calculate size changes
            df['size_change'] = df['size'].diff()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp in descending order
            df = df.sort_values('timestamp', ascending=False)
            
            return df

        except Exception as e:
            logging.error(f"Error fetching edit history for {page_title}: {e}")
            return None

    def save_edit_history(self, output_path: str, df: pd.DataFrame, page_title: str) -> Optional[str]:
        """
        Save edit history to CSV file.
        Args:
            output_path (str): Directory path to save edit history files
            df (pd.DataFrame): DataFrame containing edit history
            page_title (str): Title of the Wikipedia page
        Returns:
            Optional[str]: Path to saved file or None if failed
        """
        try:
            os.makedirs(output_path, exist_ok=True)
            safe_name = page_title.lower().replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d')
            filename = f"{safe_name}_wikipedia_edits_{timestamp}.csv"
            filepath = os.path.join(output_path, filename)
            df.to_csv(filepath, index=False)
            logging.info(f"Wikipedia edit history saved to: {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Error saving edit history for {page_title}: {e}")
            return None

def get_all_wikipedia_edits(page_title: str, output_path: str) -> Optional[str]:
    """
    Wrapper function to fetch and save Wikipedia edit history.
    Args:
        page_title (str): Title of the Wikipedia page
        output_path (str): Directory path to save edit history files
    Returns:
        Optional[str]: Path to saved CSV file or None if failed
    """
    try:
        fetcher = WikipediaEditFetcher()
        df = fetcher.fetch_edit_history(page_title)
        if df is not None and not df.empty:
            return fetcher.save_edit_history(output_path, df, page_title)
        return None
    except Exception as e:
        logging.error(f"Error processing Wikipedia edits for {page_title}: {e}")
        return None