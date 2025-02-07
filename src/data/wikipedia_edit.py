import requests
import pandas as pd
import os
from config.config import Config

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

def get_all_wikipedia_edits(crypto_name):
    """
    Fetch the full edit history of a cryptocurrency's Wikipedia page and save it as a CSV file.
    
    :param crypto_name: The name of the cryptocurrency.
    :return: Path to the saved CSV file or None if failed.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": crypto_name,
        "rvlimit": "max",  # Fetch the maximum allowed edits per request
        "rvprop": "timestamp|user|comment|ids",
        "rvdir": "newer"  # Fetch from oldest to newest
    }
    
    all_revisions = []
    cont_token = None  # Continuation token for pagination

    try:
        while True:
            if cont_token:
                params["rvcontinue"] = cont_token  # Set continuation token

            response = requests.get(WIKIPEDIA_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                print(f"No Wikipedia page found for {crypto_name}.")
                return None

            page_id = next(iter(pages))
            revisions = pages[page_id].get("revisions", [])

            if not revisions:
                print(f"No edit history found for {crypto_name}.")
                return None

            all_revisions.extend(revisions)

            # Check if there is more data to fetch
            cont_token = data.get("continue", {}).get("rvcontinue")
            if not cont_token:
                break  # No more edits available

        # Convert data into a DataFrame
        df = pd.DataFrame(all_revisions)
        df.rename(columns={"revid": "revision_id", "timestamp": "edit_time", "user": "editor", "comment": "edit_comment"}, inplace=True)

        # Ensure output directory exists
        os.makedirs(Config.WIKIPEDIA_PATH, exist_ok=True)

        # Save to CSV
        csv_filename = os.path.join(Config.WIKIPEDIA_PATH, f"{crypto_name}_edit_history.csv")
        df.to_csv(csv_filename, index=False)

        print(f"Full Wikipedia edit history saved to: {csv_filename}")
        return csv_filename

    except Exception as e:
        print(f"Error fetching Wikipedia edit history for {crypto_name}: {e}")
        return None
