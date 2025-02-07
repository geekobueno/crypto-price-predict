import requests
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_URL = "https://api.groq.com/crypto-info"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_crypto_name_from_groq(symbol):
    """Fetch the full cryptocurrency name from Groq API."""
    try:
        response = requests.get(GROQ_API_URL, params={"symbol": symbol}, headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
        response.raise_for_status()
        data = response.json()
        return data.get("name")
    except Exception as e:
        print(f"Error fetching crypto name from Groq: {e}")
        return None
