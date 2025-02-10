import os
from dotenv import load_dotenv
from groq import Groq
import logging
from typing import Optional

# Load environment variables
load_dotenv()

class GroqCaller:
    def __init__(self):
        """Initialize Groq client with API key."""
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=self.api_key)
        
    def get_crypto_name(self, symbol: str) -> Optional[str]:
        """
        Get the full cryptocurrency name from a symbol using Groq's LLM.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Optional[str]: Full cryptocurrency name or None if not found
        """
        try:
            # Craft a clear prompt for the LLM
            prompt = f"""What is the full name of the cryptocurrency with the symbol {symbol}? 
            Please respond with just the name, nothing else. For example, if asked about BTC, 
            just respond with "Bitcoin"."""
            
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="mixtral-8x7b-32768",  # You can change the model as needed
                temperature=0,  # Use 0 for more deterministic responses
                max_tokens=10   # We only need a short response
            )
            
            # Extract and clean the response
            if chat_completion.choices:
                crypto_name = chat_completion.choices[0].message.content.strip()
                # Remove any quotes or extra punctuation
                crypto_name = crypto_name.strip('"\'.,!? ')
                return crypto_name
            return None
            
        except Exception as e:
            logging.error(f"Error fetching crypto name from Groq for symbol {symbol}: {e}")
            return None

def get_crypto_name_from_groq(symbol: str) -> Optional[str]:
    """
    Wrapper function to get cryptocurrency name from Groq.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        
    Returns:
        Optional[str]: Full cryptocurrency name or None if not found
    """
    try:
        groq_caller = GroqCaller()
        return groq_caller.get_crypto_name(symbol)
    except Exception as e:
        logging.error(f"Failed to initialize Groq caller: {e}")
        return None