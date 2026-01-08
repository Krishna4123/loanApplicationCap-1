import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# Load env variables (API_KEY, BASE_URL)
load_dotenv()

# Initialize OpenAI Client
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

import time

def get_embedding(text: str) -> np.ndarray:
    """
    Generates a vector embedding for the given text using OpenAI client.
    Includes exponential backoff to handle Rate Limit (429) errors.
    """
    text = text.replace("\n", " ")
    max_retries = 5
    base_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype='float32')
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                delay = base_delay * (2 ** attempt)
                print(f"\nRate limit hit (429). Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
                continue
            else:
                print(f"Error generating embedding: {e}")
                return None
    
    print("Max retries reached. Failed to generate embedding.")
    return None
