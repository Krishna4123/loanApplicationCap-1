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

def get_embedding(text: str) -> np.ndarray:
    """
    Generates a vector embedding for the given text using OpenAI client.
    
    Args:
        text (str): The text to embed.
        
    Returns:
        np.ndarray: The embedding vector.
    """
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype='float32')
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return a zero vector or handle error appropriately
        # For simplicity, returning None implies failure
        return None
