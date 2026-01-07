import numpy as np
import pickle
import os
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self):
        """
        Initializes an in-memory vector store.
        
        Structure:
        - self.chunks: List[str] - Holds the actual text chunks.
        - self.embeddings: np.ndarray - Holds the corresponding embeddings matrix.
        """
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = None

    def add_documents(self, chunks: List[str], embeddings: List[np.ndarray]):
        """
        Adds chunks and their embeddings to the store.
        """
        if not chunks or not embeddings:
            return

        new_embeddings = np.array(embeddings)
        
        self.chunks.extend(chunks)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches for the top-k most similar chunks.
        
        Args:
            query_embedding (np.ndarray): The embedding of the user query.
            k (int): Number of results to return.
            
        Returns:
            List[Dict]: A list of results, each containing 'chunk' and 'score'.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Ensure query_embedding is 2D array (1, D)
        query_vec = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity between query and all stored embeddings
        # shape: (1, N)
        similarities = cosine_similarity(query_vec, self.embeddings)[0]
        
        # Get indices of top-k scores (sorted descending)
        top_k_indices = similarities.argsort()[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": similarities[idx]
            })
            
        return results

    def save(self, path: str):
        """Saves the store to a file."""
        with open(path, 'wb') as f:
            pickle.dump({'chunks': self.chunks, 'embeddings': self.embeddings}, f)
            print(f"VectorStore saved to {path}")

    def load(self, path: str):
        """Loads the store from a file."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.embeddings = data['embeddings']
            print(f"VectorStore loaded from {path}")
        else:
            print(f"No existing vector store found at {path}")
