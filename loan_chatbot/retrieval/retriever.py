from ingestion.embedder import get_embedding
from ingestion.vector_store import VectorStore
from typing import List, Dict, Any

class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieves the top-k most relevant text chunks for a given query.
        
        Args:
            query (str): User's question.
            k (int): Number of chunks to retrieve.
            
        Returns:
            List[str]: A list of relevant text details.
        """
        # 1. Convert query to embedding
        query_embedding = get_embedding(query)
        
        if query_embedding is None:
            return []

        # 2. Search vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        # 3. Extract text from results
        extracted_chunks = [res['chunk'] for res in results]
        
        return extracted_chunks
