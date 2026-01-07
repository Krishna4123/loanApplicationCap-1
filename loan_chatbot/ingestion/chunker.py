from typing import List

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of a specified character size with overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum number of characters per chunk.
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        
        # If we reached the end, stop
        if end == text_len:
            break
            
        # Move start forward, but backtrack by overlap
        start += chunk_size - overlap
    
    return chunks
