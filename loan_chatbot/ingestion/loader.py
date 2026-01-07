import os
from typing import List

def load_text_files(directory: str) -> List[str]:
    """
    Loads all .txt files from a directory and returns their content.
    
    Args:
        directory (str): The path to the directory containing text files.

    Returns:
        List[str]: A list of strings, where each string is the content of a file.
    """
    documents = []
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return documents

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Add filename as metadata to the content if needed, 
                    # but for now we just return raw text or a simple dict could be better?
                    # Keeping it simple as per "beginner-friendly".
                    documents.append(content)
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return documents
