import sys
import os

# Add the current directory to path so imports work
sys.path.append(os.path.join(os.getcwd(), 'loan_chatbot'))

try:
    import openai
    import gradio
    import numpy
    import sklearn
    from loan_chatbot.ingestion.loader import load_text_files
    from loan_chatbot.ingestion.chunker import chunk_text
    from loan_chatbot.ingestion.embedder import get_embedding
    from loan_chatbot.ingestion.vector_store import VectorStore
    from loan_chatbot.retrieval.retriever import Retriever
    from loan_chatbot.prompts.prompt_builder import PromptBuilder
    from loan_chatbot.llm.chat import get_llm_response
    
    print("All imports successful, including OpenAI and Gradio.")
except Exception as e:
    print(f"Import failed: {e}")
