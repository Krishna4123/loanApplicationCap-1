# Loan Consultation RAG Chatbot

A beginner-friendly Retrieval-Augmented Generation (RAG) application built with Python and Gradio.
This project demonstrates how to build a context-aware chatbot that answers questions based on a specific Loan Policy document.

## Features
- **RAG Pipeline**: Ingests loan policy text, chunks it, and retrieves relevant info using vector embeddings.
- **Persistent Storage**: Embeddings are saved to disk and can be reused across runs (no re-chunking needed).
- **CLI Control**: Choose to load existing embeddings or re-chunk when documents are updated.
- **Custom Vector Store**: A transparent, from-scratch implementation of a vector database using NumPy.
- **Context Engineering**: strict system prompts to prevent hallucinations.
- **Gradio UI**: Simple chat interface with text input for loan application details.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r loan_chatbot/requirements.txt
   ```

2. **Configure API Keys**
   - Copy `loan_chatbot/.env.example` to `loan_chatbot/.env`
   - Add your OpenAI API Key and Base URL:
     ```
     API_KEY=sk-...
     BASE_URL=https://...
     ```

3. **Add Policy Documents**
   - Place your loan policy text files in `loan_chatbot/data/loan_policies/`

4. **Run the Application**
   ```bash
   python loan_chatbot/app.py
   ```
   
   On first run, the app will:
   - Chunk your policy documents
   - Generate embeddings via OpenAI
   - Save embeddings to `vector_store.pkl`
   
   On subsequent runs, you'll see:
   ```
   Existing embeddings found!
   Options:
     1. Load existing embeddings (fast)
     2. Re-chunk and re-embed documents (slow, use after updates)
   Enter choice (1/2):
   ```
   
   Choose option 1 for instant startup, or option 2 if you've updated your policy files.

## Project Structure
- `ingestion/`: Handles loading, chunking, and embedding documents.
- `retrieval/`: Logic to search best matching documents.
- `prompts/`: System prompts and dynamic prompt assembly.
- `llm/`: Wrapper for the LLM API.
- `app.py`: The main entry point for the UI.
