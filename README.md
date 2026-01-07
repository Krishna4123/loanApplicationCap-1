# Loan Consultation RAG Chatbot

A beginner-friendly Retrieval-Augmented Generation (RAG) application built with Python and Gradio.
This project demonstrates how to build a context-aware chatbot that answers questions based on a specific Loan Policy document.

## Features
- **RAG Pipeline**: Ingests loan policy text, chunks it, and retrieves relevant info using vector embeddings.
- **Custom Vector Store**: A transparent, from-scratch implementation of a vector database using NumPy.
- **Context Engineering**: strict system prompts to prevent hallucinations.
- **Gradio UI**: Simple chat interface with file upload capability for checking loan eligibility.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r loan_chatbot/requirements.txt
   ```

2. **Configure API Keys**
   - Open `loan_chatbot/.env`
   - Add your OpenAI API Key and Base URL:
     ```
     API_KEY=sk-...
     BASE_URL=https://...
     ```

3. **Run the Application**
   ```bash
   python loan_chatbot/app.py
   ```

## Project Structure
- `ingestion/`: Handles loading, chunking, and embedding documents.
- `retrieval/`: Logic to search best matching documents.
- `prompts/`: System prompts and dynamic prompt assembly.
- `llm/`: Wrapper for the LLM API.
- `app.py`: The main entry point for the UI.
