import gradio as gr
import os
from ingestion.loader import load_text_files
from ingestion.chunker import chunk_text
from ingestion.embedder import get_embedding
from ingestion.vector_store import VectorStore
from retrieval.retriever import Retriever
from llm.chat import get_llm_response
from prompts.prompt_builder import PromptBuilder

# --- 1. Global Initialization ---
print("Initializing RAG System...")

# Initialize Vector Store
vector_store = VectorStore()

# Load and Ingest Policy Data
POLICY_DIR = os.path.join(os.path.dirname(__file__), "data", "loan_policies")
raw_docs = load_text_files(POLICY_DIR)

# Chunk and Embed
all_chunks = []
all_embeddings = []

print("Ingesting documents...")
for doc in raw_docs:
    chunks = chunk_text(doc)
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb is not None:
            all_chunks.append(chunk)
            all_embeddings.append(emb)

vector_store.add_documents(all_chunks, all_embeddings)
print(f"Ingestion complete. {len(all_chunks)} chunks stored.")

# Initialize Retriever
retriever = Retriever(vector_store)

# Load System Prompt
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
with open(SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_CONTEXT = f.read()

# --- 2. Application Logic ---

def process_application(text_input):
    """
    Handles application text input and stores in user session.
    """
    if not text_input or text_input.strip() == "":
        return "No application details provided.", None
    
    return "Application details saved! You can now ask questions about your eligibility.", text_input.strip()

def chat_logic(message, history, user_context_state):
    """
    Main RAG pipeline:
    1. Retrieve relevant policy chunks.
    2. Build prompt with System + Policy + User Context + Query.
    3. Get LLM response.
    """
    
    # 1. Retrieval
    retrieved_docs = retriever.retrieve(message, k=3)
    
    # 2. Prompt Assembly
    final_prompt = PromptBuilder.build(
        user_query=message,
        external_context=retrieved_docs,
        user_context=user_context_state
    )
    
    # 3. LLM Generation
    # We pass the loaded SYSTEM_CONTEXT to the LLM wrapper
    response = get_llm_response(final_prompt, SYSTEM_CONTEXT)
    
    return response

# --- 3. Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown("# 🏦 Smart Loan Consultant (RAG Demo)")
    gr.Markdown("Ask generic loan questions or upload your application to check eligibility.")

    # State for User Context (persists across chat)
    user_context_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Enter Application Details (Optional)")
            application_input = gr.Textbox(
                label="Your Application Details",
                placeholder="Name: John Doe\nAge: 24\nIncome: $45,000\nCredit Score: 720\nLoan Type: Personal",
                lines=8,
                interactive=True
            )
            submit_btn = gr.Button("Submit Application", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 2. Chat with Consultant")
            chatbot = gr.ChatInterface(
                fn=chat_logic,
                additional_inputs=[user_context_state]
            )

    # Event Wiring
    submit_btn.click(
        fn=process_application,
        inputs=[application_input],
        outputs=[upload_status, user_context_state]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
