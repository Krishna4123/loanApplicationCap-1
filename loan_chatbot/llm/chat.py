import os
from openai import OpenAI
from dotenv import load_dotenv

# Load env variables (API_KEY, BASE_URL)
load_dotenv()

# Initialize OpenAI Client
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def get_llm_response(final_prompt: str, system_context: str) -> str:
    """
    Sends the prompt to the LLM and returns the response.
    
    Args:
        final_prompt (str): The combined user + context prompt.
        system_context (str): The system behavior instruction.
        
    Returns:
        str: The LLM's response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=1000,
            temperature=0.9,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with LLM: {e}"
