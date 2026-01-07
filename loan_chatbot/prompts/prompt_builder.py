class PromptBuilder:
    @staticmethod
    def build(user_query: str, external_context: list[str], user_context: str = None) -> str:
        """
        Assembles the final prompt from various context sources.
        
        Structure:
        1. Context Introduction
        2. External Context (Retrieved Policy)
        3. User Context (Uploaded Application Data) - Optional
        4. Task (User Query)
        
        Args:
            user_query (str): The user's question.
            external_context (list[str]): Retrieved chunks from the vector store.
            user_context (str): The text from the uploaded user file (optional).
            
        Returns:
            str: The master prompt for the LLM.
        """
        
        # 1. Join Retrieved Policy Chunks
        context_str = "\n\n".join(external_context)
        
        # 2. Start building the prompt
        prompt = "### EXTERNAL CONTEXT (Loan Policy Data):\n"
        prompt += f"{context_str}\n\n"
        
        # 3. Add User Context if available
        if user_context:
            prompt += "### USER CONTEXT (Loan Application Data):\n"
            prompt += f"{user_context}\n\n"
        else:
            prompt += "### USER CONTEXT:\nNo application uploaded.\n\n"

        # 4. Add the Task
        prompt += "### TASK:\n"
        prompt += f"{user_query}\n\n"
        
        prompt += "Please answer the TASK using ONLY the EXTERNAL CONTEXT and USER CONTEXT provided above."
        
        return prompt
