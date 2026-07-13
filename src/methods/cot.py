from typing import Any, Dict
from src.methods.base import BasePromptingMethod

class ChainOfThoughtMethod(BasePromptingMethod):
    """Chain of Thought (CoT) prompting implementation."""
    
    def run(self, variables: Dict[str, Any]) -> str:
        """Executes the Chain of Thought pipeline:
        1. Loads system prompt.
        2. Populates inputs into the template.
        3. Invokes the model.
        """
        # Retrieve system instructions
        system_prompt = self.loader.get_system_prompt()
        
        # Populate template placeholders (e.g. {context}, {question})
        user_prompt = self.loader.format_user_prompt(variables)
        
        # Generate the response via the LLM client
        response_text = self.client.generate(
            system_prompt=system_prompt, 
            user_prompt=user_prompt
        )
        return response_text
