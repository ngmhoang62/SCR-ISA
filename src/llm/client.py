import os
from typing import Optional

class LLMClient:
    """A wrapper class for invoking LLM APIs. Supports Gemini and OpenAI."""
    
    def __init__(
        self, 
        provider: str, 
        model: str, 
        temperature: float = 0.2, 
        max_tokens: int = 1024
    ):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if self.provider == "gemini":
            from google import genai
            from google.genai import types
            # genai.Client picks up GEMINI_API_KEY environment variable automatically, 
            # but passing it explicitly makes key configuration explicit.
            api_key = os.getenv("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)
            self.types = types
        elif self.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(
                f"Unsupported provider: '{provider}'. "
                f"Currently supported options: 'gemini', 'openai'."
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Sends the system instruction and formatted prompt to the LLM.
        
        Args:
            system_prompt: The system prompt guiding the model behaviour.
            user_prompt: The formatted user prompt with variables populated.
        """
        if self.provider == "gemini":
            config = self.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config
            )
            if not response.text:
                raise RuntimeError("Empty response received from Gemini API.")
            return response.text

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
            result = response.choices[0].message.content
            if not result:
                raise RuntimeError("Empty response received from OpenAI API.")
            return result

        return ""
