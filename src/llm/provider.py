import os
import time
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)

class AzureLLMProvider:
    def __init__(self, config: dict):
        self.config = config
        load_dotenv()
        
        self.model_name = self.config.get('model', 'gpt-4o')
        
        if self.model_name == 'gpt-3.5-turbo':
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set in the environment for standard OpenAI models.")
            self.client = OpenAI(api_key=api_key)
        else:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            base_url = os.getenv("AZURE_OPENAI_BASE_URL")
            
            if not api_key or not base_url:
                raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_BASE_URL must be set in the environment for Azure models.")
                
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        
    def call_api_with_retry(self, messages: List[Dict], temperature: float = None) -> Optional[str]:
        """Call API with retry logic"""
        max_retries = self.config.get('max_retries', 3)
        temp = temperature if temperature is not None else self.config.get('temperature', 0.1)
        timeout = self.config.get('timeout_seconds', 30)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    timeout=timeout
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[ERROR] API call failed after {max_retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)
        
        return None
