from abc import ABC, abstractmethod
from typing import Any, Dict
from src.llm.client import LLMClient
from src.prompts.loader import PromptLoader

class BasePromptingMethod(ABC):
    """Abstract Base Class for prompting methods."""
    
    def __init__(self, client: LLMClient, loader: PromptLoader):
        self.client = client
        self.loader = loader

    @abstractmethod
    def run(self, variables: Dict[str, Any]) -> str:
        """Executes the prompt logic. Must be overridden by subclasses.
        
        Args:
            variables: Variables to populate the user prompt template.
        Returns:
            The raw text response from the model.
        """
        pass
