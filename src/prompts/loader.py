from typing import Any, Dict
from src.utils.helpers import load_file

class PromptLoader:
    """Manages loading of system instructions and formatting of user prompt templates."""
    
    def __init__(self, system_prompt_path: str, user_template_path: str):
        self.system_prompt_path = system_prompt_path
        self.user_template_path = user_template_path
        self.system_prompt = load_file(system_prompt_path)
        self.user_template = load_file(user_template_path)

    def get_system_prompt(self) -> str:
        """Returns the raw system prompt text."""
        return self.system_prompt

    def format_user_prompt(self, variables: Dict[str, Any]) -> str:
        """Formats the user template with dynamic input variables.
        
        Args:
            variables: Key-value dictionary containing template placeholders.
        """
        try:
            return self.user_template.format(**variables)
        except KeyError as e:
            raise KeyError(
                f"Prompt formatting failed. Expected variable {e} was not provided. "
                f"Available template: {self.user_template_path}"
            )
