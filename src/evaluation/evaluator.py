import re
from typing import Any, Dict

class Evaluator:
    """Contains standard text metrics for comparing LLM predictions to ground truth labels."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Removes duplicate whitespace, strips, and lowercases text."""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.strip().lower())

    @classmethod
    def extract_boxed_answer(cls, text: str) -> str:
        """Extracts content inside a LaTeX \boxed{...} block.
        
        Example: "The final answer is \\boxed{42}." -> "42"
        """
        if not text:
            return ""
        # Search for \boxed{...} recursively matching non-curly brace characters
        match = re.search(r'\\boxed\{([^{}]*)\}', text)
        if match:
            return match.group(1).strip()
        return text.strip()

    @classmethod
    def exact_match(cls, prediction: str, reference: str, use_boxed_extraction: bool = True) -> Dict[str, Any]:
        """Checks if the prediction matches the reference target.
        
        Args:
            prediction: Raw output string from LLM.
            reference: Target ground truth string.
            use_boxed_extraction: Whether to extract \boxed{...} from prediction.
        """
        pred_to_eval = cls.extract_boxed_answer(prediction) if use_boxed_extraction else prediction
        
        normalized_pred = cls.normalize_text(pred_to_eval)
        normalized_ref = cls.normalize_text(reference)
        
        is_match = normalized_pred == normalized_ref
        
        return {
            "extracted_prediction": pred_to_eval,
            "ground_truth": reference,
            "exact_match": is_match
        }
