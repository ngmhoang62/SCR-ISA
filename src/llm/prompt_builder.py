import os
from typing import Dict, List
import logging
logger = logging.getLogger(__name__)

class RestaurantPromptBuilder:
    def __init__(self, config: dict, reflection_text: str = ""):
        self.config = config
        self.reflection_text = reflection_text
        paths = self.config.get('paths', {})
        self.base_prompt = self._load_file(paths.get('prompts_base_system', os.path.join("prompts", "restaurant", "system_prompts", "base_system.txt")))
        self.framework_text = self._load_file(paths.get('prompts_analysis', os.path.join("prompts", "restaurant", "user_prompts", "analysis_prompt.txt")))
        self.system_prompt = self._build_system_prompt()

    def _load_file(self, filepath: str) -> str:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        logger.warning(f"[WARNING] Prompt file missing: {filepath}")
        return ""

    def _build_system_prompt(self) -> str:
        prompt = self.base_prompt
        if self.reflection_text:
            max_len = self.config.get('reflection_summary_length', 300)
            summary = self.reflection_text[:max_len] + "..." if len(self.reflection_text) > max_len else self.reflection_text
            prompt += f"\n\nEXPERT INSIGHTS: {summary}"
            
        # Save to debug output
        paths = self.config.get('paths', {})
        system_debug_path = paths.get('debug_system_prompt', os.path.join("data", "outputs", "debug_system_prompt.txt"))
        if not os.path.exists(os.path.dirname(system_debug_path)):
            os.makedirs(os.path.dirname(system_debug_path), exist_ok=True)
        
        with open(system_debug_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
            
        logger.info(f"[INFO] Compiled and saved system prompt to {system_debug_path}")
            
        return prompt

    def create_analysis_prompt(self, text: str, target: str, similar_examples: List[Dict] = None) -> Dict[str, str]:
        few_shot_text = ""
        if similar_examples and len(similar_examples) > 0:
            few_shot_text = "\n\n**RELEVANT RESTAURANT EXAMPLES:**\n"
            for i, example in enumerate(similar_examples):
                few_shot_text += f"\nExample {i+1} - Target: {example.get('target', 'N/A')}\n"
                few_shot_text += f"Review: \"{example.get('text', '')}\"\n"
                analysis = example.get('analysis', {})
                few_shot_text += f"Analysis: {analysis.get('sentiment', 'N/A')} "
                few_shot_text += f"(Confidence: {analysis.get('confidence', 80)}%)\n"
                few_shot_text += f"Key Insight: {analysis.get('reason', 'N/A')[:80]}...\n"
                few_shot_text += "-" * 60 + "\n"

        framework_formatted = self.framework_text.replace("{target}", target)

        user_prompt = f"""Analyze the sentiment in this restaurant review regarding the specific aspect: "{target}"

RESTAURANT REVIEW:
"{text}"
{few_shot_text}
{framework_formatted}

**OUTPUT REQUIREMENTS:**
Respond ONLY with a valid JSON object in this exact format:
{{
  "sentiment": "Positive" or "Negative" or "Neutral",
  "confidence": 0-100,
  "aspect_focus": "How specifically the review addresses {target}",
  "key_phrases": ["phrase1", "phrase2", "phrase3"],
  "emotional_tone": "Enthusiastic/Appreciative/Critical/Ironic/Neutral/etc",
  "reasoning": "Step-by-step explanation focusing on {target} evaluation",
  "ambiguity_level": "Low/Medium/High"
}}

IMPORTANT: Base your analysis SOLELY on comments about {target}. Ignore unrelated aspects."""

        paths = self.config.get('paths', {})
        user_debug_path = paths.get('debug_user_prompt', os.path.join("data", "outputs", "debug_user_prompt.txt"))
        if not os.path.exists(os.path.dirname(user_debug_path)):
            os.makedirs(os.path.dirname(user_debug_path), exist_ok=True)
            
        with open(user_debug_path, 'w', encoding='utf-8') as f:
            f.write(user_prompt)
            
        if getattr(self, '_first_user_prompt_logged', False) is False:
            logger.info(f"[INFO] Writing ongoing debug user prompts to {user_debug_path}")
            self._first_user_prompt_logged = True
            
        return {
            "system": self.system_prompt,
            "user": user_prompt
        }
