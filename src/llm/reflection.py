import os
from typing import Optional
from src.llm.provider import AzureLLMProvider
import logging
logger = logging.getLogger(__name__)

class RestaurantSelfReflectionEngine:
    """Self-reflection engine to establish expert persona before analysis."""
    
    def __init__(self, llm_provider: AzureLLMProvider, config: dict):
        self.llm = llm_provider
        self.config = config
        paths = self.config.get('paths', {})
        self.cache_dir = paths.get('cache_reflection_dir', os.path.join("cache", "reflection"))
        self.cache_path = paths.get('cache_reflection_file', os.path.join(self.cache_dir, "restaurant_role_reflection.txt"))
        self.reflection_cache = ""
        
        self._init_directories()
        self._handle_cache_clearing()
        
        # Initialize reflection
        self.establish_expert_persona()
        
    def _init_directories(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"[INFO] Created reflection cache directory: {self.cache_dir}")
        
    def _handle_cache_clearing(self):
        if self.config.get('force_clear_reflection_cache', False):
            logger.info("[INFO] Force clear reflection cache is enabled. Deleting existing reflection.")
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)

    def establish_expert_persona(self):
        """Establish the expert persona and generate reflection."""
        # Check cache first
        if self.config.get('cache_enabled', True) and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.reflection_cache = f.read().strip()
                    if self.reflection_cache:
                        logger.info("[INFO] Loaded restaurant expert reflection from cache.")
                        return
            except Exception as e:
                logger.warning(f"[WARNING] Error reading reflection cache: {e}")

        # If no cache, generate new reflection
        logger.info("[INFO] Generating new restaurant expert reflection. This happens once...")
        
        paths = self.config.get('paths', {})
        prompt_path = paths.get('prompts_reflection', os.path.join("prompts", "restaurant", "restaurant_reflection.txt"))
        reflection_prompt = ""
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                reflection_prompt = f.read().strip()
        else:
            logger.warning(f"[WARNING] Reflection prompt missing at {prompt_path}")
            return
            
        messages = [
            {"role": "user", "content": reflection_prompt}
        ]
        
        reflection_temp = self.config.get('reflection_temperature', 0.3)
        reflection_response = self.llm.call_api_with_retry(messages, temperature=reflection_temp)
        
        if reflection_response:
            self.reflection_cache = reflection_response
            
            # Save to cache
            if self.config.get('cache_enabled', True):
                try:
                    with open(self.cache_path, 'w', encoding='utf-8') as f:
                        f.write(self.reflection_cache)
                    logger.info("[INFO] Saved new expert reflection to cache.")
                except Exception as e:
                    logger.warning(f"[WARNING] Could not save reflection cache: {e}")
        else:
            logger.error("[ERROR] Failed to generate reflection.")
