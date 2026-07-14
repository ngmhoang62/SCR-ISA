import os
import time
import json
import pickle
import hashlib
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import logging
logger = logging.getLogger(__name__)

class HybridRestaurantCache:
    """Hybrid cache system combining performance and semantic understanding for restaurants."""
    
    def __init__(self, config: dict):
        self.config = config
        paths = self.config.get('paths', {})
        self.cache_dir = paths.get('cache_sentiment_dir', os.path.join("cache", "sentiment"))
        self.few_shot_cache_dir = paths.get('cache_few_shot_dir', os.path.join("cache", "few_shot"))
        
        self.sentiment_cache_path = paths.get('cache_sentiment_file', os.path.join(self.cache_dir, "sentiment_cache.pkl"))
        self.aspect_cache_path = paths.get('cache_aspect_file', os.path.join(self.cache_dir, "aspect_patterns.pkl"))
        self.few_shot_cache_path = paths.get('cache_few_shot_file', os.path.join(self.few_shot_cache_dir, "top_k_examples.pkl"))
        
        self.embedding_model = None
        self.sentiment_cache = {}
        self.aspect_patterns_cache = {}
        self.few_shot_cache = {}
        
        self._init_directories()
        self._handle_cache_clearing()
        self._load_caches()
        
        if self.config.get('enable_embeddings', True):
            try:
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"[WARNING] Could not load embedding model: {e}")
                
        # Load few shot JSON data
        self.few_shot_examples = self._load_few_shot_json()

    def _init_directories(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.few_shot_cache_dir, exist_ok=True)

    def _handle_cache_clearing(self):
        if self.config.get('force_clear_sentiment_cache', False):
            logger.info("[INFO] Force clear sentiment cache is enabled. Deleting existing sentiment caches.")
            if os.path.exists(self.sentiment_cache_path):
                os.remove(self.sentiment_cache_path)
            if os.path.exists(self.aspect_cache_path):
                os.remove(self.aspect_cache_path)
                
        if self.config.get('force_clear_few_shot_cache', False):
            logger.info("[INFO] Force clear few shot cache is enabled. Deleting existing top-k caches.")
            if os.path.exists(self.few_shot_cache_path):
                os.remove(self.few_shot_cache_path)

    def _load_caches(self):
        """Load caches from disk."""
        try:
            if os.path.exists(self.sentiment_cache_path):
                with open(self.sentiment_cache_path, 'rb') as f:
                    self.sentiment_cache = pickle.load(f)
            if os.path.exists(self.aspect_cache_path):
                with open(self.aspect_cache_path, 'rb') as f:
                    self.aspect_patterns_cache = pickle.load(f)
            if os.path.exists(self.few_shot_cache_path):
                with open(self.few_shot_cache_path, 'rb') as f:
                    self.few_shot_cache = pickle.load(f)
        except Exception as e:
            logger.warning(f"[WARNING] Cache load error: {e}")
            self.sentiment_cache = {}
            self.aspect_patterns_cache = {}
            self.few_shot_cache = {}

    def _save_caches(self):
        """Save sentiment and aspect caches to disk."""
        try:
            with open(self.sentiment_cache_path, 'wb') as f:
                pickle.dump(self.sentiment_cache, f)
            with open(self.aspect_cache_path, 'wb') as f:
                pickle.dump(self.aspect_patterns_cache, f)
        except Exception as e:
            logger.warning(f"[WARNING] Cache save error: {e}")
            
    def _save_few_shot_cache(self):
        try:
            with open(self.few_shot_cache_path, 'wb') as f:
                pickle.dump(self.few_shot_cache, f)
        except Exception as e:
            logger.warning(f"[WARNING] Few-shot cache save error: {e}")

    def _load_few_shot_json(self) -> Dict:
        json_path = self.config.get('paths', {}).get('few_shot_json', os.path.join("data", "few_shot", "restaurant_examples.json"))
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"[WARNING] Could not load few-shot examples from JSON: {e}")
        return {}

    def get_cache_key(self, text: str, target: str) -> str:
        """Generate cache key combining text and target."""
        normalized = f"{text.strip().lower()}|{target.strip().lower()}"
        return hashlib.md5(normalized.encode()).hexdigest()

    def categorize_target(self, target: str) -> str:
        """Categorize target into broader categories."""
        target_lower = target.lower()
        
        target_mapping = {
            'food_quality': ['food', 'quality', 'taste', 'flavor', 'dish', 'meal', 'cuisine'],
            'service': ['service', 'wait', 'staff', 'server', 'attention', 'professional'],
            'ambiance_value': ['ambiance', 'atmosphere', 'value', 'price', 'cost', 'money', 'expensive'],
            'expert_implicit': ['experience', 'overall', 'dining', 'restaurant', 'establishment']
        }
        
        for category, keywords in target_mapping.items():
            if any(keyword in target_lower for keyword in keywords):
                return category
                
        return 'expert_implicit'

    def find_target_specific_examples(self, text: str, target: str) -> List[Dict]:
        """Find target-specific similar examples using top-k and cache."""
        if not self.embedding_model:
            return []
            
        cache_key = self.get_cache_key(text, target)
        if cache_key in self.few_shot_cache:
            return self.few_shot_cache[cache_key]
        
        target_category = self.categorize_target(target)
        top_k = self.config.get('top_k', 3)
        relevant_examples = []
        
        if target_category in self.few_shot_examples:
            scored_examples = []
            query = f"{text} {target}"
            query_embedding = self.embedding_model.encode([query])[0]
            
            for example in self.few_shot_examples[target_category]:
                if not isinstance(example, dict) or 'text' not in example:
                    continue
                example_text = f"{example['text']} {example['target']}"
                example_embedding = self.embedding_model.encode([example_text])[0]
                
                similarity = np.dot(query_embedding, example_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(example_embedding)
                )
                scored_examples.append((similarity, example))
                
            # Sort by similarity descending
            scored_examples.sort(key=lambda x: x[0], reverse=True)
            relevant_examples = [ex for sim, ex in scored_examples[:top_k]]
            
        # Save to few-shot cache
        self.few_shot_cache[cache_key] = relevant_examples
        self._save_few_shot_cache()
        
        return relevant_examples

    def get_cached_result(self, text: str, target: str) -> Optional[Dict]:
        """Get cached sentiment result."""
        cache_key = self.get_cache_key(text, target)
        return self.sentiment_cache.get(cache_key)

    def add_to_cache(self, text: str, target: str, result: Dict):
        """Add result to cache with pattern extraction."""
        cache_key = self.get_cache_key(text, target)
        
        self.sentiment_cache[cache_key] = {
            'sentiment': result.get('sentiment'),
            'confidence': result.get('confidence'),
            'target': target,
            'timestamp': time.time()
        }
        
        target_category = self.categorize_target(target)
        if target_category not in self.aspect_patterns_cache:
            self.aspect_patterns_cache[target_category] = []
            
        pattern_summary = {
            'text_preview': text[:100],
            'sentiment': result.get('sentiment'),
            'confidence': result.get('confidence'),
            'key_phrases': result.get('key_phrases', [])[:3]
        }
        self.aspect_patterns_cache[target_category].append(pattern_summary)
        
        if len(self.sentiment_cache) % 50 == 0:
            self._save_caches()
