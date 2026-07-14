import re
import json
import time
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.cache import HybridRestaurantCache
from src.llm.provider import AzureLLMProvider
from src.llm.prompt_builder import RestaurantPromptBuilder
from src.llm.reflection import RestaurantSelfReflectionEngine
import logging
logger = logging.getLogger(__name__)

class HybridRestaurantSentimentAnalyzer:
    """Hybrid analyzer for restaurant reviews combining caching, reflection, and LLM parsing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = HybridRestaurantCache(config)
        self.llm = AzureLLMProvider(config)
        
        self.metrics = {
            'total_requests': 0,
            'cached_responses': 0,
            'api_calls': 0,
            'total_time': 0,
            'successful_parses': 0
        }
        
        self.reflection_engine = None
        self.prompt_engine = None
        
        reflection_text = ""
        if self.config.get('enable_self_reflection', True):
            self.reflection_engine = RestaurantSelfReflectionEngine(self.llm, self.config)
            reflection_text = self.reflection_engine.reflection_cache
            
        self.prompt_engine = RestaurantPromptBuilder(self.config, reflection_text)
    
    def _calculate_text_sophistication(self, text: str) -> float:
        """Calculate text sophistication level for adaptive processing"""
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
            
        avg_word_length = sum(len(w) for w in text.split()) / word_count
        complex_word_ratio = len(re.findall(r'\b\w{8,}\b', text)) / word_count
        hedge_ratio = len(re.findall(r'\b(somewhat|fairly|relatively|quite|rather|arguably)\b', text.lower())) / word_count
        culinary_terms = len(re.findall(r'\b(al dente|umami|seared|braised|reduction|infusion|aroma)\b', text.lower()))
        professional_terms = len(re.findall(r'\b(establishment|culinary|ambiance|presentation|professional|assessment)\b', text.lower()))
        
        sophistication_score = (
            (avg_word_length / 5) +
            (complex_word_ratio * 20) +
            (hedge_ratio * 15) +
            (culinary_terms / word_count * 25) +
            (professional_terms / word_count * 30)
        )
        return sophistication_score
    
    def _validate_restaurant_sentiment(self, sentiment: str, text: str, target: str) -> str:
        """Restaurant-specific sentiment validation"""
        if not sentiment:
            return "Neutral"
        
        sentiment_lower = sentiment.strip().lower()
        text_lower = text.lower()
        
        restaurant_positive = ["positive", "good", "great", "excellent", "amazing", "delicious", "tasty", "enjoyed"]
        restaurant_negative = ["negative", "bad", "terrible", "awful", "poor", "overcooked", "bland", "disappointed"]
        
        for word in restaurant_positive:
            if word in sentiment_lower: return "Positive"
        
        for word in restaurant_negative:
            if word in sentiment_lower: return "Negative"
            
        if any(w in sentiment_lower for w in ["neutral", "okay", "fine", "adequate", "acceptable", "average"]):
            return "Neutral"
        
        pos_indicators = ["love", "like", "enjoy", "perfect", "best", "recommend", "excellent"]
        neg_indicators = ["hate", "dislike", "worst", "avoid", "terrible", "awful", "poor"]
        
        pos_count = sum(1 for w in pos_indicators if w in text_lower)
        neg_count = sum(1 for w in neg_indicators if w in text_lower)
        
        if pos_count > neg_count: return "Positive"
        elif neg_count > pos_count: return "Negative"
        
        return "Neutral"
    
    def _parse_json_response(self, raw_output: str, text: str, target: str) -> Dict:
        """Robust JSON parsing with restaurant context"""
        if not raw_output:
            return self._get_default_response()
            
        cleaned = raw_output.strip()
        cleaned = re.sub(r'^```json\s*|\s*```$', '', cleaned)
        
        try:
            result = json.loads(cleaned)
            self.metrics['successful_parses'] += 1
            if 'sentiment' in result:
                result['sentiment'] = self._validate_restaurant_sentiment(result['sentiment'], text, target)
            return result
        except json.JSONDecodeError:
            pass
            
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, cleaned, re.DOTALL)
        
        for match in matches:
            try:
                normalized = re.sub(r"'([^']*)'", r'"\1"', match)
                result = json.loads(normalized)
                self.metrics['successful_parses'] += 1
                if 'sentiment' in result:
                    result['sentiment'] = self._validate_restaurant_sentiment(result['sentiment'], text, target)
                return result
            except:
                continue
                
        return self._extract_sentiment_fallback(raw_output, text, target)
    
    def _extract_sentiment_fallback(self, raw_output: str, text: str, target: str) -> Dict:
        """Fallback sentiment extraction"""
        text_lower = raw_output.lower()
        sentiment = "Neutral"
        confidence = 50
        
        if "positive" in text_lower or "pos" in text_lower:
            sentiment, confidence = "Positive", 70
        elif "negative" in text_lower or "neg" in text_lower:
            sentiment, confidence = "Negative", 70
            
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "aspect_focus": f"Fallback analysis for {target}",
            "key_phrases": [],
            "reasoning": "Unable to parse structured response",
            "ambiguity_level": "High"
        }
    
    def _get_default_response(self) -> Dict:
        return {
            "sentiment": "Neutral",
            "confidence": 50,
            "aspect_focus": "Analysis failed",
            "key_phrases": [],
            "reasoning": "System error",
            "ambiguity_level": "High"
        }
    
    def analyze_single(self, text: str, target: str) -> Dict:
        """Analyze single restaurant review"""
        start_time = time.time()
        
        # Check cache first
        if self.config.get('cache_enabled', True):
            cached_result = self.cache.get_cached_result(text, target)
            if cached_result:
                latency = time.time() - start_time
                self.metrics['cached_responses'] += 1
                self.metrics['total_requests'] += 1
                self.metrics['total_time'] += latency
                
                return {
                    "sentiment": cached_result.get('sentiment', 'Neutral'),
                    "confidence": cached_result.get('confidence', 50),
                    "cached": True,
                    "response_time_ms": round(latency * 1000, 2)
                }
                
        # Get similar examples
        similar_examples = []
        if self.config.get('enable_embeddings', True):
            similar_examples = self.cache.find_target_specific_examples(text, target)
            
        # Create prompts
        prompts = self.prompt_engine.create_analysis_prompt(text, target, similar_examples)
        
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ]
        
        api_start = time.time()
        raw_output = self.llm.call_api_with_retry(messages)
        api_latency = time.time() - api_start
        
        if raw_output:
            self.metrics['api_calls'] += 1
            self.metrics['total_time'] += api_latency
            result = self._parse_json_response(raw_output, text, target)
        else:
            result = self._get_default_response()
            
        result.update({
            "sophistication_score": round(self._calculate_text_sophistication(text), 2),
            "similar_examples_found": len(similar_examples),
            "cached": False,
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        })
        
        if self.config.get('cache_enabled', True):
            self.cache.add_to_cache(text, target, result)
            
        self.metrics['total_requests'] += 1
        return result
    
    def analyze_batch(self, texts: List[str], targets: List[str]) -> List[Dict]:
        """Analyze batch in parallel"""
        if len(texts) != len(targets):
            raise ValueError("Texts and targets must have same length")
            
        results = []
        max_workers = self.config.get('max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.analyze_single, texts[i], targets[i]): i
                for i in range(len(texts))
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    logger.warning(f"[WARNING] Error processing item {idx}: {e}")
                    results.append((idx, self._get_default_response()))
                    
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        total_requests = self.metrics['total_requests']
        cache_hits = self.metrics['cached_responses']
        api_calls = self.metrics['api_calls']
        
        metrics = {
            'total_reviews': total_requests,
            'cache_hit_rate': f"{(cache_hits/total_requests*100):.1f}%" if total_requests > 0 else "0%",
            'api_calls': api_calls,
            'parse_success_rate': f"{(self.metrics['successful_parses']/api_calls*100):.1f}%" if api_calls > 0 else "0%",
            'avg_response_time_ms': f"{(self.metrics['total_time']/total_requests*1000):.1f}" if total_requests > 0 else "N/A",
            'self_reflection_enabled': self.config.get('enable_self_reflection', True)
        }
        return metrics
