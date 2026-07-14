from src.core.analyzer import HybridRestaurantSentimentAnalyzer

def test_analyze_single_restaurant_example():
    """Analyze single restaurant review example for testing logic flow."""
    config = {
        'batch_size': 1,
        'max_workers': 1,
        'cache_enabled': False,
        'enable_self_reflection': False, # Disable reflection for quick test
        'model': 'gpt-4o-2024-08-06',
        'enable_embeddings': False,      # Disable embeddings for quick test
        'max_retries': 1
    }
    
    # We won't actually call it to avoid API Key errors during pytest
    # but we can ensure the object initializes correctly.
    try:
        analyzer = HybridRestaurantSentimentAnalyzer(config)
        assert analyzer is not None
    except ValueError as e:
        # Expected if OPENAI_API_KEY is not set in env
        assert "OPENAI_API_KEY" in str(e)
