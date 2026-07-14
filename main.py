import os
import yaml
import time
import pandas as pd
import numpy as np

import logging
from src.utils.logger import setup_logger

from src.core.analyzer import HybridRestaurantSentimentAnalyzer
from src.utils.reporter import (
    generate_restaurant_analysis_report,
    format_batch_results,
    print_batch_summary
)

logger = logging.getLogger(__name__)

def load_config() -> dict:
    config_path = os.path.join("configs", "default_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        
    rest_config = config_data.get('restaurant', {})
    
    # Inject model name into paths to prevent overwriting across different models
    model_name = rest_config.get("model", "gpt-4o")
    paths = rest_config.get("paths", {})
    for key, path in paths.items():
        if isinstance(path, str) and "{model}" in path:
            paths[key] = path.format(model=model_name)
            
    return rest_config

def process_restaurant_sentiment_hybrid():
    """Main processing function for restaurant sentiment analysis"""
    config = load_config()
    paths = config.get('paths', {})
    setup_logger(paths.get('log_dir', 'logs'))
    
    analyzer = HybridRestaurantSentimentAnalyzer(config)
    
    input_csv_path = paths.get('input_csv', 'data/inputs/restaurants/restaurants_test_extracted.csv')
    output_csv_path = paths.get('output_csv', 'data/outputs/restaurant_sentiment_results.csv')
    logger.info("=" * 60)
    logger.info(f"[INFO] INITIATING HYBRID RESTAURANT SENTIMENT ANALYSIS")
    logger.info("=" * 60)
    
    logger.info(f"[INFO] Loading dataset from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file missing: {input_csv_path}")
        
    df = pd.read_csv(input_csv_path)
    
    required_columns = ['text', 'aspect_term', 'polarity', 'is_implicit']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    logger.info(f"[INFO] Total Reviews: {len(df)}")
    logger.info(f"[INFO] Model: {config.get('model')}")
    logger.info(f"[INFO] Parallel Workers: {config.get('max_workers')}")
    logger.info(f"[INFO] Cache Enabled: {config.get('cache_enabled')}")
    logger.info(f"[INFO] Self-Reflection: {config.get('enable_self_reflection')}")
    
    all_results = []
    start_time = time.time()
    batch_size = config.get('batch_size', 10)
    
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        logger.info(f"[INFO] Processing batch {batch_start//batch_size + 1}/"
              f"{np.ceil(len(df)/batch_size).astype(int)} "
              f"({batch_start}-{batch_end-1})")
              
        texts = batch_df['text'].astype(str).tolist()
        targets = batch_df['aspect_term'].astype(str).tolist()
        
        batch_results = analyzer.analyze_batch(texts, targets)
        
        formatted_results = format_batch_results(batch_results, texts, targets, df, batch_start)
        all_results.extend(formatted_results)
        print_batch_summary(batch_results)
        
    results_df = pd.DataFrame(all_results)
    
    total_time = time.time() - start_time
    perf_metrics = analyzer.get_performance_metrics()
    
    logger.info(f"\n[INFO] RESTAURANT ANALYSIS COMPLETE")
    logger.info(f"[INFO] Total Time: {total_time/60:.2f} minutes")
    logger.info(f"[INFO] Average per review: {total_time/len(df):.2f} seconds")
    logger.info(f"[INFO] Cache Hit Rate: {perf_metrics['cache_hit_rate']}")
    logger.info(f"[INFO] API Calls Made: {perf_metrics['api_calls']}")
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"[INFO] Results saved to: {output_csv_path}")
    
    if not results_df.empty:
        logger.info(f"\n[INFO] FINAL RESTAURANT SENTIMENT DISTRIBUTION:")
        sentiment_dist = results_df['Predicted_Sentiment'].value_counts()
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(results_df)) * 100
            avg_conf = results_df[results_df['Predicted_Sentiment'] == sentiment]['Confidence'].mean()
            logger.info(f"       {sentiment}: {count} reviews ({percentage:.1f}%), Avg Confidence: {avg_conf:.1f}")
            
    # Trigger Evaluation
    from src.evaluation.evaluator import evaluate_results
    eval_output_path = paths.get('eval_output', 'data/outputs/restaurants/evaluation_results.txt')
    evaluate_results(output_csv_path, eval_output_path)
            
    return results_df

if __name__ == "__main__":
    process_restaurant_sentiment_hybrid()
