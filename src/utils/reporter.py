import pandas as pd
import logging
logger = logging.getLogger(__name__)

def generate_restaurant_analysis_report(results_df: pd.DataFrame) -> pd.DataFrame:
    """Generate detailed restaurant analysis report."""
    if results_df.empty:
        return pd.DataFrame()
        
    report_data = []
    if 'Target' in results_df.columns:
        for target in results_df['Target'].unique():
            subset = results_df[results_df['Target'] == target]
            sentiment_counts = subset['Predicted_Sentiment'].value_counts()
            total = len(subset)
            
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                count = sentiment_counts.get(sentiment, 0)
                percentage = (count / total * 100) if total > 0 else 0
                avg_conf = subset[subset['Predicted_Sentiment'] == sentiment]['Confidence'].mean()
                
                report_data.append({
                    'Target_Aspect': target,
                    'Sentiment': sentiment,
                    'Count': count,
                    'Percentage': round(percentage, 1),
                    'Avg_Confidence': round(avg_conf, 1) if not pd.isna(avg_conf) else 0,
                    'Sample_Size': total
                })
                
    return pd.DataFrame(report_data)

def format_batch_results(batch_results, texts, targets, df, batch_start) -> list:
    """Formats the raw dictionary results into flattened records for the DataFrame."""
    formatted_results = []
    for i, result in enumerate(batch_results):
        idx = batch_start + i
        if idx < len(df):
            row_result = {
                'Text': texts[i],
                'Target': targets[i],
                'Original_Sentiment': df.iloc[idx].get('polarity', 'N/A') if 'polarity' in df.columns else 'N/A',
                'Predicted_Sentiment': result.get('sentiment', 'Neutral'),
                'Is_Implicit': df.iloc[idx].get('is_implicit', False) if 'is_implicit' in df.columns else False,
                'Confidence': result.get('confidence', 50),
                'Aspect_Focus': result.get('aspect_focus', '')[:100],
                'Key_Phrases': ', '.join(result.get('key_phrases', [])[:3]),
                'Emotional_Tone': result.get('emotional_tone', 'Unknown'),
                'Sophistication_Score': result.get('sophistication_score', 0),
                'Ambiguity_Level': result.get('ambiguity_level', 'Medium'),
                'Cached': result.get('cached', False),
                'Similar_Examples': result.get('similar_examples_found', 0),
                'Response_Time_MS': result.get('response_time_ms', 0),
                'Reasoning': result.get('reasoning', '')[:150]
            }
            formatted_results.append(row_result)
    return formatted_results

def print_batch_summary(batch_results) -> None:
    """Prints a quick positive/negative/neutral count for a batch."""
    batch_sentiments = [r.get('sentiment', 'Neutral') for r in batch_results]
    pos_count = sum(1 for s in batch_sentiments if s == 'Positive')
    neg_count = sum(1 for s in batch_sentiments if s == 'Negative')
    neu_count = sum(1 for s in batch_sentiments if s == 'Neutral')
    logger.info(f"       Results: Positive={pos_count}, Negative={neg_count}, Neutral={neu_count}")
