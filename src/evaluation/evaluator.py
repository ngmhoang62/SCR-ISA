import pandas as pd
from sklearn.metrics import f1_score
import logging
import os

logger = logging.getLogger(__name__)

def evaluate_results(csv_path: str, output_txt_path: str):
    """
    Reads the predicted CSV results and calculates Macro-F1 score
    for the overall dataset (SA) and implicit subset (ISA).
    """
    logger.info(f"[EVALUATION] Reading results from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"[EVALUATION] Failed to read CSV: {e}")
        return
        
    if 'Original_Sentiment' not in df.columns or 'Predicted_Sentiment' not in df.columns or 'Is_Implicit' not in df.columns:
        logger.error("[EVALUATION] Missing required columns for evaluation.")
        return
        
    # Drop N/A or cases where Original_Sentiment wasn't known
    df = df[df['Original_Sentiment'].notna() & (df['Original_Sentiment'] != 'N/A')]
    
    if len(df) == 0:
        logger.warning("[EVALUATION] No valid ground truth labels found.")
        return
        
    # Standardize to lowercase
    y_true_all = df['Original_Sentiment'].astype(str).str.lower().tolist()
    y_pred_all = df['Predicted_Sentiment'].astype(str).str.lower().tolist()
    
    # Calculate Macro-F1 SA
    f1_sa = f1_score(y_true_all, y_pred_all, average='macro')
    
    # Filter for ISA
    # Ensure Is_Implicit is parsed as boolean
    implicit_df = df[df['Is_Implicit'].astype(str).str.lower() == 'true']
    if len(implicit_df) > 0:
        y_true_isa = implicit_df['Original_Sentiment'].astype(str).str.lower().tolist()
        y_pred_isa = implicit_df['Predicted_Sentiment'].astype(str).str.lower().tolist()
        f1_isa = f1_score(y_true_isa, y_pred_isa, average='macro')
    else:
        f1_isa = 0.0
        
    report = (
        "RESTAURANT DOMAIN EVALUATION RESULTS\n"
        "====================================\n"
        f"Total evaluated samples (SA): {len(df)}\n"
        f"Implicit samples (ISA): {len(implicit_df)}\n\n"
        f"Macro-F1 SA: {f1_sa:.4f}\n"
        f"Macro-F1 ISA: {f1_isa:.4f}\n"
    )
    
    logger.info(f"\n{report}")
    
    # Write to file
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    logger.info(f"[EVALUATION] Report saved to {output_txt_path}")
