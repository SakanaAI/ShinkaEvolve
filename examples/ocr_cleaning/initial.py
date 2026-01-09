from typing import List, Dict

# EVOLVE-BLOCK-START
def clean_text(text: str, data_type: str) -> str:
    """
    Inputs:
      text: The noisy text string (e.g., "090-l234-5678")
      data_type: 'phone', 'amount', or 'address'
    Returns:
      Cleaned text string
    """
    # CRITICAL: Import re module inside function to ensure it's always available
    # The LLM should keep this import and add regex patterns below
    import re
    
    # Seed implementation: basic character replacements
    # TODO for LLM: Add sophisticated regex patterns and validation logic
    text = text.replace("O", "0")
    text = text.replace("l", "1")
    text = text.replace("I", "1")
    
    return text
# EVOLVE-BLOCK-END

def run_experiment(dataset: List[Dict], score_fn=None, **kwargs):
    """
    evaluate.py から呼び出されるエントリポイント
    """
    predictions = []
    scores = []
    
    for item in dataset:
        # 進化対象の関数を実行
        pred = clean_text(item['input'], item['type'])
        predictions.append(pred)
        
        # 個別のスコア計算（デバッグ用）
        if score_fn:
            score = score_fn(pred, item['ground_truth'])
            scores.append(score)
    
    return {
        "predictions": predictions,
        "scores": scores,
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "num_samples": len(dataset),
        # 上位3件の予測結果をログに残す（人間がWebUIで確認するため）
        "sample_predictions": list(zip([d['input'] for d in dataset[:3]], predictions[:3]))
    }