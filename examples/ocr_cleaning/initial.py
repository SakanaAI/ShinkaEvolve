from typing import List, Dict
import re  # STRUCTURAL FIX: Keep at top level to prevent NameError during evolution

# EVOLVE-BLOCK-START
def clean_text(text: str, data_type: str) -> str:
    """
    Context-Aware OCR Error Correction
    
    CHALLENGE: The same character can have different meanings based on context!
    - In phone numbers: "O" should become "0" (zero)
    - In company names: "O" should stay "O" (letter)
    - In invoice IDs: "INV-2024" has "I" as letter, but numbers have "1"
    
    Args:
        text: Noisy OCR text (e.g., "T0Y0TA-l2345" or "O9O-1234-5678")
        data_type: One of 'phone', 'amount', 'company_code', 'invoice_id', 'address'
    
    Returns:
        Corrected text with context-appropriate character restoration
    """
    # NOTE: 're' module is imported at top of file - do not add import here
    
    # === SEED IMPLEMENTATION (Naive - to be evolved) ===
    # This naive approach treats all cases the same - which is WRONG!
    # The LLM must evolve context-specific logic
    
    if data_type == 'phone':
        # Phone: all letters should become digits
        text = text.replace("O", "0").replace("l", "1").replace("I", "1")
    elif data_type == 'amount':
        # Amount: digits and punctuation only
        text = text.replace("O", "0").replace("l", "1")
    elif data_type == 'company_code':
        # Company code: TRICKY! Prefix is letters, suffix is digits
        # "T0Y0TA-l2345" → "TOYOTA-12345"
        # HINT: Structure has parts separated by hyphen
        if "-" in text:
            parts = text.split("-")
            # TODO: Evolve different character mappings for each part
            # Currently using same naive replacements for both parts
            prefix = parts[0].replace("O", "0").replace("l", "1").replace("I", "1")
            suffix = parts[1].replace("O", "0").replace("l", "1").replace("I", "1") if len(parts) > 1 else ""
            text = f"{prefix}-{suffix}" if suffix else prefix
        else:
            text = text.replace("O", "0").replace("l", "1").replace("I", "1")
            
    elif data_type == 'invoice_id':
        # Invoice: "lNV-2O24-OOl23" → "INV-2024-00123"
        # HINT: Structure has prefix (letters) and suffix parts (numbers)
        if "-" in text:
            parts = text.split("-")
            # TODO: Evolve different processing for prefix vs suffix segments
            # Currently using naive replacements for all parts
            cleaned_parts = [part.replace("O", "0").replace("l", "1").replace("I", "1") for part in parts]
            text = "-".join(cleaned_parts)
        else:
            text = text.replace("O", "0").replace("l", "1").replace("I", "1")
            
    else:
        # Address: mixed, preserve structure
        text = text.replace("O", "0").replace("l", "1")
    
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