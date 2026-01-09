import os
import sys
from pathlib import Path

# ============================================================================
# CRITICAL: Setup module search path for imports
# ============================================================================
# Shinka's eval_hydra.py does chdir to examples/ocr_cleaning/, so we need
# to ensure both the current directory and parent directories are in path
current_dir = Path(__file__).parent.resolve()
repo_root = current_dir.parent.parent

# Add paths for reliable imports
for p in [str(current_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shinka.core import run_shinka_eval

# ============================================================================
# Levenshtein distance with fallback implementation
# ============================================================================
try:
    import Levenshtein
except ImportError:
    import warnings
    warnings.warn(
        "python-Levenshtein not installed. Using pure Python fallback. "
        "Install via: pip install python-Levenshtein",
        ImportWarning
    )
    
    class Levenshtein:
        """Pure Python implementation of Levenshtein distance"""
        @staticmethod
        def distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return Levenshtein.distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

# ============================================================================
# Import data generator (now safe after path setup)
# ============================================================================
try:
    from data_generator import OCRDataGenerator
except ImportError as e:
    print(f"ERROR: Cannot import data_generator: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"sys.path: {sys.path[:5]}")
    raise


# ============================================================================
# Validation and Aggregation Functions
# ============================================================================

def validate_ocr_result(result):
    """
    Validates the structure of run_experiment output.
    
    Args:
        result: The return value from initial.py's run_experiment()
    
    Returns:
        (is_valid: bool, error_message: str | None)
    """
    if not isinstance(result, dict):
        return False, f"Result must be dict, got {type(result).__name__}"
    
    required_keys = ["average_score", "num_samples"]
    missing_keys = [k for k in required_keys if k not in result]
    
    if missing_keys:
        return False, f"Result missing keys: {missing_keys}"
    
    # Additional validation
    if not isinstance(result["average_score"], (int, float)):
        return False, f"average_score must be numeric, got {type(result['average_score'])}"
    
    if result["average_score"] < 0.0 or result["average_score"] > 1.0:
        return False, f"average_score out of range [0, 1]: {result['average_score']}"
    
    return True, None


def aggregate_ocr_metrics(results: list) -> dict:
    """
    Aggregates results from multiple runs into final metrics for Shinka.
    
    Args:
        results: List of dicts returned from run_experiment (length = num_runs)
    
    Returns:
        dict with 'combined_score' key (mandatory) and optional metrics
    """
    if not results:
        return {
            "combined_score": 0.0,
            "error": "No results to aggregate"
        }
    
    # Since num_runs=1, we only have one result
    result = results[0]
    avg_score = result["average_score"]
    
    metrics = {
        "combined_score": float(avg_score),  # ★ Shinka maximizes this
        "accuracy": float(avg_score),
        "num_samples": result["num_samples"],
    }
    
    # Optional: Add preview data for WebUI
    if "sample_predictions" in result:
        metrics["public"] = {
            "sample_predictions": result["sample_predictions"][:5]
        }
    
    return metrics


# ============================================================================
# Main Evaluation Entry Point
# ============================================================================

def main(program_path: str, results_dir: str):
    """
    Main evaluation function called by Shinka's eval_hydra.py
    
    Args:
        program_path: Absolute path to the program to evaluate (initial.py or evolved)
        results_dir: Directory to save results (metrics.json, correct.json)
    """
    print("=" * 80)
    print("OCR CLEANING EVALUATION")
    print("=" * 80)
    print(f"Program:  {program_path}")
    print(f"Results:  {results_dir}")
    print(f"CWD:      {os.getcwd()}")
    print("=" * 80)
    
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # ========================================
        # 1. Generate Test Dataset (Train/Val Split)
        # ========================================
        # Anti-overfitting: Use different seeds for variety
        # The program should generalize, not memorize specific cases
        print("[1/3] Generating test dataset...")
        gen = OCRDataGenerator(seed=42)
        train_dataset = gen.generate_batch(batch_size=30)  # Training examples
        gen_val = OCRDataGenerator(seed=12345)  # Different seed for validation
        val_dataset = gen_val.generate_batch(batch_size=20)  # Validation examples
        test_dataset = train_dataset + val_dataset  # Combined for now
        print(f"      Generated {len(test_dataset)} samples (30 train + 20 val)")
        
        # Sample preview
        if test_dataset:
            sample = test_dataset[0]
            print(f"      Sample: '{sample['input']}' → '{sample['ground_truth']}'")
        
        # ========================================
        # 2. Define Scoring Function
        # ========================================
        def score_fn(prediction: str, ground_truth: str) -> float:
            """Calculate normalized edit distance (1.0 = perfect match)"""
            dist = Levenshtein.distance(prediction, ground_truth)
            max_len = max(len(prediction), len(ground_truth), 1)
            return 1.0 - (dist / max_len)
        
        # ========================================
        # 3. Run Shinka Evaluation Framework
        # ========================================
        print("[2/3] Running evaluation via run_shinka_eval...")
        
        def get_kwargs(run_index: int) -> dict:
            """Provide kwargs for each evaluation run"""
            return {
                "dataset": test_dataset,
                "score_fn": score_fn
            }
        
        metrics, correct, error_msg = run_shinka_eval(
            program_path=program_path,
            results_dir=results_dir,
            experiment_fn_name="run_experiment",
            num_runs=1,
            get_experiment_kwargs=get_kwargs,
            validate_fn=validate_ocr_result,
            aggregate_metrics_fn=aggregate_ocr_metrics,
        )
        
        # ========================================
        # 4. Report Results
        # ========================================
        print("[3/3] Evaluation complete!")
        print("=" * 80)
        
        if correct:
            score = metrics.get('combined_score', 0.0)
            print(f"✓ SUCCESS: Score = {score:.4f}")
        else:
            print(f"✗ FAILED: {error_msg}")
            print(f"   Score: {metrics.get('combined_score', 0.0):.4f}")
        
        print("=" * 80)
        
    except Exception as e:
        # ========================================
        # Critical Error Handler
        # ========================================
        import traceback
        print("=" * 80)
        print("✗ FATAL ERROR IN EVALUATE.PY")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80)
        
        # Re-raise to ensure Shinka marks this as failed
        raise


# ============================================================================
# Script Entry Point (when called directly by Shinka)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OCR Cleaning Task Evaluator for ShinkaEvolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--program_path",
        type=str,
        required=True,
        help="Absolute path to program file to evaluate"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths for safety
    program_path = os.path.abspath(args.program_path)
    results_dir = os.path.abspath(args.results_dir)
    
    main(program_path, results_dir)