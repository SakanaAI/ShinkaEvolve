import os
import json
import time
import random
import argparse
import importlib.util
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np

from shinka.llm import LLMClient
from lm_judge_prompt import make_lm_input_and_output_processors


def evaluate_with_lm_judge(
    program_path: str,
    results_dir: str,
    lm_input_and_output_processors: Callable | Tuple[Callable, Callable] = (
        make_lm_input_and_output_processors),
    llm_judge_names=[
        "azure-gpt-4.1",
        "us.anthropic.claude-sonnet-4-6-v1:0",
        "gemini-2.5-pro",
    ],
    llm_judge_kwargs=dict(
        temperatures=0.0,
        max_tokens=8196,
        reasoning_efforts="low",
        model_sample_probs=None,
        output_model=None,
        verbose=True
    ),
    limit_max_characters: Optional[int] = None,
    num_samples: int = 20,
    seed: int = 42,
):

    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None:
        print(f"Error: Could not load spec for module at {program_path}")
        return
    if spec.loader is None:
        print(f"Error: No loader found for module at {program_path}")
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    random.seed(seed)
    random_ints = [random.randint(0, 10000) for _ in range(num_samples)]

    start_t = time.time()

    error = ""
    correct = True

    if isinstance(lm_input_and_output_processors, tuple):
        get_evaluation_prompt, extract_results = lm_input_and_output_processors
    else:
        get_evaluation_prompt, extract_results = lm_input_and_output_processors(
            number_of_samples=num_samples)
    try:
        start_t = time.time()
        novel_outputs = module.run_experiment(random_ints)
        if limit_max_characters is not None:
            novel_outputs = [
                output[:limit_max_characters] for output in novel_outputs]

        if not isinstance(llm_judge_names, list):
            llm_judge_names = [llm_judge_names]
        if not isinstance(llm_judge_kwargs, list):
            llm_judge_kwargs = [llm_judge_kwargs] * len(llm_judge_names)

        llm_judges = [LLMClient(
            model_names=llm_judge_names[i],
            **llm_judge_kwargs[i],
        ) for i in range(len(llm_judge_names))]

        lm_judge_sys_prompt, lm_judge_message = get_evaluation_prompt(
            novel_outputs)

        results_dict = {}
        all_final_scores = []
        total_cost = 0.0

        for llm_judge_idx in range(len(llm_judges)):
            llm_judge = llm_judges[llm_judge_idx]

            llm_judge_kwargs = llm_judge.get_kwargs()

            llm_judge_response = llm_judge.query(
                msg=lm_judge_message,
                system_msg=lm_judge_sys_prompt,
                llm_kwargs=llm_judge_kwargs,
            )

            total_costs = llm_judge_response.cost or 0
            response_content = llm_judge_response.content

            llm_judge_scores: dict = extract_results(response_content)

            total_cost += total_costs
            for k, v in llm_judge_scores.items():
                results_dict['judge{}_{}'.format(llm_judge_idx + 1, k)] = v
            all_final_scores.append(llm_judge_scores.get(
                'final_novelty_score', 0.0))

        results_dict['combined_score'] = float(np.mean(all_final_scores))

        if results_dict['combined_score'] is None:
            results_dict['combined_score'] = 0.0

        metrics = {}
        metrics["runtime"] = time.time() - start_t
        metrics["public"] = results_dict
        metrics["private"] = {"evaluation_cost": total_cost}
        metrics["combined_score"] = results_dict['combined_score']
        error = ""
        correct = True
    except Exception as e:
        print(f"Error: {e}")
        metrics = {
            "combined_score": 0,
            "public": {},
            "private": {},
            "runtime": 0,
        }
        error = str(e)
        correct = False

    print(metrics)
    elapsed = metrics["runtime"]
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"Completed after {hours}h {minutes}m {seconds}s")
    # Save correct to JSON file
    correct_file = os.path.join(results_dir, "correct.json")
    with open(correct_file, "w") as f:
        json.dump({"correct": correct, "error": error}, f, indent=4)
    print(f"Correct saved to {correct_file}")

    # Save metrics to JSON file
    metrics_file = os.path.join(
        results_dir,
        "metrics.json",
    )
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Novelty evaluation functions and default script"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to the program to evaluate",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results and logs",
    )
    parsed_args = parser.parse_args()
    Path(parsed_args.results_dir).mkdir(parents=True, exist_ok=True)
    evaluate_with_lm_judge(parsed_args.program_path, parsed_args.results_dir)
