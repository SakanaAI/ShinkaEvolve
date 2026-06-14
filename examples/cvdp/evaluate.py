"""
ShinkaEvolve evaluator for CVDP benchmark problems.

Runs a candidate Verilog module inside the CVDP Docker-based CocoTB harness
and parses pytest output for continuous scoring.

Called by ShinkaEvolve as:
    python evaluate.py --program_path <candidate.sv> --results_dir <dir>

Environment variables:
    CVDP_PROBLEM_FILE  Path to JSONL file with CVDP problems
                       (default: problems/cvdp_lite.jsonl)
    CVDP_PROBLEM_ID    Problem ID like "cvdp_copilot_lfsr_0001"
                       (default: first problem in the file)
    CVDP_TIMEOUT       Docker run timeout in seconds (default: 120)
    CVDP_SIM_IMAGE     Docker image for OSS simulation
                       (default: ghcr.io/efabless/cocotb-sim:latest)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


OSS_SIM_IMAGE_DEFAULT = "nvidia/cvdp-sim:v1.0.0"


def _safe_workspace_path(workspace: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` strictly inside ``workspace``.

    The CVDP harness paths (``harness.files`` keys and ``VERILOG_SOURCES``)
    come from the problem JSONL. An absolute path or one containing ``..``
    could otherwise write outside the temp workspace before the harness'
    docker-compose runs. Reject anything that does not stay inside.
    """
    rel = Path(rel_path)
    if rel.is_absolute() or rel.drive or rel.anchor:
        raise ValueError(f"Refusing absolute harness path: {rel_path!r}")

    workspace_resolved = workspace.resolve()
    candidate = (workspace_resolved / rel).resolve()
    if candidate != workspace_resolved and workspace_resolved not in candidate.parents:
        raise ValueError(
            f"Harness path escapes workspace: {rel_path!r} -> {candidate}"
        )
    return candidate


def _load_problems(jsonl_path: Path) -> List[Dict]:
    problems = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _find_problem(problems: List[Dict], problem_id: Optional[str]) -> Dict:
    if not problem_id:
        return problems[0]
    for p in problems:
        if p["id"] == problem_id:
            return p
    available = [p["id"] for p in problems[:10]]
    raise ValueError(
        f"Problem '{problem_id}' not found. Available: {available}..."
    )


def _extract_module_name(problem: Dict) -> str:
    env_content = problem["harness"]["files"].get("src/.env", "")
    for line in env_content.splitlines():
        if line.strip().startswith("TOPLEVEL"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return "design"


def _extract_rtl_path(problem: Dict) -> str:
    env_content = problem["harness"]["files"].get("src/.env", "")
    for line in env_content.splitlines():
        if line.strip().startswith("VERILOG_SOURCES"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                sources = parts[1].strip().split()
                return sources[0]
    return "/code/rtl/design.sv"


def _prepare_workspace(
    problem: Dict,
    candidate_path: str,
    sim_image: str,
) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="cvdp_eval_"))

    for rel_path, content in problem["harness"]["files"].items():
        file_path = _safe_workspace_path(workspace, rel_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    dockerfile = workspace / "Dockerfile"
    if dockerfile.exists():
        text = dockerfile.read_text(encoding="utf-8")
        text = text.replace("__OSS_SIM_IMAGE__", sim_image)
        text = text.replace("__VERIF_EDA_IMAGE__", sim_image)
        text = text.replace("__OSS_PNR_IMAGE__", sim_image)
        dockerfile.write_text(text, encoding="utf-8")

    compose_file = workspace / "docker-compose.yml"
    if compose_file.exists():
        text = compose_file.read_text(encoding="utf-8")
        text = text.replace("__OSS_SIM_IMAGE__", sim_image)
        text = text.replace("__VERIF_EDA_IMAGE__", sim_image)
        text = text.replace("__OSS_PNR_IMAGE__", sim_image)

        if "./rtl" not in text and "/code/rtl" not in text:
            text = text.replace(
                "- ./src/:/src/:ro",
                "- ./src/:/src/:ro\n      - ./rtl/:/code/rtl/:ro",
            )
        compose_file.write_text(text, encoding="utf-8")

    rtl_source = _extract_rtl_path(problem)
    rtl_rel = rtl_source.replace("/code/", "")
    dest = _safe_workspace_path(workspace, rtl_rel)
    dest.parent.mkdir(parents=True, exist_ok=True)

    candidate_text = Path(candidate_path).read_text(encoding="utf-8")
    lines = candidate_text.splitlines(keepends=True)
    clean = "".join(
        line for line in lines
        if "EVOLVE-BLOCK-START" not in line and "EVOLVE-BLOCK-END" not in line
    )

    dest.write_text(clean, encoding="utf-8")

    return workspace


def _parse_pytest_output(output: str) -> Tuple[int, int]:
    passed = 0
    failed = 0

    m_passed = re.search(r"(\d+)\s+passed", output)
    if m_passed:
        passed = int(m_passed.group(1))

    m_failed = re.search(r"(\d+)\s+failed", output)
    if m_failed:
        failed = int(m_failed.group(1))

    return passed, failed


def _classify_error(output: str) -> str:
    lower = output.lower()
    if "syntax error" in lower:
        return "syntax_error"
    if "unable to bind" in lower:
        return "port_binding_error"
    if "unknown module type" in lower:
        return "missing_module"
    if "error" in lower and ("compile" in lower or "build" in lower):
        return "compile_error"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    if "no such file" in lower or "not found" in lower:
        return "file_not_found"
    return "runtime_error"


def evaluate_cvdp(
    program_path: str,
    results_dir: str,
    problem: Dict,
    sim_image: str,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    project_name = f"cvdp_{uuid.uuid4().hex[:8]}"
    workspace = None

    try:
        workspace = _prepare_workspace(problem, program_path, sim_image)

        compose_file = workspace / "docker-compose.yml"
        if not compose_file.exists():
            return {
                "combined_score": 0.0,
                "public": {"stage": "setup", "error_class": "no_compose_file"},
                "private": {},
                "text_feedback": "No docker-compose.yml in problem harness.",
            }

        svc_result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "config", "--services"],
            capture_output=True, text=True, timeout=10, cwd=str(workspace),
        )
        services = [s for s in svc_result.stdout.strip().splitlines() if s.strip()]
        if not services:
            services = ["direct"]

        start = time.perf_counter()
        full_output = ""
        all_passed = 0
        all_failed = 0
        service_errors = []

        for service_name in services:
            run_cmd = [
                "docker", "compose",
                "-f", str(compose_file),
                "-p", project_name,
                "run", "--rm", service_name,
            ]

            try:
                result = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=str(workspace),
                )
            except subprocess.TimeoutExpired:
                return {
                    "combined_score": 0.0,
                    "public": {
                        "stage": "simulate",
                        "error_class": "timeout",
                        "elapsed": timeout_seconds,
                        "failed_service": service_name,
                    },
                    "private": {},
                    "text_feedback": f"Service '{service_name}' timed out after {timeout_seconds}s.",
                }

            svc_output = result.stdout + "\n" + result.stderr
            full_output += f"\n=== Service: {service_name} ===\n{svc_output}"

            svc_passed, svc_failed = _parse_pytest_output(svc_output)
            if svc_passed + svc_failed > 0:
                all_passed += svc_passed
                all_failed += svc_failed
            elif result.returncode != 0:
                service_errors.append(service_name)
                all_failed += 1

        elapsed = time.perf_counter() - start

        passed = all_passed
        failed = all_failed + len(service_errors)
        total = passed + failed

        if total == 0:
            error_class = _classify_error(full_output)
            return {
                "combined_score": 0.0,
                "public": {
                    "stage": "compile" if "error" in full_output.lower() else "simulate",
                    "error_class": error_class,
                    "elapsed": elapsed,
                    "passed": 0,
                    "failed": 0,
                    "total": 0,
                },
                "private": {"output": full_output[:3000]},
                "text_feedback": f"No tests ran ({error_class}). Output:\n{full_output[:1000]}",
            }

        score = (passed / total) * 100.0

        feedback_parts = []
        if passed == total:
            feedback_parts.append(f"All {total} tests passed.")
        else:
            feedback_parts.append(
                f"{passed}/{total} tests passed ({score:.1f}%). "
                f"{failed} tests failed."
            )

        assertion_errors = re.findall(
            r"AssertionError:.*|assert .*", full_output
        )
        if assertion_errors:
            for ae in assertion_errors[:3]:
                feedback_parts.append(f"  {ae.strip()[:200]}")

        return {
            "combined_score": score,
            "public": {
                "stage": "pass" if passed == total else "partial",
                "elapsed": elapsed,
                "passed": passed,
                "failed": failed,
                "total": total,
            },
            "private": {
                "output_tail": full_output[-2000:] if len(full_output) > 2000 else full_output,
            },
            "text_feedback": "\n".join(feedback_parts),
        }

    finally:
        if workspace:
            try:
                compose_path = workspace / "docker-compose.yml"
                if compose_path.exists():
                    subprocess.run(
                        ["docker", "compose", "-f", str(compose_path),
                         "-p", project_name, "down", "--volumes", "--remove-orphans"],
                        capture_output=True,
                        timeout=30,
                        cwd=str(workspace),
                    )
            except Exception:
                pass

        if workspace and workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)


def _save_results(results_dir: str, metrics: dict, correct: bool, error: str):
    correct_path = Path(results_dir) / "correct.json"
    metrics_path = Path(results_dir) / "metrics.json"
    correct_path.write_text(
        json.dumps({"correct": correct, "error": error}, indent=4),
        encoding="utf-8",
    )
    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")


def main(program_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    problem_file = os.environ.get("CVDP_PROBLEM_FILE", "problems/cvdp_lite.jsonl")
    problem_id = os.environ.get("CVDP_PROBLEM_ID", "")
    timeout = int(os.environ.get("CVDP_TIMEOUT", "120"))
    sim_image = os.environ.get("CVDP_SIM_IMAGE", OSS_SIM_IMAGE_DEFAULT)

    problem_path = Path(problem_file)
    if not problem_path.is_absolute():
        problem_path = Path(__file__).resolve().parent / problem_path

    try:
        problems = _load_problems(problem_path)
    except FileNotFoundError:
        metrics = {"combined_score": 0.0, "public": {}, "private": {}}
        _save_results(results_dir, metrics, False, f"Problem file not found: {problem_path}")
        return

    try:
        problem = _find_problem(problems, problem_id or None)
    except ValueError as e:
        metrics = {"combined_score": 0.0, "public": {}, "private": {}}
        _save_results(results_dir, metrics, False, str(e))
        return

    docker_check = subprocess.run(
        ["docker", "info"], capture_output=True, timeout=10
    )
    if docker_check.returncode != 0:
        metrics = {"combined_score": 0.0, "public": {}, "private": {}}
        _save_results(results_dir, metrics, False, "Docker is not available. Install and start Docker.")
        return
    
    # Check if the required CVDP simulation image exists
    try:
        image_check = subprocess.run(
            ["docker", "images", "-q", sim_image],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if not image_check.stdout.strip():
            print(f"CVDP simulation image '{sim_image}' not found.")
            print("   This image must be built locally from the CVDP repository.")
            print("   Steps:")
            print("   1. Clone https://github.com/NVlabs/cvdp-benchmark")
            print("   2. Build: docker build -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .")
            print("   3. Or set CVDP_SIM_IMAGE to a different cocotb+iverilog image")
            print("   See examples/cvdp/README.md for details.")
            metrics = {"combined_score": 0.0, "public": {}, "private": {}}
            _save_results(results_dir, metrics, False, f"CVDP simulation image '{sim_image}' not found. Build it from the CVDP repository.")
            return
    except Exception as e:
        print(f"Warning checking Docker images: {e}")

    try:
        metrics = evaluate_cvdp(
            program_path=program_path,
            results_dir=results_dir,
            problem=problem,
            sim_image=sim_image,
            timeout_seconds=timeout,
        )
    except Exception as e:
        metrics = {"combined_score": 0.0, "public": {}, "private": {}}
        _save_results(results_dir, metrics, False, str(e))
        return

    correct = metrics.get("combined_score", 0.0) == 100.0
    error = "" if correct else metrics.get("text_feedback", "")
    _save_results(results_dir, metrics, correct, error)

    print(f"Problem: {problem['id']} [{', '.join(problem['categories'])}]")
    print(f"Score: {metrics['combined_score']:.1f}/100")
    pub = metrics.get("public", {})
    if pub.get("total", 0) > 0:
        print(f"Tests: {pub['passed']}/{pub['total']} passed")
        print(f"Elapsed: {pub.get('elapsed', 0):.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Verilog candidate against CVDP benchmark")
    parser.add_argument("--program_path", type=str, default="initial.sv")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
