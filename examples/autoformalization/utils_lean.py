import json
import re
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

from openai import OpenAI

from lean_interact import LeanREPLConfig, AutoLeanServer, TempRequireProject
from lean_interact.interface import Command, FileCommand, LeanError
from lean_interact.utils import remove_lean_comments

logger = logging.getLogger(__name__)


def generate_prompt(file_path: str) -> str:
    """
    Generate a prompt for an LLM-based prover agent for the completion of the proof provided in ``file_path``.

    Args:
        file_path (str): the path to a Lean 4 file.

    Returns:
        str: a prompt for an LLM-based prover agent based on the contents of ``file_path``.

    Notes:
        - This function is based on the example provided by the DeepSeek Development team.
        https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B
    """
    formal_statement = Path(file_path).read_text(encoding="UTF-8")

    prompt = """
    Complete the following Lean 4 code:

    ```lean4
    {}
    ```

    Please *only* output the submitted, fully proven lean program. Do not add any reasoning or explanation to your answer. You are not allowed to include "sorrys" in your result. Make sure to include all imports in your final answer.
    """.strip()

    return prompt.format(formal_statement)


def generate_proof(
    file_path: str, model: str, proof_client: OpenAI, sampling_params: dict, timeout: int
) -> Tuple[str, Optional[str]]:
    """
    Complete the proof provided at ``file_path`` using ``model``. The ``model`` should be hosted via the vLLM-based
    OpenAI client.

    Args:
        file_path (str): the path to a Lean 4 file containing an incomplete proof (may include ``sorry``s)
        model (str): the name of the LLM to use. Recommended are ``DeepSeek-Prover-V2-7B``,
            ``Goedel-LM/Goedel-Prover-V2-8B`` or``Goedel-LM/Goedel-Prover-V2-32B``.
        proof_client (OpenAI): the inference API.
        sampling_params (dict): a dictionary of the sampling params that are passed as parameters to the API request.
        timeout (int): The timeout for the request in seconds.

    Returns:
        str: the generated proof.

    """
    try:
        prompt = generate_prompt(file_path)
        if not prompt:
            return file_path, None
        response = proof_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            **sampling_params,
            timeout=timeout,
        )
        proof_text = response.choices[0].message.content

        results_dir, _ = os.path.split(file_path)
        fname = os.path.join(results_dir, "unprocessed_proof.lean")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(proof_text)

        proof_text = postprocess(proof_text)
        fname = os.path.join(results_dir, "processed_proof.lean")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(proof_text)

        # Reformat proofs by removing comments, as they can sometimes cause issues when validating the proofs
        return file_path, remove_lean_comments(proof_text)

    except Exception as e:
        logger.error(f"Error generating proof for {file_path}: {e}")
        return file_path, None


def postprocess(proof: str) -> str:
    """
    Postprocess the ``proof``, fixing common syntax errors.

    Args:
        proof (str): a Lean 4 proof.

    Returns:
        (str): the post-processed and cleaned proof.
    """
    try:
        proof = proof.split("```lean4")[1].replace("`", "")
    except IndexError:
        proof = proof.replace("`", "")
    proof = validate_imports(proof_text=proof)
    # Reformat proofs by removing comments, as they can sometimes cause issues when validating the proofs
    clean_lean = remove_lean_comments(proof)
    if clean_lean.endswith("D"):
        clean_lean = clean_lean[:-1]
    # Replace "∑ n in range" with "∑ n ∈ range"
    lean_txt = fix_range_notation(clean_lean)
    # Replace " pi " with π
    lean_txt = re.sub(r'\s+pi\s+', ' π ', lean_txt)
    return lean_txt


def fix_range_notation(text: str) -> str:
    """
    Replace '∑/∏ variable in range' with '∑/∏ variable ∈ range' in Lean 4 code, regardless of the variable name.

    Args:
        text (str): The input string containing Lean code.

    Returns:
        str: The string with all sum notations fixed.
    """
    pattern = r"([∑∏]\s+)(\w+)(\s+)in(\s+)"
    replacement = r"\1\2\3∈\4"
    return re.sub(pattern, replacement, text)


def validate_imports(
    proof_text: str,
    standard_imports: Optional[str] = None,
    open_imports: Optional[str] = None,
) -> str:
    """
    Check whether the imports are present in the proof header. Add missing imports if required.

    Args:
        proof_text (str): the proof text.
        standard_imports (Optional[List]): a list of the standard imports required to solve most proofs in chemical
            physics.
        open_imports (Optional[List]): a list of the standard opens required to solve mst proofs in chemical physics.

    Returns:
        str: the proof text including the required imports.

    """
    if not standard_imports: # Default argument
        standard_imports = ["import mathlib", ]

    if not open_imports:
        open_imports = ["Real", "BigOperators", "Topology", "Set", "Filter", "Finset"]

    imports, opens, proof = [], [], []

    proof_lines = proof_text.split("\n")
    for line in proof_lines:
        if line.startswith("import"):  # Verify imports
            continue

        if line.lower() == "lean":  # fix parsing issues
            continue

        elif line.startswith("open"):  # Open statement
            curated_line = line
            for statement in open_imports:
                if statement not in curated_line:
                    curated_line += f" {statement}"
            opens.append(curated_line)

        else:  # The rest of the proof text
            proof.append(line)

    for statement in standard_imports:  # Add the remaining standard imports
        if statement not in imports:
            imports.append(statement)

    proof_text = "\n".join(imports) + "\n" + "\n".join(opens) + "\n" + "\n".join(proof)

    return proof_text.strip()


def validate_lean(
    path_or_str: str, allow_sorry: bool, timeout: int, verbose: bool, lean_version: str = "v4.24.0"
) -> Tuple[bool, str]:
    """
    Verify the validity of the Lean program  found at ``path`` via LeanInteract. The function builds a Lean
    Read-Eval-Print-Loop (REPL) for solving Lean programs. The function returns ``True`` if the lean program has run
    successfully and when the Lean code is considered valid.

    Args:
        path_or_str (str): The path of the file to be operated on by the REPL.
        allow_sorry (bool): True to allow for partially complete proofs that include ``sorry`` statements.
        timeout (int): The timeout for the request in seconds.
        verbose (bool): Whether to print additional information when downloading and building the Lean REPL,
            and running the Lean REPL request using ``run``.
        lean_version (str): The Lean version used. Default is ``"v4.24.0"``, which is the latest version around October
            2025.

    Returns:
        bool: ``True`` if the lean program associated with ``path`` is considered valid and has run successfully.
        str: the outcome or exception associated with the validation.

    Notes:
        - Store the lean output as a json, storing the header and formalization code only, to run as a ``Command``.
        - `Formalization_code` must end by `:=`.
    """
    project = TempRequireProject(lean_version=lean_version, require="mathlib")
    config = LeanREPLConfig(project=project)

    try:
        server = AutoLeanServer(config)
        command = FileCommand(path=path_or_str) if path_or_str.endswith(".lean") else Command(cmd=path_or_str)
        server_output = server.run(command, timeout=timeout, verbose=verbose, add_to_session_cache=False)

        logger.info(server_output.messages)

        if not server_output.lean_code_is_valid(allow_sorry=allow_sorry):
            return False, f"The provided lean file {path_or_str} is invalid or contains 'sorry'."
        elif isinstance(server_output, LeanError):
            return False, f"{path_or_str}:" + server_output.message
        else:
            return (
                True,
                f"Run {path_or_str} terminated successfully: {server_output}",
            )  # The content may still contain errors

    except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError) as e:
        logger.error(f"Error while checking the lean file {path_or_str}: {e}")
        return False, e

async def async_validate_lean(
    path_or_str: str, allow_sorry: bool, timeout: int, verbose: bool, lean_version: str = "v4.24.0"
) -> Tuple[bool, str]:
    """
    Verify the validity of the Lean program  found at ``path`` via LeanInteract. The function builds a Lean
    Read-Eval-Print-Loop (REPL) for solving Lean programs. The function returns ``True`` if the lean program has run
    successfully and when the Lean code is considered valid.

    Args:
        path_or_str (str): The path of the file to be operated on by the REPL.
        allow_sorry (bool): True to allow for partially complete proofs that include ``sorry`` statements.
        timeout (int): The timeout for the request in seconds.
        verbose (bool): Whether to print additional information when downloading and building the Lean REPL,
            and running the Lean REPL request using ``run``.
        lean_version (str): The Lean version used. Default is ``"v4.24.0"``, which is the latest version around October
            2025.

    Returns:
        bool: ``True`` if the lean program associated with ``path`` is considered valid and has run successfully.
        str: the outcome or exception associated with the validation.

    Notes:
        - Store the lean output as a json, storing the header and formalization code only, to run as a ``Command``.
        - `Formalization_code` must end by `:=`.
    """
    return validate_lean(
        path_or_str, allow_sorry=allow_sorry, timeout=timeout, verbose=verbose, lean_version=lean_version
    )
