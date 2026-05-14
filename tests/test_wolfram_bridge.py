"""Functional tests for the Wolfram bridge script.

These run the real ``wolfram_llm_bridge.wl`` through ``wolframscript`` and
check its input-validation and JSON-I/O behaviour. They need no network or
credentials (the LLM-call paths are not exercised), only the
``wolframscript`` binary, so they are skipped where it is unavailable —
mirroring how the repo's credentialed integration tests behave on CI.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from shinka.utils.wolfram import (
    build_wolframscript_argv,
    is_wolframscript_available,
)

BRIDGE = (
    Path(__file__).resolve().parents[1]
    / "shinka"
    / "llm"
    / "providers"
    / "wolfram_llm_bridge.wl"
)

requires_wolframscript = pytest.mark.skipif(
    not is_wolframscript_available(),
    reason="wolframscript not installed",
)


def _run_bridge(input_text, tmp_path):
    in_path = tmp_path / "in.json"
    out_path = tmp_path / "out.json"
    in_path.write_text(input_text, encoding="utf-8")
    argv = build_wolframscript_argv(["-file", str(BRIDGE), str(in_path), str(out_path)])
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    result = None
    if out_path.exists():
        result = json.loads(out_path.read_text(encoding="utf-8"))
    return proc, result


def test_bridge_file_present_and_well_formed():
    """Cheap structural check that runs even without wolframscript."""
    assert BRIDGE.is_file()
    text = BRIDGE.read_text(encoding="utf-8")
    assert "ServiceConnect" in text
    assert "ServiceExecute" in text
    assert '"Chat"' in text


@requires_wolframscript
def test_bridge_rejects_malformed_input_json(tmp_path):
    proc, result = _run_bridge("this is not json", tmp_path)
    assert proc.returncode != 0
    assert result is not None
    assert "import input JSON" in result["error"]


@requires_wolframscript
def test_bridge_rejects_missing_required_fields(tmp_path):
    proc, result = _run_bridge(json.dumps({"service": "OpenAI"}), tmp_path)
    assert proc.returncode != 0
    assert result is not None
    assert "missing one of" in result["error"]
