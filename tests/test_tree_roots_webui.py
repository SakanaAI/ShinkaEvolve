import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_TREE_HTML = REPO_ROOT / "shinka" / "webui" / "viz_tree.html"


def _extract_js_function_source(function_name: str) -> str:
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")
    match = re.search(rf"function {function_name}\([^)]*\)\s*\{{", html)
    assert match, f"Could not find function {function_name} in viz_tree.html"

    brace_depth = 0
    start = match.start()
    body_start = html.find("{", match.start())

    for idx in range(body_start, len(html)):
        char = html[idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return html[start : idx + 1]

    raise AssertionError(f"Could not extract complete function body for {function_name}")


def _run_normalize_tree_roots(nodes: list[dict]) -> list[dict]:
    function_source = _extract_js_function_source("normalizeTreeRoots")
    script = "\n".join(
        [
            function_source,
            f"const input = {json.dumps(nodes)};",
            "const result = normalizeTreeRoots(input);",
            "console.log(JSON.stringify(result));",
        ]
    )
    completed = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_normalize_tree_roots_virtualizes_unified_and_spawned_roots():
    nodes = [
        {
            "id": "___unified_root___",
            "parent_id": None,
            "agent_name": "Initial Program",
            "generation": 0,
            "isUnifiedRoot": True,
        },
        {
            "id": "gen0_a",
            "parent_id": "___unified_root___",
            "agent_name": "seed-a",
            "generation": 0,
        },
        {
            "id": "gen0_b",
            "parent_id": "___unified_root___",
            "agent_name": "seed-b",
            "generation": 0,
        },
        {
            "id": "spawned_root",
            "parent_id": None,
            "agent_name": "spawned-root",
            "generation": 6,
        },
        {
            "id": "spawned_child",
            "parent_id": "spawned_root",
            "agent_name": "spawned-child",
            "generation": 7,
        },
    ]

    normalized = _run_normalize_tree_roots(nodes)
    normalized_by_id = {node["id"]: node for node in normalized}

    assert nodes[0]["parent_id"] is None
    assert nodes[3]["parent_id"] is None

    root_ids = [node["id"] for node in normalized if not node.get("parent_id")]
    assert root_ids == ["___virtual_root___"]
    assert normalized_by_id["___virtual_root___"]["isVirtual"] is True
    assert normalized_by_id["___unified_root___"]["parent_id"] == "___virtual_root___"
    assert normalized_by_id["spawned_root"]["parent_id"] == "___virtual_root___"
    assert normalized_by_id["spawned_child"]["parent_id"] == "spawned_root"


def test_normalize_tree_roots_leaves_single_root_trees_unchanged():
    nodes = [
        {
            "id": "root",
            "parent_id": None,
            "agent_name": "root",
            "generation": 0,
        },
        {
            "id": "child",
            "parent_id": "root",
            "agent_name": "child",
            "generation": 1,
        },
    ]

    normalized = _run_normalize_tree_roots(nodes)

    assert normalized == nodes
