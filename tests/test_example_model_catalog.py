from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import pytest
import yaml  # type: ignore[import-untyped]
from yaml.nodes import (  # type: ignore[import-untyped]
    MappingNode,
    Node,
    ScalarNode,
    SequenceNode,
)

from shinka.embed.client import resolve_embedding_backend
from shinka.embed.providers.pricing import get_model_price
from shinka.llm.providers.model_resolver import resolve_model_backend
from shinka.llm.providers.pricing import get_model_prices
from shinka.pricing import PricingConfig, PricingMode, refresh_model_catalog


REPO_ROOT = Path(__file__).parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
TUTORIAL_PATH = EXAMPLES_DIR / "shinka_tutorial.ipynb"
MAINTAINED_CONFIG_DIRS = (
    REPO_ROOT / "shinka" / "configs",
    REPO_ROOT / "skills" / "shinka-setup" / "scripts",
    REPO_ROOT / "skills" / "shinka-convert" / "scripts",
)

LLM_FIELDS = {
    "llm_judge_names",
    "llm_models",
    "meta_llm_models",
    "model_names",
    "novelty_llm_models",
}
EMBEDDING_FIELDS = {"embedding_model"}

# These backends report their own costs or are intentionally unpriced. Any new
# exemption must be added explicitly instead of silently accepting a missing price.
EXTERNALLY_PRICED_PREFIXES = {
    "headless/": "headless",
    "local/": "local_openai",
}

STALE_MODEL_REPLACEMENTS = {
    "claude-4-6-sonnet": "claude-sonnet-4-6",
    "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0": (
        "us.anthropic.claude-sonnet-4-6-v1:0"
    ),
    "us.anthropic.claude-sonnet-4-20250514-v1:0": (
        "us.anthropic.claude-sonnet-4-6-v1:0"
    ),
}


@dataclass(frozen=True)
class ModelReference:
    model_name: str
    kind: Literal["llm", "embedding"]
    source: str

    @property
    def test_id(self) -> str:
        return f"{self.source}-{self.model_name}"


def _field_kind(field_name: str) -> Literal["llm", "embedding"] | None:
    normalized_name = field_name.rsplit(".", 1)[-1]
    if normalized_name in LLM_FIELDS:
        return "llm"
    if normalized_name in EMBEDDING_FIELDS:
        return "embedding"
    return None


def _literal_strings(node: ast.AST) -> Iterator[tuple[str, int]]:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child.value, child.lineno


class _PythonModelVisitor(ast.NodeVisitor):
    def __init__(self, source_label: str) -> None:
        self.source_label = source_label
        self.references: set[ModelReference] = set()

    def _record(self, node: ast.AST, field_name: str) -> None:
        kind = _field_kind(field_name)
        if kind is None:
            return
        for model_name, line_number in _literal_strings(node):
            self.references.add(
                ModelReference(
                    model_name=model_name,
                    kind=kind,
                    source=f"{self.source_label}:{line_number}",
                )
            )

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._record(node.value, target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._record(node.value, node.target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        for keyword in node.keywords:
            if keyword.arg is not None:
                self._record(keyword.value, keyword.arg)

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
        ):
            for argument in node.args:
                self._record(argument, node.func.value.id)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:  # noqa: N802
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                self._record(value, key.value)
        self.generic_visit(node)

    def _record_function_defaults(self, arguments: ast.arguments) -> None:
        positional_args = [*arguments.posonlyargs, *arguments.args]
        default_args = positional_args[-len(arguments.defaults) :]
        for argument, positional_default in zip(default_args, arguments.defaults):
            self._record(positional_default, argument.arg)
        for argument, keyword_default in zip(
            arguments.kwonlyargs, arguments.kw_defaults
        ):
            if keyword_default is not None:
                self._record(keyword_default, argument.arg)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._record_function_defaults(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._record_function_defaults(node.args)
        self.generic_visit(node)


def _python_references(path: Path) -> Iterator[ModelReference]:
    relative_path = path.relative_to(REPO_ROOT)
    visitor = _PythonModelVisitor(str(relative_path))
    visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    yield from visitor.references


def _notebook_references(path: Path) -> Iterator[ModelReference]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    relative_path = path.relative_to(REPO_ROOT)
    for cell_number, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        visitor = _PythonModelVisitor(f"{relative_path}:cell-{cell_number}")
        visitor.visit(ast.parse(source, filename=str(path)))
        yield from visitor.references


def _yaml_references(path: Path) -> Iterator[ModelReference]:
    root = yaml.compose(path.read_text(encoding="utf-8"))
    if root is None:
        return
    relative_path = path.relative_to(REPO_ROOT)
    yield from _walk_yaml(root, str(relative_path))


def _walk_yaml(node: Node, source_label: str) -> Iterator[ModelReference]:
    if isinstance(node, MappingNode):
        for key, value in node.value:
            if isinstance(key, ScalarNode):
                kind = _field_kind(key.value)
                if kind is not None:
                    for scalar in _yaml_scalars(value):
                        yield ModelReference(
                            model_name=scalar.value,
                            kind=kind,
                            source=f"{source_label}:{scalar.start_mark.line + 1}",
                        )
            yield from _walk_yaml(value, source_label)
    elif isinstance(node, SequenceNode):
        for child in node.value:
            yield from _walk_yaml(child, source_label)


def _yaml_scalars(node: Node) -> Iterator[ScalarNode]:
    if isinstance(node, ScalarNode):
        if node.tag.endswith(":str") and node.value:
            yield node
    elif isinstance(node, SequenceNode):
        for child in node.value:
            yield from _yaml_scalars(child)


def _collect_example_model_references() -> tuple[ModelReference, ...]:
    references: set[ModelReference] = set()
    for path in EXAMPLES_DIR.rglob("*.py"):
        relative_parts = path.relative_to(EXAMPLES_DIR).parts
        if any(part.startswith("results") for part in relative_parts):
            continue
        references.update(_python_references(path))
    for pattern in ("*.yaml", "*.yml"):
        for path in EXAMPLES_DIR.rglob(pattern):
            references.update(_yaml_references(path))
        for config_dir in MAINTAINED_CONFIG_DIRS:
            for path in config_dir.rglob(pattern):
                references.update(_yaml_references(path))
    references.update(_notebook_references(TUTORIAL_PATH))
    return tuple(
        sorted(references, key=lambda ref: (ref.source, ref.kind, ref.model_name))
    )


EXAMPLE_MODEL_REFERENCES = _collect_example_model_references()


@pytest.fixture(scope="module", autouse=True)
def _use_bundled_catalog(tmp_path_factory: pytest.TempPathFactory) -> None:
    refresh_model_catalog(
        PricingConfig(
            mode=PricingMode.OFFLINE,
            cache_dir=tmp_path_factory.mktemp("pricing-cache"),
        )
    )


def test_example_model_collector_covers_all_executable_formats() -> None:
    source_suffixes = {
        Path(ref.source.split(":", 1)[0]).suffix for ref in EXAMPLE_MODEL_REFERENCES
    }
    assert source_suffixes == {".ipynb", ".py", ".yaml"}


@pytest.mark.parametrize(
    "reference",
    EXAMPLE_MODEL_REFERENCES,
    ids=lambda reference: reference.test_id,
)
def test_example_model_literals_are_current(reference: ModelReference) -> None:
    replacement = STALE_MODEL_REPLACEMENTS.get(reference.model_name)
    assert replacement is None, (
        f"{reference.source} uses stale model {reference.model_name!r}; "
        f"replace it with {replacement!r}"
    )


@pytest.mark.parametrize(
    "reference",
    EXAMPLE_MODEL_REFERENCES,
    ids=lambda reference: reference.test_id,
)
def test_example_models_resolve_with_finite_pricing(reference: ModelReference) -> None:
    for prefix, expected_provider in EXTERNALLY_PRICED_PREFIXES.items():
        if reference.model_name.startswith(prefix):
            resolved = resolve_model_backend(reference.model_name)
            assert reference.kind == "llm"
            assert resolved.provider == expected_provider
            return

    if reference.kind == "embedding":
        resolved_embedding = resolve_embedding_backend(reference.model_name)
        pricing_model = (
            resolved_embedding.api_model_name
            if resolved_embedding.provider == "azure_openai"
            else reference.model_name
        )
        price = get_model_price(pricing_model)
        assert math.isfinite(price) and price > 0, (
            f"{reference.source} has invalid input pricing for "
            f"{reference.model_name!r}: {price!r}"
        )
        return

    resolved_llm = resolve_model_backend(reference.model_name)
    pricing_model = (
        resolved_llm.api_model_name
        if resolved_llm.provider in {"azure_openai", "openrouter"}
        else reference.model_name
    )
    prices = get_model_prices(pricing_model)
    for price_kind in ("input_price", "output_price"):
        price = prices[price_kind]
        assert math.isfinite(price) and price > 0, (
            f"{reference.source} has invalid {price_kind} for "
            f"{reference.model_name!r}: {price!r}"
        )


def test_tutorial_contains_no_cost_pricing_preflight() -> None:
    notebook = json.loads(TUTORIAL_PATH.read_text(encoding="utf-8"))
    source = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "pricing_snapshot = refresh_model_catalog()" in source
    assert 'print("pricing catalog:", pricing_snapshot.metadata())' in source
    assert "fallback if no keys detected" not in source
