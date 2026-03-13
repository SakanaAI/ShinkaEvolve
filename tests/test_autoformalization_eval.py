import importlib
import sys
import types

import pytest


@pytest.fixture
def autoformalization_eval(monkeypatch):
    lean_interact = types.ModuleType("lean_interact")
    lean_interact.LeanREPLConfig = object
    lean_interact.AutoLeanServer = object
    lean_interact.Command = object
    lean_interact.FileCommand = object
    lean_interact.TempRequireProject = object

    interface = types.ModuleType("lean_interact.interface")
    interface.BaseREPLResponse = object
    interface.Command = object
    interface.FileCommand = object
    interface.LeanError = RuntimeError

    utils = types.ModuleType("lean_interact.utils")
    utils.remove_lean_comments = lambda text: text

    monkeypatch.setitem(sys.modules, "lean_interact", lean_interact)
    monkeypatch.setitem(sys.modules, "lean_interact.interface", interface)
    monkeypatch.setitem(sys.modules, "lean_interact.utils", utils)

    sys.modules.pop("examples.autoformalization.utils_lean", None)
    sys.modules.pop("examples.autoformalization.evaluate", None)

    module = importlib.import_module("examples.autoformalization.evaluate")
    return importlib.reload(module)


def test_validate_task_semantics_rejects_vacuous_theorem(autoformalization_eval):
    ok, error = autoformalization_eval.validate_task_semantics(
        "import mathlib\n\ntheorem abelian_group (G : Type*) [Group G] : True := by trivial\n"
    )

    assert ok is False
    assert "vacuous" in error


def test_validate_task_semantics_accepts_target_shaped_theorem(autoformalization_eval):
    theorem = """
import mathlib

theorem abelian_group {G : Type*} [Group G] (a b : G) :
  (a * b = b * a) ->
  let H := Subgroup.closure ({a, b} : Set G) in
  forall x y : G, x ∈ H -> y ∈ H -> x * y = y * x
:= by
  intro h
  sorry
"""
    ok, error = autoformalization_eval.validate_task_semantics(theorem)

    assert ok is True
    assert error is None


def test_validate_proof_short_circuits_semantic_failures(
    autoformalization_eval,
    monkeypatch,
    tmp_path,
):
    program_path = tmp_path / "initial.lean"
    program_path.write_text(
        "theorem abelian_group () :=\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        autoformalization_eval,
        "validate_lean",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    ok, error = autoformalization_eval.validate_proof(
        (
            str(program_path),
            "theorem abelian_group (G : Type*) [Group G] : True := by trivial",
        )
    )

    assert ok is False
    assert "evolved Lean file" in error


def test_aggregate_metrics_zeroes_semantically_invalid_outputs(
    autoformalization_eval,
    tmp_path,
):
    program_path = tmp_path / "main.lean"
    program_path.write_text("theorem abelian_group () :=", encoding="utf-8")

    metrics = autoformalization_eval.aggregate_hypothesis_generation_metrics(
        (
            str(program_path),
            "theorem abelian_group (G : Type*) [Group G] : True := by trivial",
        ),
        str(tmp_path),
    )

    assert metrics["combined_score"] == 0.0
    assert "candidate_semantic_error" in metrics["private"]


def test_aggregate_metrics_scores_only_valid_target_theorems(
    autoformalization_eval,
    tmp_path,
    monkeypatch,
):
    class FakeResponse:
        messages = []

        @staticmethod
        def lean_code_is_valid(allow_sorry: bool = False):
            return True

    program_path = tmp_path / "main.lean"
    program_text = """theorem abelian_group {G : Type*} [Group G] (a b : G) :
  (a * b = b * a) ->
  let H := Subgroup.closure ({a, b} : Set G) in
  forall x y : G, x ∈ H -> y ∈ H -> x * y = y * x
:="""
    program_path.write_text(program_text, encoding="utf-8")

    monkeypatch.setattr(autoformalization_eval, "check_lean", lambda *args, **kwargs: FakeResponse())

    metrics = autoformalization_eval.aggregate_hypothesis_generation_metrics(
        (
            str(program_path),
            """import mathlib

theorem abelian_group {G : Type*} [Group G] (a b : G) :
  (a * b = b * a) ->
  let H := Subgroup.closure ({a, b} : Set G) in
  forall x y : G, x ∈ H -> y ∈ H -> x * y = y * x
:= by
  intro h
  sorry""",
        ),
        str(tmp_path),
    )

    assert metrics["combined_score"] > 0.0
    assert metrics["public"]["formalization_length"] == len(program_text)


def test_aggregate_metrics_zeroes_repaired_proofs_when_candidate_drifts(
    autoformalization_eval,
    tmp_path,
):
    program_path = tmp_path / "main.lean"
    program_path.write_text("theorem abelian_group () :=\n", encoding="utf-8")

    proof_text = """import mathlib

theorem abelian_group {G : Type*} [Group G] (a b : G) :
  (a * b = b * a) ->
  let H := Subgroup.closure ({a, b} : Set G) in
  forall x y : G, x ∈ H -> y ∈ H -> x * y = y * x
:= by
  intro h
  sorry"""

    metrics = autoformalization_eval.aggregate_hypothesis_generation_metrics(
        (str(program_path), proof_text),
        str(tmp_path),
    )

    assert metrics["combined_score"] == 0.0
    assert metrics["private"]["candidate_semantic_error"] is not None
