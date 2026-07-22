"""Regression tests for prompt-evolver constructor defaults."""

import pytest

from shinka.core.prompt_evolver import AsyncSystemPromptEvolver, SystemPromptEvolver


@pytest.mark.parametrize("evolver_class", [SystemPromptEvolver, AsyncSystemPromptEvolver])
def test_constructor_accepts_default_patch_types(evolver_class):
    evolver = evolver_class(llm_client=object())

    assert evolver.patch_types == ["diff", "full"]
    assert sum(evolver.patch_type_probs) == pytest.approx(1.0)


@pytest.mark.parametrize("evolver_class", [SystemPromptEvolver, AsyncSystemPromptEvolver])
def test_constructor_rejects_invalid_patch_types(evolver_class):
    with pytest.raises(ValueError, match="Invalid patch type"):
        evolver_class(llm_client=object(), patch_types=["bogus"])
