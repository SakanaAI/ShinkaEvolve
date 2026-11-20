"""The bandit state file is JSON that any parser reads, not only Python's."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shinka.llm.bandit_state import (
    BanditStateError,
    read_bandit_state,
    write_bandit_state,
)
from shinka.llm.prioritization import AsymmetricUCB, FixedSampler, ThompsonSampler

ARMS = ["model-a", "model-b"]
LINEAR_MODE = dict(
    shift_by_baseline=False, shift_by_parent=False, asymmetric_scaling=False
)
BANDITS = {
    "ucb": lambda: AsymmetricUCB(arm_names=ARMS, seed=0, cost_aware_coef=0.5),
    "ucb-linear": lambda: AsymmetricUCB(arm_names=ARMS, seed=0, **LINEAR_MODE),
    "thompson": lambda: ThompsonSampler(arm_names=ARMS, seed=0),
    "thompson-linear": lambda: ThompsonSampler(arm_names=ARMS, seed=0, **LINEAR_MODE),
    "fixed": lambda: FixedSampler(arm_names=ARMS, seed=0),
}
ALL_BANDITS = pytest.mark.parametrize("make_bandit", BANDITS.values(), ids=BANDITS)

# Files as Shinka wrote them before non-finite numbers became strings (commit
# 1bf7113): -Infinity and Infinity are bare tokens, which only Python reads.
STATE_FILES_WITH_BARE_INFINITY = {
    "ucb": (
        '{"format": "shinka-bandit-state", "version": 1, "state": {"arm_names": '
        '["model-a", "model-b"], "n_submitted": [1.0, 1.0], "n_completed": [1.0, '
        '0.0], "s": [-1.3240862212527813, -Infinity], "divs": [0.95, 0.0], '
        '"baseline": 0.25, "obs_max": -1.2727929268652307, "obs_min": -Infinity, '
        '"n_costs": [1.0, 0.0], "total_costs": [0.125, 0.0], "min_cost_observed": '
        '0.125, "max_cost_observed": 0.125}}'
    ),
    "ucb-linear": (
        '{"format": "shinka-bandit-state", "version": 1, "state": {"arm_names": '
        '["model-a", "model-b"], "n_submitted": [0.0, 0.0], "n_completed": [0.0, '
        '0.0], "s": [0.0, 0.0], "divs": [0.0, 0.0], "baseline": 0.25, "obs_max": '
        '-Infinity, "obs_min": Infinity, "n_costs": [0.0, 0.0], "total_costs": '
        '[0.0, 0.0], "min_cost_observed": Infinity, "max_cost_observed": -Infinity}}'
    ),
    "thompson": (
        '{"format": "shinka-bandit-state", "version": 1, "state": {"arm_names": '
        '["model-a", "model-b"], "n_submitted": [1.0, 1.0], "n_completed": [1.0, '
        '0.0], "s": [-1.3240862212527813, -Infinity], "divs": [0.95, 0.0], "alpha": '
        '[1.95, 1.0], "beta": [1.0, 1.0], "baseline": 0.25, "obs_max": '
        '-1.2727929268652307, "obs_min": -Infinity}}'
    ),
}


def _strict_json(text: str) -> Any:
    """Parse as a strict parser does: no NaN, Infinity or -Infinity tokens."""

    def reject(token: str) -> Any:
        raise ValueError(f"{token} is not JSON")

    return json.loads(text, parse_constant=reject)


def _observe(bandit: Any) -> Any:
    bandit.set_baseline_score(0.25)
    bandit.update_submitted("model-a")
    bandit.update("model-a", 0.75, baseline=0.5)
    bandit.update_cost("model-a", 0.125)
    bandit.update_submitted("model-b")
    return bandit


def _assert_same_state(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if key == "arm_names":
            assert actual[key] == value
        else:
            np.testing.assert_array_equal(np.asarray(actual[key]), np.asarray(value))


@pytest.mark.parametrize("observe", [lambda bandit: bandit, _observe], ids=["new", "used"])
@ALL_BANDITS
def test_saved_state_is_strict_json_and_loads_unchanged(make_bandit, observe, tmp_path):
    source = observe(make_bandit())
    path = tmp_path / "bandit_state.pkl"

    source.save_state(path)

    document = _strict_json(path.read_text(encoding="utf-8"))
    assert document["format"] == "shinka-bandit-state"
    assert document["version"] == 2
    target = make_bandit()
    target.load_state(path)
    _assert_same_state(target.get_state(), source.get_state())


def test_non_finite_numbers_are_written_as_strings(tmp_path: Path) -> None:
    path = tmp_path / "bandit_state.pkl"
    state = {
        "arm_names": ["a", "b", "c"],
        "s": np.array([0.5, -np.inf, np.inf]),
        "all_non_finite": [float("-inf"), float("-inf")],
        "obs_min": float("-inf"),
        "obs_max": np.float64("inf"),
        "broken": float("nan"),
        "baseline": 0.25,
    }

    write_bandit_state(path, state)

    saved = _strict_json(path.read_text(encoding="utf-8"))["state"]
    assert saved["s"] == [0.5, "-Infinity", "Infinity"]
    assert saved["all_non_finite"] == ["-Infinity", "-Infinity"]
    assert saved["obs_min"] == "-Infinity"
    assert saved["obs_max"] == "Infinity"
    assert saved["broken"] == "NaN"
    restored = read_bandit_state(path)
    assert restored["arm_names"] == ["a", "b", "c"]
    assert restored["s"] == [0.5, -math.inf, math.inf]
    assert restored["all_non_finite"] == [-math.inf, -math.inf]
    assert restored["obs_min"] == -math.inf
    assert restored["obs_max"] == math.inf
    assert math.isnan(restored["broken"])
    assert restored["baseline"] == 0.25


@pytest.mark.parametrize("kind", STATE_FILES_WITH_BARE_INFINITY)
def test_state_file_with_bare_infinity_tokens_still_loads(kind: str, tmp_path: Path) -> None:
    text = STATE_FILES_WITH_BARE_INFINITY[kind]
    with pytest.raises(ValueError, match="is not JSON"):
        _strict_json(text)
    path = tmp_path / "bandit_state.pkl"
    path.write_text(text, encoding="utf-8")

    bandit = BANDITS[kind]()
    bandit.load_state(path)

    state = bandit.get_state()
    expected = json.loads(text)["state"]
    for key, value in expected.items():
        if key == "arm_names":
            assert state[key] == value
        else:
            np.testing.assert_array_equal(np.asarray(state[key]), np.asarray(value))
    # The next save rewrites it as strict JSON.
    bandit.save_state(path)
    _strict_json(path.read_text(encoding="utf-8"))
    reloaded = BANDITS[kind]()
    reloaded.load_state(path)
    _assert_same_state(reloaded.get_state(), state)


def test_arm_named_like_a_non_finite_number_stays_a_name(tmp_path: Path) -> None:
    path = tmp_path / "bandit_state.pkl"

    write_bandit_state(path, {"arm_names": ["NaN", "Infinity"], "p": [0.5, 0.5]})

    assert read_bandit_state(path)["arm_names"] == ["NaN", "Infinity"]


@pytest.mark.parametrize(
    "state",
    ['{"s": [1.0, "minus infinity"]}', '{"obs_max": "huge"}', '{"s": [1.0, "model-a"]}'],
)
def test_other_strings_in_numeric_fields_are_refused(state: str, tmp_path: Path) -> None:
    path = tmp_path / "bandit_state.pkl"
    path.write_text(
        '{"format": "shinka-bandit-state", "version": 2, "state": ' + state + "}",
        encoding="utf-8",
    )

    with pytest.raises(BanditStateError):
        read_bandit_state(path)


def test_state_of_a_later_format_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bandit_state.pkl"
    path.write_text(
        '{"format": "shinka-bandit-state", "version": 3, "state": {}}', encoding="utf-8"
    )

    with pytest.raises(BanditStateError, match="unknown format or version"):
        read_bandit_state(path)
