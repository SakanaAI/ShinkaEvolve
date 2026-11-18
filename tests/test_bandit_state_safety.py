"""Bandit state must survive non-finite inputs and never execute a state file."""

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from shinka.llm.bandit_state import BanditStateError, read_bandit_state
from shinka.llm.prioritization import AsymmetricUCB, FixedSampler, ThompsonSampler

ARMS = ["model-a", "model-b", "model-c"]
LEARNING_BANDITS = [AsymmetricUCB, ThompsonSampler]
ALL_BANDITS = [*LEARNING_BANDITS, FixedSampler]
NON_FINITE = [float("nan"), float("inf"), float("-inf")]
LINEAR_MODE = dict(
    shift_by_baseline=False, shift_by_parent=False, asymmetric_scaling=False
)


def _trained(cls, **kwargs):
    bandit = cls(arm_names=ARMS, seed=0, **kwargs)
    bandit.set_baseline_score(0.25)
    for arm, reward in zip(ARMS * 2, [1.0, 0.4, 0.7, 0.9, 0.5, 0.6]):
        bandit.update_submitted(arm)
        bandit.update(arm, reward, baseline=0.3)
        bandit.update_cost(arm, 0.01)
    return bandit


def _assert_same_state(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if key == "arm_names":
            assert actual[key] == value
        else:
            np.testing.assert_array_equal(np.asarray(actual[key]), np.asarray(value))


def _assert_selects(bandit) -> None:
    for _ in range(10):
        one_hot, probabilities = bandit.select_llm()
        assert one_hot.sum() == 1.0
        assert np.isfinite(probabilities).all()
        assert np.isclose(probabilities.sum(), 1.0)


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize("mode", [{}, LINEAR_MODE], ids=["exponential", "linear"])
@pytest.mark.parametrize("cls", LEARNING_BANDITS)
def test_non_finite_reward_counts_as_a_failed_proposal(cls, mode, bad):
    bandit = _trained(cls, **mode)
    expected = _trained(cls, **mode)

    bandit.update("model-a", bad, baseline=0.3)
    expected.update("model-a", None, baseline=0.3)

    _assert_same_state(bandit.get_state(), expected.get_state())
    _assert_selects(bandit)


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize("cls", LEARNING_BANDITS)
def test_non_finite_parent_baseline_falls_back_to_the_run_baseline(cls, bad):
    bandit = _trained(cls)
    expected = _trained(cls)

    bandit.update("model-a", 0.8, baseline=bad)
    expected.update("model-a", 0.8, baseline=0.25)

    _assert_same_state(bandit.get_state(), expected.get_state())
    _assert_selects(bandit)


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize("cls", ALL_BANDITS)
def test_non_finite_cost_and_baseline_score_are_ignored(cls, bad):
    bandit = _trained(cls, **({"cost_aware_coef": 0.5} if cls is AsymmetricUCB else {}))
    before = bandit.get_state()

    bandit.update_cost("model-a", bad)
    bandit.set_baseline_score(bad)

    _assert_same_state(bandit.get_state(), before)
    _assert_selects(bandit)


@pytest.mark.parametrize("mode", [{}, LINEAR_MODE], ids=["exponential", "linear"])
@pytest.mark.parametrize("cls", LEARNING_BANDITS)
def test_state_poisoned_by_an_older_run_is_reset_on_load(cls, mode, tmp_path, caplog):
    poisoned = _trained(cls, **mode).get_state()
    poisoned["s"][0] = np.nan
    poisoned["baseline"] = float("nan")
    if "total_costs" in poisoned:
        poisoned["total_costs"][1] = np.nan
    path = tmp_path / "bandit_state.pkl"
    with open(path, "wb") as handle:
        pickle.dump(poisoned, handle)

    bandit = cls(arm_names=ARMS, seed=0, **mode)
    with caplog.at_level(logging.WARNING, logger="shinka.llm.prioritization"):
        bandit.load_state(path)

    assert "Resetting reward statistics" in caplog.text
    _assert_same_state(
        bandit.get_state(), cls(arm_names=ARMS, seed=0, **mode).get_state()
    )
    _assert_selects(bandit)
    for arm in ARMS:
        bandit.update_submitted(arm)
        bandit.update(arm, 0.5, baseline=0.1)
    _assert_selects(bandit)


@pytest.mark.parametrize("mode", [{}, LINEAR_MODE], ids=["exponential", "linear"])
@pytest.mark.parametrize("cls", ALL_BANDITS)
def test_state_round_trips_as_json(cls, mode, tmp_path):
    source = _trained(cls, **(mode if cls is not FixedSampler else {}))
    path = tmp_path / "bandit_state.pkl"

    source.save_state(path)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["format"] == "shinka-bandit-state"
    assert document["state"]["arm_names"] == ARMS
    target = cls(arm_names=ARMS, seed=0, **(mode if cls is not FixedSampler else {}))
    target.load_state(path)
    _assert_same_state(target.get_state(), source.get_state())


# Python 3 has only ever defaulted to pickle protocols 3 and above.
@pytest.mark.parametrize("protocol", range(3, pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("cls", ALL_BANDITS)
def test_legacy_pickle_state_still_loads_then_becomes_json(cls, protocol, tmp_path):
    source = _trained(cls)
    legacy = source.get_state()
    legacy["baseline"] = np.float64(legacy["baseline"])
    path = tmp_path / "bandit_state.pkl"
    with open(path, "wb") as handle:
        pickle.dump(legacy, handle, protocol=protocol)

    target = cls(arm_names=ARMS, seed=0)
    target.load_state(path)

    _assert_same_state(target.get_state(), source.get_state())
    target.save_state(path)
    assert path.read_bytes().startswith(b"{")


@pytest.mark.filterwarnings("ignore:numpy.core:DeprecationWarning")
def test_legacy_pickle_written_by_numpy_1_still_loads(tmp_path):
    source = _trained(AsymmetricUCB)
    numpy_2_pickle = pickle.dumps(source.get_state(), protocol=3)
    assert b"numpy._core.multiarray\n" in numpy_2_pickle
    path = tmp_path / "bandit_state.pkl"
    path.write_bytes(
        numpy_2_pickle.replace(b"numpy._core.multiarray\n", b"numpy.core.multiarray\n")
    )

    target = AsymmetricUCB(arm_names=ARMS, seed=0)
    target.load_state(path)

    _assert_same_state(target.get_state(), source.get_state())


class _RunsCommandOnLoad:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def test_pickle_that_would_run_code_is_refused(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "bandit_state.pkl"
    state = _trained(AsymmetricUCB).get_state()
    state["baseline"] = _RunsCommandOnLoad(marker)
    with open(path, "wb") as handle:
        pickle.dump(state, handle)

    bandit = AsymmetricUCB(arm_names=ARMS, seed=0)
    untouched = bandit.get_state()
    with pytest.raises(BanditStateError, match="not allowed"):
        bandit.load_state(path)

    assert not marker.exists()
    _assert_same_state(bandit.get_state(), untouched)


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not a state file",
        b'{"format": "something-else", "version": 1, "state": {}}',
        b'{"format": "shinka-bandit-state", "version": 1, "state": {"s": [[1.0]]}}',
        b'{"format": "shinka-bandit-state", "version": 1, "state": {"s": {"a": 1}}}',
        pickle.dumps(["not", "a", "dict"]),
        pickle.dumps({"s": np.array([object()], dtype=object)}),
    ],
)
def test_malformed_state_files_raise_a_state_error(content, tmp_path):
    path = tmp_path / "bandit_state.pkl"
    path.write_bytes(content)

    with pytest.raises(BanditStateError):
        read_bandit_state(path)
