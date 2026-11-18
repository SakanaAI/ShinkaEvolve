"""Bandit state must survive non-finite inputs and never execute a state file."""

import json
import logging
import os
import pickle
import subprocess
import sys
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


class _Reduced:
    """Pickles as a call of `function(*arguments)`, then `__setstate__(state)`."""

    def __init__(self, function, arguments, state=None):
        self.reduced = (function, arguments, *(() if state is None else (state,)))

    def __reduce__(self):
        return self.reduced


def _numpy_pickles_that_read_foreign_memory() -> dict[str, object]:
    """Arrays whose bytes NumPy would follow as object pointers, or read past.

    Every one is built from the constructors a real array pickle names, so an
    unpickler that only restricts which globals may be called still runs them.
    """
    from numpy._core import multiarray, numeric

    pointer = b"\x41" * 8
    from_buffer = getattr(numeric, "_frombuffer")  # private; what protocol 5 names
    empty_array = (np.ndarray, (0,), b"b")

    def object_field_dtype(offset=0):
        fields = {"a": (np.dtype("O") if offset == 0 else np.dtype("f8"), offset)}
        return _Reduced(
            np.dtype, ("V8", False, True), (3, "|", None, ("a",), fields, 8, 1, 0)
        )

    def rebuilt(dtype, data=pointer):
        return _Reduced(
            multiarray._reconstruct, empty_array, (1, (1,), dtype, False, data)
        )

    return {
        "object array over raw bytes": _Reduced(np.ndarray, ((1,), "O", pointer)),
        "object array as a shape": _Reduced(
            np.ndarray, (_Reduced(np.ndarray, ((1,), "O", pointer)), "f8")
        ),
        "object dtype in array state": rebuilt(np.dtype("O")),
        "object field in array state": rebuilt(np.dtype([("a", "O")])),
        "hidden object field": rebuilt(object_field_dtype()),
        "hidden object field from a buffer": _Reduced(
            from_buffer, (pointer, object_field_dtype(), (1,), "C")
        ),
        "hidden object field in a scalar": _Reduced(
            multiarray.scalar, (object_field_dtype(), pointer)
        ),
        "field beyond the item": rebuilt(object_field_dtype(offset=1 << 30)),
        "shape larger than the data": rebuilt(np.dtype("f8"), b"\x00" * 4),
    }


def test_crafted_numpy_pickles_are_refused_without_touching_their_memory(tmp_path):
    """Run in a child process: reading such an array used to end in a segfault."""
    paths = []
    for index, value in enumerate(_numpy_pickles_that_read_foreign_memory().values()):
        path = tmp_path / f"crafted_{index}.pkl"
        path.write_bytes(pickle.dumps({"s": value}, protocol=4))
        paths.append(str(path))
    reader = (
        "import sys\n"
        "from shinka.llm.bandit_state import BanditStateError, read_bandit_state\n"
        "from pathlib import Path\n"
        "for path in sys.argv[1:]:\n"
        "    try:\n"
        "        read_bandit_state(Path(path))\n"
        "        print('loaded')\n"
        "    except BanditStateError:\n"
        "        print('refused')\n"
    )

    child = subprocess.run(
        [sys.executable, "-c", reader, *paths],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert child.returncode == 0, child.stderr[-2000:]
    assert child.stdout.split() == ["refused"] * len(paths)


@pytest.mark.parametrize("protocol", range(3, pickle.HIGHEST_PROTOCOL + 1))
def test_legacy_reader_returns_what_the_pickle_held(protocol, tmp_path):
    state = {
        "arm_names": ["model-a", "model-b"],
        "s": np.array([0.5, -np.inf, 1e300]),
        "counts": np.arange(3),
        "flags": np.array([True, False]),
        "small": np.array([1.5, 2.5], dtype=np.float32),
        "empty": np.zeros(0),
        "baseline": np.float64(0.25),
        "single": np.float32(1.5),
        "seen": np.int64(7),
        "active": np.bool_(True),
        "obs_max": float("-inf"),
        "plain_int": 3,
        "large_int": 2**40,
        "plain_bool": False,
        "missing": None,
    }
    path = tmp_path / "bandit_state.pkl"
    path.write_bytes(pickle.dumps(state, protocol=protocol))

    restored = read_bandit_state(path)

    assert restored.keys() == state.keys()
    for key, value in state.items():
        expected = value.tolist() if isinstance(value, (np.generic, np.ndarray)) else value
        assert restored[key] == expected, key
        assert type(restored[key]) is type(expected), key


@pytest.mark.parametrize(
    "state",
    [
        {"s": np.zeros((2, 2))},
        {"s": np.zeros(2, dtype=">f8")},
        {"s": np.zeros(2, dtype="c16")},
        {"s": np.array(["a"])},
        {"s": {"nested": 1.0}},
        {1: np.zeros(2)},
    ],
    ids=["matrix", "big-endian", "complex", "strings", "nested-dict", "int-key"],
)
def test_legacy_reader_refuses_shapes_a_bandit_state_never_has(state, tmp_path):
    path = tmp_path / "bandit_state.pkl"
    path.write_bytes(pickle.dumps(state, protocol=4))

    with pytest.raises(BanditStateError):
        read_bandit_state(path)


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
