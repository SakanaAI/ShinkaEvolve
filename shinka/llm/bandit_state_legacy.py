"""Reads the pickle that bandit states were saved as before they became JSON.

Nothing is unpickled. The file is walked opcode by opcode, and only what a
saved bandit state consists of is accepted: a dictionary from names to None,
booleans, numbers, lists of those or of strings, and NumPy scalars or flat
arrays of fixed-size numeric types. Array bytes are decoded with `struct`
after their length has been checked, and no NumPy constructor is ever called,
so a crafted file can neither run code nor have memory read that it does not
contain.

Scheduled for removal. This reader first ships in the release after 0.0.8 and
is to be deleted two releases after that one (0.0.11 if the numbering
continues), together with `_state_from_legacy_pickle` in `bandit_state` and
the legacy-pickle tests in `tests/test_bandit_state_safety.py`. From then on a
pickle at the state path is an unreadable state: the runner warns and starts
with a fresh bandit.
"""

from __future__ import annotations

import pickletools
import struct
from dataclasses import dataclass
from typing import Any, Callable, Optional

# NumPy type codes of a bandit state's arrays and scalars -> struct formats.
_STRUCT_FORMATS = {
    "f8": "d",
    "f4": "f",
    "f2": "e",
    "i8": "q",
    "i4": "i",
    "i2": "h",
    "i1": "b",
    "u8": "Q",
    "u4": "I",
    "u2": "H",
    "u1": "B",
    "b1": "?",
}
# NumPy 1.x pickles name `numpy.core`, NumPy 2.x pickles `numpy._core`.
_NUMPY_CORE = ("numpy.core", "numpy._core")
_DTYPE = ("numpy", "dtype")
_NDARRAY = ("numpy", "ndarray")
_RECONSTRUCT = tuple((f"{core}.multiarray", "_reconstruct") for core in _NUMPY_CORE)
_SCALAR = tuple((f"{core}.multiarray", "scalar") for core in _NUMPY_CORE)
_FROMBUFFER = tuple((f"{core}.numeric", "_frombuffer") for core in _NUMPY_CORE)
_KNOWN_GLOBALS = frozenset(
    {_DTYPE, _NDARRAY, *_RECONSTRUCT, *_SCALAR, *_FROMBUFFER}
)
_MARK = object()


class LegacyStateError(ValueError):
    """The file is not a pickle of a bandit state."""


@dataclass(frozen=True)
class _Global:
    """A name the pickle refers to. It is compared, never imported or called."""

    module: str
    name: str

    @property
    def key(self) -> tuple[str, str]:
        return self.module, self.name


@dataclass
class _Dtype:
    code: str
    byte_order_known: bool = False


@dataclass
class _Array:
    """An array between its creation and the opcode that supplies its data."""

    values: Any = None
    filled: bool = False


def read_legacy_state(data: bytes) -> dict[str, Any]:
    """Return the plain values of a pickled bandit state."""
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    for opcode, argument, _ in pickletools.genops(data):
        if opcode.name == "STOP":
            break
        operation = _OPERATIONS.get(opcode.name)
        if operation is None:
            raise LegacyStateError(
                f"bandit state pickle uses {opcode.name}, which is not allowed"
            )
        operation(stack, memo, argument)
    if len(stack) != 1 or not isinstance(stack[0], dict):
        raise LegacyStateError("legacy bandit state is not a dictionary")
    return {name: _finished(value) for name, value in stack[0].items()}


def _finished(value: Any) -> Any:
    if isinstance(value, _Array):
        _require(value.filled, "an array has no data")
        return value.values
    return value


def _require(condition: bool, problem: str) -> None:
    if not condition:
        raise LegacyStateError(f"bandit state pickle is malformed: {problem}")


def _pop(stack: list[Any]) -> Any:
    _require(bool(stack) and stack[-1] is not _MARK, "an operand is missing")
    return stack.pop()


def _pop_to_mark(stack: list[Any]) -> list[Any]:
    items: list[Any] = []
    while True:
        _require(bool(stack), "a mark is missing")
        item = stack.pop()
        if item is _MARK:
            return items[::-1]
        items.append(item)


def _top(stack: list[Any], kind: type) -> Any:
    _require(bool(stack) and isinstance(stack[-1], kind), "an operand has the wrong type")
    return stack[-1]


def _push(value: Any) -> Callable[..., None]:
    return lambda stack, memo, argument: stack.append(value)


def _push_argument(stack: list[Any], memo: dict[int, Any], argument: Any) -> None:
    stack.append(bytes(argument) if isinstance(argument, bytearray) else argument)


def _push_tuple(size: Optional[int]) -> Callable[..., None]:
    def push(stack: list[Any], memo: dict[int, Any], argument: Any) -> None:
        if size is None:
            stack.append(tuple(_pop_to_mark(stack)))
        else:
            stack.append(tuple(reversed([_pop(stack) for _ in range(size)])))

    return push


def _remember(stack: list[Any], memo: dict[int, Any], argument: Any) -> None:
    _require(bool(stack) and stack[-1] is not _MARK, "nothing to remember")
    memo[len(memo) if argument is None else argument] = stack[-1]


def _recall(stack: list[Any], memo: dict[int, Any], argument: Any) -> None:
    _require(argument in memo, "an unknown value is referenced")
    stack.append(memo[argument])


def _append(stack: list[Any], memo: dict[int, Any], argument: Any) -> None:
    item = _pop(stack)
    _top(stack, list).append(item)


def _extend(stack: list[Any], memo: dict[int, Any], argument: Any) -> None:
    items = _pop_to_mark(stack)
    _top(stack, list).extend(items)


def _set_items(marked: bool) -> Callable[..., None]:
    def set_items(stack: list[Any], memo: dict[int, Any], argument: Any) -> None:
        items = _pop_to_mark(stack) if marked else [_pop(stack), _pop(stack)][::-1]
        _require(len(items) % 2 == 0, "a dictionary entry has no value")
        target = _top(stack, dict)
        for name, value in zip(items[::2], items[1::2]):
            _require(isinstance(name, str), "a dictionary key is not a name")
            target[name] = value

    return set_items


def _push_global(stack: list[Any], memo: dict[int, Any], argument: Any) -> None:
    if argument is None:  # STACK_GLOBAL takes both names from the stack.
        name, module = _pop(stack), _pop(stack)
    else:
        module, _, name = argument.partition(" ")
    _require(isinstance(module, str) and isinstance(name, str), "a global is unnamed")
    if (module, name) not in _KNOWN_GLOBALS:
        raise LegacyStateError(
            f"bandit state pickle references {module}.{name}, which is not allowed"
        )
    stack.append(_Global(module, name))


def _reduce(stack: list[Any], memo: dict[int, Any], argument: Any) -> None:
    arguments, function = _pop(stack), _pop(stack)
    _require(
        isinstance(function, _Global) and isinstance(arguments, tuple),
        "a call has no known function",
    )
    if function.key == _DTYPE:
        stack.append(_new_dtype(arguments))
    elif function.key in _RECONSTRUCT:
        _require(
            len(arguments) == 3
            and isinstance(arguments[0], _Global)
            and arguments[0].key == _NDARRAY,
            "an array is rebuilt as another type",
        )
        stack.append(_Array())
    elif function.key in _SCALAR:
        _require(len(arguments) == 2, "a scalar has the wrong arguments")
        stack.append(_decode(arguments[0], arguments[1], shape=()))
    elif function.key in _FROMBUFFER:
        _require(len(arguments) == 4, "a buffer has the wrong arguments")
        data, dtype, shape, _order = arguments
        stack.append(_Array(_decode(dtype, data, shape), filled=True))
    else:
        raise LegacyStateError(
            f"bandit state pickle calls {function.module}.{function.name}, "
            "which is not allowed"
        )


def _build(stack: list[Any], memo: dict[int, Any], argument: Any) -> None:
    state = _pop(stack)
    _require(bool(stack) and isinstance(state, tuple), "an object has no state")
    target = stack[-1]
    if isinstance(target, _Dtype):
        # (version, byte order, subarray, field names, fields, itemsize, alignment, flags)
        _require(
            len(state) == 8
            and state[0] == 3
            and state[1] in ("<", "|")
            and state[2:5] == (None, None, None),
            "a dtype is not a plain little-endian number",
        )
        target.byte_order_known = True
    elif isinstance(target, _Array):
        # (version, shape, dtype, is_fortran, data)
        _require(len(state) == 5 and not target.filled, "an array has the wrong state")
        target.values = _decode(state[2], state[4], state[1])
        target.filled = True
    else:
        raise LegacyStateError("bandit state pickle sets state on an unknown object")


def _new_dtype(arguments: tuple) -> _Dtype:
    _require(
        len(arguments) == 3
        and isinstance(arguments[0], str)
        and arguments[0] in _STRUCT_FORMATS,
        "a dtype is not a fixed-size number",
    )
    return _Dtype(arguments[0])


def _decode(dtype: Any, data: Any, shape: Any) -> Any:
    """Decode the bytes of a scalar (shape `()`) or of a flat array."""
    _require(
        isinstance(dtype, _Dtype) and dtype.byte_order_known,
        "array data has no known dtype",
    )
    _require(isinstance(data, bytes), "array data is not bytes")
    _require(
        isinstance(shape, tuple)
        and len(shape) <= 1
        and all(type(size) is int and size >= 0 for size in shape),
        "an array is not flat",
    )
    count = shape[0] if shape else 1
    item_size = int(dtype.code[1:])
    _require(len(data) == count * item_size, "array data has the wrong length")
    values = list(struct.unpack(f"<{count}{_STRUCT_FORMATS[dtype.code]}", data))
    return values if shape else values[0]


_OPERATIONS: dict[str, Callable[..., None]] = {
    "PROTO": lambda stack, memo, argument: None,
    "FRAME": lambda stack, memo, argument: None,
    "NONE": _push(None),
    "NEWTRUE": _push(True),
    "NEWFALSE": _push(False),
    "MARK": _push(_MARK),
    **{
        name: _push_argument
        for name in (
            "BININT",
            "BININT1",
            "BININT2",
            "LONG1",
            "BINFLOAT",
            "BINUNICODE",
            "SHORT_BINUNICODE",
            "BINUNICODE8",
            "BINBYTES",
            "SHORT_BINBYTES",
            "BINBYTES8",
            "BYTEARRAY8",
        )
    },
    "EMPTY_DICT": lambda stack, memo, argument: stack.append({}),
    "EMPTY_LIST": lambda stack, memo, argument: stack.append([]),
    "EMPTY_TUPLE": _push(()),
    "TUPLE": _push_tuple(None),
    "TUPLE1": _push_tuple(1),
    "TUPLE2": _push_tuple(2),
    "TUPLE3": _push_tuple(3),
    "BINPUT": _remember,
    "LONG_BINPUT": _remember,
    "MEMOIZE": _remember,
    "BINGET": _recall,
    "LONG_BINGET": _recall,
    "APPEND": _append,
    "APPENDS": _extend,
    "SETITEM": _set_items(marked=False),
    "SETITEMS": _set_items(marked=True),
    "GLOBAL": _push_global,
    "STACK_GLOBAL": _push_global,
    "REDUCE": _reduce,
    "BUILD": _build,
}
