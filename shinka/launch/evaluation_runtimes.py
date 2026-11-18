"""How long local evaluations have taken, the basis of the adaptive hang limit."""

import threading
from typing import Any, Dict, Optional


def reports_correct(results: Optional[Dict[str, Any]]) -> bool:
    correct = (results or {}).get("correct")
    return isinstance(correct, dict) and correct.get("correct") is True


class LongestRuntimes:
    """Seconds the longest local evaluations took; None until there is one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # With a correct result. Failed evaluations are left out: a crash after
        # one second says nothing about how long a healthy one needs.
        self.successful: Optional[float] = None
        # Whatever the outcome: crashes, incorrect results and evaluations that
        # were killed count. It is what there is to go by before any success.
        self.finished: Optional[float] = None

    def note_successful(self, seconds: float) -> None:
        with self._lock:
            self.successful = max(seconds, self.successful or 0.0)

    def note_finished(self, seconds: float) -> None:
        with self._lock:
            self.finished = max(seconds, self.finished or 0.0)
