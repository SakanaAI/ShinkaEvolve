from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Protocol


class AgentRunner(Protocol):
    """Protocol for an agent runner that executes a prompt in a workspace."""

    def __call__(
        self,
        user_prompt: str,
        workdir: Path,
        *,
        system_prompt: Optional[str] = None,
        profile: Optional[str],
        sandbox: str,
        approval_mode: str,
        max_seconds: int,
        max_events: int,
        extra_cli_config: Dict[str, Any],
        codex_path: Optional[str] = None,
        resume_session_id: Optional[str] = None,
        session_kind: str = "unknown",
    ) -> Iterator[Dict[str, Any]]: ...
