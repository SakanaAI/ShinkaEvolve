"""Shared event utilities for agent backends."""

from typing import Any, Dict, Optional


def extract_session_id(event: Dict[str, Any]) -> Optional[str]:
    """Extract session/thread ID from an agent event payload.

    Handles multiple event formats from different agent backends:
    - thread.* events with thread_id (Codex CLI format)
    - Direct session_id field (ShinkaAgent/Claude format)
    - Nested session.id or session.session_id objects

    Args:
        event: Event dictionary from agent backend.

    Returns:
        Session ID string if found, None otherwise.
    """
    if not isinstance(event, dict):
        return None

    # Thread events (Codex CLI format)
    event_type = event.get("type")
    if isinstance(event_type, str) and event_type.startswith("thread."):
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            return thread_id

    # Direct session_id field (ShinkaAgent/Claude format)
    session_id = event.get("session_id")
    if isinstance(session_id, str) and session_id:
        return session_id

    # Nested session object
    session_obj = event.get("session")
    if isinstance(session_obj, dict):
        candidate = session_obj.get("id") or session_obj.get("session_id")
        if isinstance(candidate, str) and candidate:
            return candidate

    return None
