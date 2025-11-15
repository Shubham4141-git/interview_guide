"""Service wrapper for running the LangGraph supervisor graph."""

from __future__ import annotations

from typing import Any, Dict, Optional

from interview_guide.graph import build_graph

_GRAPH_APP = build_graph()


def execute_agent(
    query: str,
    user_id: Optional[str] = None,
    session_id: Optional[int] = None,
    intent: Optional[str] = None,
    slots: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the orchestrated agent flow and return the resulting state."""
    state: Dict[str, Any] = {"query": query}
    if user_id:
        state["user_id"] = user_id
    if session_id is not None:
        state["session_id"] = session_id
    if intent:
        state["intent"] = intent
    if slots:
        state["slots"] = slots

    return _GRAPH_APP.invoke(state)
