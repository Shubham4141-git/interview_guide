"""Schemas for orchestrated agent execution."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User message to route through the supervisor graph.")
    user_id: Optional[str] = Field(None, description="Logical user identifier for personalization.")
    session_id: Optional[int] = Field(None, description="Existing session to continue, if any.")
    intent: Optional[str] = Field(
        None,
        description="Optional override to force a specific intent (primarily for testing or tooling).",
    )
    slots: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional slot overrides passed straight into the agent state.",
    )


class AgentState(BaseModel):
    """Mirrors the graph state returned by the orchestrator."""

    model_config = ConfigDict(extra="allow")

    user_id: Optional[str] = None
    session_id: Optional[Union[int, str]] = None
    intent: Optional[str] = None
    query: Optional[str] = None
    slots: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[str] = None
    questions: List[Dict[str, Any]] = Field(default_factory=list)
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    skill_groups: List[List[str]] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    scores: List[Dict[str, Any]] = Field(default_factory=list)
    profile: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    state: AgentState
