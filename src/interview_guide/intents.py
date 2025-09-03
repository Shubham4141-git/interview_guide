from enum import Enum
from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel, Field

class IntentType(str, Enum):
    """All high-level actions the supervisor can route to."""
    INGEST_JD = "INGEST_JD"
    QGEN_FROM_JD = "QGEN_FROM_JD"
    QGEN_TOPIC = "QGEN_TOPIC"
    EVALUATE_ANSWERS = "EVALUATE_ANSWERS"
    RECOMMEND_RESOURCES = "RECOMMEND_RESOURCES"
    SHOW_PROFILE = "SHOW_PROFILE"
    UPDATE_PREFS = "UPDATE_PREFS"
    HELP = "HELP"

class Intent(BaseModel):
    """Structured output from the router (LLM classifier).

    The router should always fill `type` and `confidence`.
    Other fields are optional and may remain None when not relevant.
    """

    # Mandatory
    type: IntentType = Field(..., description="Selected intent label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Router confidence in [0,1]")

    # Optional slots extracted from the user query
    jd_text: Optional[str] = Field(None, description="Raw JD text pasted by the user")
    jd_url: Optional[str] = Field(None, description="URL to a JD posting")
    topic: Optional[str] = Field(None, description="Topic/skill for question generation")
    answers: Optional[List[str]] = Field(None, description="One or more user answers to evaluate")
    skill: Optional[str] = Field(None, description="Skill to recommend resources for")
    # prefs: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
    #     None, description="User preference overrides (e.g., free_only, time_per_week)"
    # )
    prefs: Optional[Union[Dict[str, Any], str]] = None
    session_selector: Optional[str] = Field(
        None, description='Which prior session to use (e.g., "latest")'
    )
    notes: Optional[str] = Field(None, description="Short router note or clarification request")
    clarification_needed: bool = Field(False, description="If True, the router needs one clarifying answer before proceeding")

    class Config:
        extra = "ignore"