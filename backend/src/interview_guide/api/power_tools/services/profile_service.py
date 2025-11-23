"""Service wrappers for profile summaries."""

from typing import Any, Dict, Optional

from interview_guide.agents.profile import summarize_profile


def get_profile_summary(
    user_id: Optional[str] = None,
    session_id: Optional[int] = None,
    session_selector: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch profile summary data using the profile agent."""
    slots: Dict[str, Any] = {}
    if session_selector:
        slots["session_selector"] = session_selector
    return summarize_profile(slots, session_id=session_id, user_id=user_id)
