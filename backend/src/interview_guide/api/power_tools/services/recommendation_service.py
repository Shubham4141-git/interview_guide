"""Service helpers for resource recommendations."""

from typing import Any, Dict, List, Optional

from interview_guide.agents.recommender import recommend_from_slots


def recommend_resources(
    skill: Optional[str],
    topic: Optional[str],
    prefs: Optional[Dict[str, Any]],
    max_results: int,
) -> List[Dict[str, Any]]:
    """Return recommended learning resources from the existing recommender agent."""
    slots: Dict[str, Any] = {
        "skill": skill,
        "topic": topic,
        "prefs": prefs or {},
    }
    return recommend_from_slots(slots, max_results=max_results)
