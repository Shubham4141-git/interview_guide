"""User-centric helper utilities."""

from typing import Dict, List, Optional

from interview_guide.storage import list_user_ids

from .profile_service import get_profile_summary
from .recommendation_service import recommend_resources


def fetch_user_ids(limit: int = 50) -> List[str]:
    """Return known user IDs."""
    return list_user_ids(limit=limit)


def build_user_overview(user_id: str, max_resources: int = 5) -> Dict[str, object]:
    """Return profile summary and suggested resources for a user."""
    summary = get_profile_summary(user_id=user_id, session_selector="all")
    weaknesses = summary.get("weakness_skills") or []
    strengths = summary.get("strength_skills") or []
    focus_skill: Optional[str] = None
    resources: List[Dict[str, object]] = []

    if weaknesses:
        focus_skill = weaknesses[0]
    elif strengths:
        focus_skill = strengths[0]

    if focus_skill:
        resources = recommend_resources(
            skill=focus_skill, topic=None, prefs=None, max_results=max_resources
        )

    return {
        "user_id": user_id,
        "profile": summary,
        "resources": resources,
        "focus_skill": focus_skill,
    }
