"""
Recommender agent:
- builds a search query from slots (skill/topic + prefs)
- calls tools.tavily.search
- returns a normalized list of resource cards
"""
from typing import Dict, Any, List
from interview_guide.tools.tavily import search as tavily_search

def _build_query(skill: str, prefs: Dict[str, Any]) -> tuple[str, list[str] | None]:
    q_parts: List[str] = []
    if skill:
        q_parts.append(skill)
    if prefs.get("free_only"):
        q_parts.append("free")
    # focus on learning-type results
    q_parts.extend(["tutorial", "course", "blog"])

    include_domains = None
    sources = prefs.get("sources")
    if isinstance(sources, list) and sources:
        domains = []
        for s in sources:
            s = str(s).lower()
            if s == "youtube":
                domains.append("youtube.com")
            # add more mappings here if needed
        include_domains = domains or None

    query = " ".join([p for p in q_parts if p]) or "learning resources"
    return query, include_domains

def recommend_from_slots(slots: Dict[str, Any], max_results: int = 5) -> List[Dict[str, Any]]:
    """Pure function: takes router slots, returns a list of resource dicts."""
    skill = slots.get("skill") or slots.get("topic") or ""
    prefs = slots.get("prefs") or {}
    query, include_domains = _build_query(skill, prefs)

    results = tavily_search(query=query, max_results=max_results, include_domains=include_domains)
    # already normalized to: {title, url, snippet, score, source}
    return results