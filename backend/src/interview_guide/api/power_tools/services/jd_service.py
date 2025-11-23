"""Service layer that orchestrates JD-related agent calls."""

from typing import Any, Dict

from interview_guide.agents.jd_fetcher import fetch_jd_from_url
from interview_guide.agents.jd_parser import extract_profile_from_slots


def fetch_jd_text(url: str, max_chars: int) -> str:
    """Fetch raw JD text from a public URL."""
    return fetch_jd_from_url(url=url, max_chars=max_chars)


def build_jd_profile(jd_text: str) -> Dict[str, Any]:
    """Invoke the JD parser agent to derive a structured profile."""
    profile = extract_profile_from_slots({"jd_text": jd_text})
    return profile.model_dump()
