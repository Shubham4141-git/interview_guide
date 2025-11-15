"""Service helpers for sessions and stored questions."""

from typing import Any, Dict, List, Optional

from interview_guide.storage import new_session, list_sessions, get_questions


def create_session(user_id: Optional[str], jd_text: Optional[str]) -> Dict[str, Any]:
    """Create a new session and return basic metadata."""
    session_id = new_session(user_id=user_id, jd_text=jd_text)
    # Fetch the newly created session details
    resolved_user = user_id or "anon"
    sessions = list_sessions(resolved_user, limit=5)
    session = next((s for s in sessions if int(s["id"]) == session_id), None)
    if session is None:
        session = {"id": session_id, "user_id": resolved_user, "created_at": "", "jd_text": jd_text}
    return session


def list_user_sessions(user_id: str, limit: int) -> List[Dict[str, Any]]:
    """List recent sessions for a user."""
    return list_sessions(user_id=user_id, limit=limit)


def list_session_questions(session_id: int) -> List[Dict[str, Any]]:
    """Retrieve stored questions for a given session."""
    return get_questions(session_id=session_id)
