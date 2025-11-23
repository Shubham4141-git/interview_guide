"""Service helpers around the question generation agent."""

from typing import Dict, Any, List, Optional, Tuple

from interview_guide.agents.qgen import generate_from_topic
from interview_guide.storage import add_questions


def generate_questions(
    topic: str,
    count: int,
    session_id: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Generate interview questions using the existing agent, optionally persisting them."""
    slots: Dict[str, Any] = {"topic": topic, "n": count}
    questions = generate_from_topic(slots, n=count)

    if session_id is not None:
        add_questions(session_id=session_id, questions=questions)

    return questions, session_id
