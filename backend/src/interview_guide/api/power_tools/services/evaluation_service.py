"""Service helpers for answer evaluation."""

from typing import Dict, Any, List, Optional, Sequence, Tuple

from interview_guide.agents.evaluator import evaluate_from_slots
from interview_guide.storage import add_scores


def evaluate_answers(
    answers: List[str],
    topic: str | None = None,
    session_id: Optional[int] = None,
    question_texts: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Evaluate answers using the existing evaluator agent, optionally persisting results."""
    slots: Dict[str, Any] = {"answers": answers}
    if topic:
        slots["topic"] = topic

    results = evaluate_from_slots(slots)

    if session_id is not None:
        prepared: List[Dict[str, Any]] = []
        for idx, item in enumerate(results):
            record = dict(item)
            if question_texts and idx < len(question_texts):
                record["question_text"] = question_texts[idx]
            prepared.append(record)
        add_scores(session_id=session_id, scores=prepared)

    return results, session_id
