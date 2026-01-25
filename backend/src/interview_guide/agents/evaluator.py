"""
Answer Evaluator
- Input: slots: {answers: [str], topic?: str}
- Output: list of scorecards per answer: {answer, score(0-5), feedback, confidence}
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field, conint, confloat
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from interview_guide.configuration import settings

# -------- structured output --------
class ScoreItem(BaseModel):
    answer: str
    score: conint(ge=0, le=5) = Field(..., description="0=incorrect, 5=excellent")
    feedback: str = Field(..., description="1-2 sentences: why this score; how to improve")
    confidence: confloat(ge=0.0, le=1.0) = 0.7
    skill: str | None = Field(default=None, description="short skill/topic label inferred from the answer (e.g., 'SQL joins')")

class ScoreList(BaseModel):
    items: List[ScoreItem] = Field(default_factory=list)

_SYSTEM = (
    "You are a strict, fair interview evaluator.\n"
    "Score each answer from 0–5 using the FULL range (do NOT collapse to only 0 and 5).\n"
    "Rubric anchors (integers only):\n"
    "0 = Off-topic or factually wrong.\n"
    "1 = Mostly incorrect or missing the core idea; major misconceptions.\n"
    "2 = Partially correct but with significant gaps or inaccuracies.\n"
    "3 = Core idea correct but lacks depth/examples/precision; some gaps.\n"
    "4 = Mostly correct and clear, minor omissions or imprecision.\n"
    "5 = Accurate, complete, and well-structured with precise terminology.\n"
    "Guidelines: Assign 3 for an okay-but-thin answer; use 1–2 for partial credit; use 4 when strong but not perfect.\n"
    "Feedback: 1–2 sentences. Mention one strength (if any) and one concrete improvement (missing concept/edge case/definition).\n"
    "Also provide a short skill label for each answer (e.g., 'SQL joins', 'window functions', 'A/B testing').\n"
    "Return ONLY JSON: {\"items\":[{\"answer\":\"...\",\"score\":0-5,\"feedback\":\"...\",\"confidence\":0-1,\"skill\":\"...\"}]}\n"
)

_TEMPLATE = (
    "{system}\n"
    "Topic (optional): {topic}\n"
    "Answers:\n{answers_block}\n"
    "Return JSON exactly as specified."
)

def _llm():
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return ChatOpenAI(
        model=settings.llm_model_eval,
        temperature=0.1,
    )

def _answers_block(answers: List[str]) -> str:
    lines = []
    for i, a in enumerate(answers, 1):
        lines.append(f"{i}. {a.strip()}")
    return "\n".join(lines)

def evaluate_from_slots(slots: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Evaluate user answers.
    slots should contain: answers: List[str]; topic (optional)
    returns: List[dict] with keys: answer, score, feedback, confidence
    """
    answers = slots.get("answers")
    if not isinstance(answers, list) or not answers:
        raise ValueError("EVALUATE_ANSWERS requires 'answers' as a non-empty list of strings.")
    topic = (slots.get("topic") or "").strip()

    prompt = ChatPromptTemplate.from_template(_TEMPLATE)
    chain = prompt | _llm().with_structured_output(ScoreList, method="function_calling")

    payload = {
        "system": _SYSTEM,
        "topic": topic or "(not specified)",
        "answers_block": _answers_block(answers),
    }

    resp: ScoreList = chain.invoke(payload)
    # normalize
    return [
        {
            "answer": it.answer.strip(),
            "score": int(it.score),
            "feedback": it.feedback.strip(),
            "confidence": float(it.confidence),
            "skill": (it.skill.strip() if isinstance(it.skill, str) else None),
        }
        for it in (resp.items or [])
    ]
