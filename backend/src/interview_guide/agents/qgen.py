"""
Question Generator (topic-based)
- uses OpenAI via langchain-openai
- returns a small list of clean questions (default 5)
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from interview_guide.configuration import settings
from typing import Dict, Any, List
import re

class QuestionItem(BaseModel):
    text: str = Field(...)

class QuestionsList(BaseModel):
    questions: List[QuestionItem] = Field(default_factory=list)

_TEMPLATE = (
    "You generate interview practice questions.\n"
    "Topic: \"{topic}\"\n"
    "Target: {n} concise questions. No answers.\n"
    "Return JSON matching this schema exactly (no extra text):\n"
    "{{\"questions\": [{{\"text\": \"...\"}}]}}\n"
)

# Heuristic: try to honor a requested number of questions if present
_MIN_Q = 1
_MAX_Q = 25

_DEF_KEYS = ("n", "num", "count", "k")
_TEXT_KEYS = ("original_query", "query", "topic_raw")

_DEF_RE = re.compile(r"\b(\d{1,2})\b")

def _requested_n(slots: Dict[str, Any], default: int) -> int:
    # 1) direct numeric fields in slots
    for k in _DEF_KEYS:
        v = slots.get(k)
        try:
            iv = int(v)
            if _MIN_Q <= iv <= _MAX_Q:
                return iv
        except Exception:
            pass
    # 2) parse from any available raw text
    for k in _TEXT_KEYS:
        t = slots.get(k)
        if isinstance(t, str) and t:
            m = _DEF_RE.search(t)
            if m:
                try:
                    iv = int(m.group(1))
                    if _MIN_Q <= iv <= _MAX_Q:
                        return iv
                except Exception:
                    pass
    # 3) fallback
    return max(_MIN_Q, min(default, _MAX_Q))

def _llm():
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return ChatOpenAI(model=settings.llm_model_default, temperature=0.3)

def generate_from_topic(slots: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
    topic = (slots.get("topic") or "").strip()
    if not topic:
        raise ValueError("QGEN_TOPIC requires a 'topic' slot.")
    n_final = _requested_n(slots, n)
    prompt = ChatPromptTemplate.from_template(_TEMPLATE)
    chain = prompt | _llm().with_structured_output(QuestionsList, method="function_calling")
    resp: QuestionsList = chain.invoke({"topic": topic, "n": n_final})
    out = [{"text": q.text.strip(), "topic": topic, "difficulty": "auto"} for q in resp.questions]
    # ensure length respects requested n and removes empties
    out = [q for q in out if q.get("text")] [: n_final]
    return out
