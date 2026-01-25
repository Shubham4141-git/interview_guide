from typing import Any, Dict, List
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from .intents import Intent, IntentType
from .configuration import settings

# System instructions for the router
_SYSTEM = """You are an intent router for an agentic interview assistant.
Classify the user's message into one of these intents:
- INGEST_JD: user provides a JD (text or URL) to store or parse
- QGEN_FROM_JD: generate questions based on skills from a JD
- QGEN_TOPIC: generate questions for a given topic/skill (no JD needed)
- EVALUATE_ANSWERS: evaluate user-provided answers
- RECOMMEND_RESOURCES: suggest courses/blogs/videos for a skill/weakness
- SHOW_PROFILE: summarize strengths/weaknesses from history
- UPDATE_PREFS: update user preferences (e.g., free_only, time_per_week, sources)
- HELP: fallback when unclear

Rules:
- Return a *single* best-fit intent.
- If confidence < 0.6, set clarification_needed=true and briefly say what you need in `notes`.
- Extract slots when present (jd_text, jd_url, topic, answers, skill, prefs, session_selector).
- Output must match the provided Pydantic model exactly.
"""

# --- Heuristic: detect raw-pasted JD blobs ---
JD_KEYWORDS = (
    "responsibilities", "requirements", "role", "about the role", "what you will do",
    "what you'll do", "qualifications", "skills", "must have", "nice to have",
    "experience", "we are looking", "job description", "key responsibilities",
)
_JD_BULLET = re.compile(r"^\s*[-•*]\s+", re.M)
_YEARS_RE   = re.compile(r"\b\d+\+?\s*years?\b", re.I)

def _looks_like_jd(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()

    bullets = len(_JD_BULLET.findall(t))
    keywords = sum(1 for k in JD_KEYWORDS if k in low)
    has_years = bool(_YEARS_RE.search(t))
    longish = len(t) >= 140 

    # Strong JD signals only
    if bullets >= 2:
        return True
    if keywords >= 2 and longish:
        return True
    if has_years and (bullets >= 1 or keywords >= 1):
        return True

    return False

# Prompt used by the LLM classifier
_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    MessagesPlaceholder("history", optional=True),
    ("human", "{query}"),
])


def _make_llm():
    if settings.openai_api_key is None:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return ChatOpenAI(
        model=settings.llm_model_default,
        temperature=0.0,
    )

# Build chain with structured output
_LLM = _make_llm()
_CHAIN = (_PROMPT | _LLM.with_structured_output(Intent, method="function_calling"))


def classify(query: str, history: List[Dict[str, Any]] | None = None) -> Intent:
    """Classify a free-form user query into an Intent object."""
    payload = {"query": query, "history": history or []}
    print(f"[router] classify start query_len={len(query or '')}", flush=True)
    intent: Intent = _CHAIN.invoke(payload)
    print(
        f"[router] classify done intent={intent.type.value if intent.type else None} "
        f"confidence={intent.confidence}",
        flush=True,
    )

    if intent.confidence is None:
        intent.confidence = 0.5
    if not intent.type:
        intent.type = IntentType.HELP

    # Heuristic override: if the raw text looks like a JD, force INGEST_JD
    raw = (query or "").strip()
    try:
        looks_jd = _looks_like_jd(raw)
    except Exception:
        looks_jd = False

    # Don’t override if the user is clearly issuing another command
    starts_with_cmd = raw.lower().startswith((
        "evaluate", "generate", "recommend", "show", "update", "list", "use"
    ))

    if looks_jd and not starts_with_cmd:
        intent.type = IntentType.INGEST_JD
        # Set fields directly on the Intent model (no .slots on this schema)
        try:
            intent.jd_text = raw  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            intent.clarification_needed = False  # type: ignore[attr-defined]
        except Exception:
            pass
        intent.confidence = max(intent.confidence or 0.0, 0.85)

    return intent
