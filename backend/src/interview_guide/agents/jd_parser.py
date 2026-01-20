"""
JD Parser (role-agnostic)
- Input: slots containing jd_text (or jd_url handled upstream)
- Output: an open, domain-agnostic JD profile usable for any job (tech + non-tech)

This module intentionally avoids hard-coded domain buckets. The LLM is asked to
produce free-text themes and subroles tailored to the JD. A generic, dictionary-
free fallback derives tasks/themes from the text if the LLM call fails.

Backward-compat APIs retained:
- extract_skills_from_slots(slots) -> List[str]
- extract_skills_and_weights_from_slots(slots) -> (List[str], Dict[str, float])
"""
from __future__ import annotations
from typing import Dict, Any, List, Set, Tuple, Optional
import re
from collections import Counter
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from interview_guide.configuration import settings

# ---------------- LLM ----------------

def _llm():
    if not settings.google_api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY).")
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        api_key=settings.google_api_key,
        temperature=0.15,
    )

# ---------------- Schemas ----------------

class JDOpenProfile(BaseModel):
    role: Optional[str] = None
    subroles: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)  # free-text labels, 3–6 items
    weights_by_theme: Any = Field(default_factory=dict)  # sum ~ 1.0
    tasks: Any = Field(default_factory=list)     # 10–20 concise task statements
    skills: Any = Field(default_factory=list)    # overall deduped skills/tools/methods (incl. soft skills if explicit)
    technical_skills: Any = Field(default_factory=list)  # strictly technical skills/tools only
    non_technical_skills: Any = Field(default_factory=list)  # soft skills, management, HR, ops
    management_responsibilities: Any = Field(default_factory=list)  # people/process/stakeholder duties
    job_type: Optional[str] = None  # "tech" | "non-tech"
    job_type_confidence: Optional[float] = None
    contexts: Any = Field(default_factory=list)  # constraints/shifts/compliance/equipment/domain notes

# For backward compatibility
class SkillList(BaseModel):
    skills: List[str] = Field(default_factory=list)

# ---------------- Prompts ----------------

_SYSTEM_PROFILE = (
    "You analyze a Job Description (JD). Return ONLY strict JSON with these keys:\n"
    "role: canonical job role title (free text).\n"
    "subroles: 1–4 subroles or specializations (free text).\n"
    "themes: 3–6 high-level THEMES YOU invent for this JD (free text labels).\n"
    "weights_by_theme: numeric weights over those themes summing to ~1.0 (0–1 floats).\n"
    "tasks: 10–20 short, concrete task statements extracted from the JD, plain language.\n"
    "skills: deduplicated list of concrete skills/tools/methods (include soft skills only if explicit).\n"
    "technical_skills: list of strictly technical skills/tools/technologies from the JD.\n"
    "non_technical_skills: list of soft skills, management, HR, or process skills from the JD.\n"
    "management_responsibilities: list of people/process/stakeholder responsibilities (short phrases).\n"
    "job_type: either \"tech\" or \"non-tech\" based on the primary role focus.\n"
    "job_type_confidence: float 0-1 for the job_type decision.\n"
    "contexts: constraints/shift patterns/compliance/equipment/domain notes (short phrases), if present.\n"
    "Return ONLY valid JSON matching these keys. No extra text."
)

_TEMPLATE_PROFILE = (
    "{system}\n\nJD:\n{jd_text}\n\n"
    "Return JSON exactly like:\n"
    "{{\"role\":\"Security Guard\",\"subroles\":[\"Night Shift\",\"Corporate Office\"],"
    "\"themes\":[\"Patrol & Monitoring\",\"Access Control\",\"Incident Response\"],"
    "\"weights_by_theme\":{{\"Patrol & Monitoring\":0.45,\"Access Control\":0.35,\"Incident Response\":0.20}},"
    "\"tasks\":[\"Perform scheduled patrols\",\"Monitor CCTV\",\"Check visitor badges\",\"Respond to alarms\"],"
    "\"skills\":[\"Report writing\",\"De-escalation\",\"CCTV operation\",\"Radio comms\"],"
    "\"technical_skills\":[\"CCTV operation\"],"
    "\"non_technical_skills\":[\"Report writing\",\"De-escalation\"],"
    "\"management_responsibilities\":[\"Supervise overnight shift\"],"
    "\"job_type\":\"non-tech\",\"job_type_confidence\":0.8,"
    "\"contexts\":[\"Night shifts\",\"HIPAA environment\",\"Use of metal detector\"]}}"
)

# ---------------- Generic helpers / fallback ----------------

_STOP = {
    "the","a","an","and","or","for","of","to","in","on","with","by","at","from","as",
    "is","are","be","will","shall","must","may","can","should","into","per","via","that",
}

_SENT_SPLIT_RE = re.compile(r"[\n\.;!?]+\s*")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-/+#]*")


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out


def _extract_candidate_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    # prefer lines that look like duties: start with verbs or contain action words
    verbs = {"monitor","patrol","prepare","respond","investigate","coordinate","manage","operate","inspect","secure","assist","train","teach","design","develop","analyze","report","communicate","maintain","audit","handle","plan","organize","support"}
    ranked = []
    for p in parts:
        low = p.lower()
        score = 0
        # leading bullet markers
        if p[:2] in {"- ","* "} or low.startswith(("responsible for","duty","duties","responsibilities")):
            score += 2
        # verb presence
        if any(v in low for v in verbs):
            score += 1
        # reasonable length
        if 30 <= len(p) <= 240:
            score += 1
        ranked.append((score, p))
    ranked.sort(key=lambda t: (-t[0], len(t[1])))
    # keep top 30 candidate sentences
    return [p for _, p in ranked[:30]]


def _fallback_tasks_themes(jd_text: str) -> Tuple[List[str], List[str], Dict[str, float]]:
    """Pure-text fallback: derive tasks and themes without any domain dictionary."""
    cands = _extract_candidate_sentences(jd_text)
    tasks = _dedupe_keep_order(cands)[:20]

    # Build keyword counts from tasks
    counts: Counter[str] = Counter()
    for t in tasks:
        for w in _WORD_RE.findall(t.lower()):
            if w in _STOP or len(w) < 3:
                continue
            counts[w] += 1

    # Pick top words as theme seeds; group similar words under the seed
    top = [w for w, _ in counts.most_common(6)] or []
    themes = [w.capitalize() for w in top]
    if not themes:
        themes = ["General"]

    # Weights proportional to counts of seed words
    total = sum(counts[w] for w in top) or 1
    weights = {th: round(counts[th.lower()]/total, 4) for th in themes}

    return tasks, themes, weights


def _regex_skills(jd_text: str) -> List[str]:
    # very light generic skill extraction: title-case alphabetic tokens and tool-like tokens
    tokens = _WORD_RE.findall(jd_text)
    # collect capitalized or tool-ish tokens; remove stopwords
    raw = [t for t in tokens if t.lower() not in _STOP]
    # stitch simple bigrams for phrases like "Access Control", "Customer Service"
    phrases: List[str] = []
    for i in range(len(raw) - 1):
        a, b = raw[i], raw[i + 1]
        if a[0].isupper() and b[0].isupper():
            phrases.append(f"{a} {b}")
    # single tokens that look like skills
    singles = [t for t in raw if t[0].isupper() or len(t) >= 4]
    skills = _dedupe_keep_order(phrases + singles)
    # trim excessive list
    return skills[:80]

# ---------------- Public API ----------------

def extract_profile_from_slots(slots: Dict[str, Any]) -> JDOpenProfile:
    jd_text = (slots.get("jd_text") or "").strip()
    if not jd_text:
        raise ValueError("INGEST_JD/QGEN_FROM_JD requires 'jd_text' in slots.")

    prompt = ChatPromptTemplate.from_template(_TEMPLATE_PROFILE)
    llm = _llm()
    chain = prompt | llm.with_structured_output(JDOpenProfile)

    try:
        prof: JDOpenProfile = chain.invoke({"system": _SYSTEM_PROFILE, "jd_text": jd_text})
        print("[JD_PARSER] Using LLM profile")
        # --- Coerce fields in case the model returned JSON-as-strings ---
        import json
        # weights_by_theme
        w = getattr(prof, "weights_by_theme", {})
        if isinstance(w, str):
            try:
                w = json.loads(w)
            except Exception:
                # try to fix single quotes
                try:
                    w = json.loads(w.replace("'", '"'))
                except Exception:
                    w = {}
        if not isinstance(w, dict):
            w = {}
        prof.weights_by_theme = w  # type: ignore[attr-defined]

        # tasks / skills / contexts may come back as strings
        def _as_list(x):
            if isinstance(x, list):
                return [str(i) for i in x]
            if isinstance(x, str):
                xs = x.strip()
                if xs.startswith("["):
                    try:
                        arr = json.loads(xs)
                        return [str(i) for i in (arr if isinstance(arr, list) else [])]
                    except Exception:
                        pass
                # fallback: comma-separated
                return [s.strip() for s in xs.split(",") if s.strip()]
            return []

        prof.tasks = _as_list(getattr(prof, "tasks", []))  # type: ignore[attr-defined]
        prof.skills = _as_list(getattr(prof, "skills", []))  # type: ignore[attr-defined]
        prof.technical_skills = _as_list(getattr(prof, "technical_skills", []))  # type: ignore[attr-defined]
        prof.non_technical_skills = _as_list(getattr(prof, "non_technical_skills", []))  # type: ignore[attr-defined]
        prof.management_responsibilities = _as_list(  # type: ignore[attr-defined]
            getattr(prof, "management_responsibilities", [])
        )
        prof.contexts = _as_list(getattr(prof, "contexts", []))  # type: ignore[attr-defined]

        def _normalize_job_type(value: Any) -> Optional[str]:
            if not value:
                return None
            low = str(value).strip().lower()
            if low in {"tech", "technical", "engineering"}:
                return "tech"
            if low in {"non-tech", "nontech", "non technical", "nontechnical"}:
                return "non-tech"
            return None

        prof.job_type = _normalize_job_type(getattr(prof, "job_type", None))  # type: ignore[attr-defined]
        try:
            conf = getattr(prof, "job_type_confidence", None)
            conf_val = float(conf) if conf is not None else None
        except Exception:
            conf_val = None
        if conf_val is not None:
            conf_val = max(0.0, min(1.0, conf_val))
        prof.job_type_confidence = conf_val  # type: ignore[attr-defined]

    except Exception as e:
        print("[JD_PARSER] Using fallback profile —", repr(e))
        # LLM failed — build a heuristic profile with NO domain dictionaries
        tasks, themes, weights = _fallback_tasks_themes(jd_text)
        prof = JDOpenProfile(
            role=None,
            subroles=[],
            themes=themes,
            weights_by_theme=weights,
            tasks=tasks,
            skills=_regex_skills(jd_text),
            technical_skills=[],
            non_technical_skills=[],
            management_responsibilities=[],
            job_type=None,
            job_type_confidence=None,
            contexts=[],
        )

    # Normalize fields
    # skills
    seen: Set[str] = set()
    skills_out: List[str] = []
    for s in (prof.skills or []):
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            skills_out.append(s.strip())

    # weights sum ~1.0
    weights = prof.weights_by_theme or {}
    ssum = sum(max(0.0, float(v)) for v in weights.values())
    if ssum > 0:
        weights = {k: round(max(0.0, float(v))/ssum, 4) for k, v in weights.items()}

    # themes aligned with weights keys if available
    themes = list(weights.keys()) or (prof.themes or [])

    def _clean_list(items: Any, limit: int) -> List[str]:
        cleaned = _dedupe_keep_order([str(i).strip() for i in (items or []) if str(i).strip()])
        return cleaned[:limit]

    technical_skills_out = _clean_list(getattr(prof, "technical_skills", []), 80)
    non_technical_skills_out = _clean_list(getattr(prof, "non_technical_skills", []), 80)
    management_out = _clean_list(getattr(prof, "management_responsibilities", []), 40)

    return JDOpenProfile(
        role=(prof.role or None),
        subroles=[sr.strip() for sr in (prof.subroles or []) if sr and sr.strip()][:4],
        themes=themes[:6] if themes else [],
        weights_by_theme=weights,
        tasks=_dedupe_keep_order(
            list(dict.fromkeys([t.strip() for t in (prof.tasks or []) if t and t.strip()]))
        )[:20],
        skills=skills_out[:80],
        technical_skills=technical_skills_out,
        non_technical_skills=non_technical_skills_out,
        management_responsibilities=management_out,
        job_type=getattr(prof, "job_type", None),
        job_type_confidence=getattr(prof, "job_type_confidence", None),
        contexts=_dedupe_keep_order([c.strip() for c in (prof.contexts or []) if c and c.strip()])[:12],
    )

# ---------- Backward-compatible helpers ----------

def extract_skills_and_weights_from_slots(slots: Dict[str, Any]) -> Tuple[List[str], Dict[str, float]]:
    prof = extract_profile_from_slots(slots)
    return list(prof.skills), dict(prof.weights_by_theme)


def extract_skills_from_slots(slots: Dict[str, Any]) -> List[str]:
    skills, _ = extract_skills_and_weights_from_slots(slots)
    return skills
