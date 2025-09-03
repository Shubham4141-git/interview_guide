"""
Profile agent
- Reads recent scores from SQLite and builds a strengths/weakness summary.
- Includes per-skill aggregation (using the `skill` field from evaluator output when available).
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from statistics import mean
from interview_guide.storage import get_scores_by_skill

# Skill aggregation thresholds (keep simple)
_STRONG_MIN = 4  # avg >= 4 → strong
_WEAK_MAX = 2    # avg <= 2 → weak

def _aggregate_by_skill(scores: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Return (skill_stats, strength_skills, weakness_skills).
    skill_stats: { skill: {"count": int, "avg": float} }
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for s in scores:
        skill = (s.get("skill") or "").strip() if isinstance(s.get("skill"), str) else None
        if not skill:
            # Try to infer from question_text if available
            qt = s.get("question_text")
            if isinstance(qt, str) and qt:
                # very light heuristic: first 3 words
                skill = " ".join(qt.split()[:3]).lower()
            else:
                continue
        entry = buckets.setdefault(skill, {"count": 0, "sum": 0.0})
        try:
            entry["count"] += 1
            entry["sum"] += float(s.get("score", 0))
        except Exception:
            pass

    stats: Dict[str, Any] = {}
    for k, v in buckets.items():
        cnt = max(1, int(v["count"]))
        avg = round(v["sum"] / cnt, 2)
        stats[k] = {"count": cnt, "avg": avg}

    # Pick strengths/weaknesses by avg thresholds
    strengths = [k for k, v in stats.items() if v["avg"] >= _STRONG_MIN]
    weaknesses = [k for k, v in stats.items() if v["avg"] <= _WEAK_MAX]

    # Sort by avg desc/asc respectively, then by count
    strengths.sort(key=lambda k: (stats[k]["avg"], stats[k]["count"]), reverse=True)
    weaknesses.sort(key=lambda k: (stats[k]["avg"], stats[k]["count"]))

    # Limit lists to keep UI compact
    strengths = strengths[:5]
    weaknesses = weaknesses[:5]
    return stats, strengths, weaknesses

def _bucket(score: int) -> str:
    if score >= 5:
        return "excellent"
    if score >= 4:
        return "strong"
    if score >= 3:
        return "average"
    if score >= 2:
        return "weak"
    return "very weak"

def _summarize(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not scores:
        return {
            "total_answers": 0,
            "avg_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "notes": "No history yet. Answer a few questions and evaluate to build your profile."
        }

    vals = [int(s.get("score", 0)) for s in scores]
    avg = round(mean(vals), 2) if vals else 0.0

    # Basic histogram
    buckets = {"excellent": 0, "strong": 0, "average": 0, "weak": 0, "very weak": 0}
    for s in scores:
        buckets[_bucket(int(s.get("score", 0)))] += 1

    # Per-skill aggregation (uses evaluator-provided `skill` when available)
    skill_stats, strength_skills, weakness_skills = _aggregate_by_skill(scores)

    # Answer-level notes (keep for context)
    strength_notes: List[str] = []
    weakness_notes: List[str] = []
    for s in scores[-20:]:  # recent
        fb = (s.get("feedback") or "").lower()
        if any(k in fb for k in ["clear", "excellent", "correct", "complete", "good"]):
            strength_notes.append(fb[:80] + ("..." if len(fb) > 80 else ""))
        if any(k in fb for k in ["missing", "incorrect", "improve", "lack", "weak"]):
            weakness_notes.append(fb[:80] + ("..." if len(fb) > 80 else ""))

    # de-dup a bit
    strength_notes = list(dict.fromkeys(strength_notes))[:5]
    weakness_notes = list(dict.fromkeys(weakness_notes))[:5]

    return {
        "total_answers": len(scores),
        "avg_score": avg,
        "distribution": buckets,
        "strength_skills": strength_skills,
        "weakness_skills": weakness_skills,
        "skill_stats": skill_stats,
        "strengths": strength_notes,    # keep old key for backward compatibility
        "weaknesses": weakness_notes,    # keep old key for backward compatibility
        "recent": scores[-5:][::-1],
    }

def summarize_profile(slots: Dict[str, Any], session_id: int | None, user_id: str | None) -> Dict[str, Any]:
    """
    Read scores from DB and return a profile summary dict.

    Priority:
      1) if slots specify session_selector == 'latest' -> latest session (handled implicitly by DB order);
      2) else, if session_id is provided in state, use it;
      3) else, fall back to all scores for user_id (if provided).
    """
    # Try specific session first
    sel = (slots or {}).get("session_selector")
    scores: List[Dict[str, Any]] = []
    if session_id and sel != "all":
        scores = get_scores_by_skill(session_id=session_id)
    elif user_id:
        scores = get_scores_by_skill(user_id=user_id)
    else:
        # general fallback: entire table (not ideal for multi-user, but fine for local dev)
        scores = get_scores_by_skill()

    return _summarize(scores)