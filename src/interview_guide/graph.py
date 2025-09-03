import os
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from .router import classify
from .intents import Intent, IntentType
from .agents.recommender import recommend_from_slots
from .agents.qgen import generate_from_topic
from .agents.jd_parser import extract_profile_from_slots
from .agents.evaluator import evaluate_from_slots
from .agents.profile import summarize_profile
from .agents.jd_fetcher import fetch_jd_from_url
from .storage import add_questions
from .storage.prefs import set_prefs, get_prefs
from .configuration import settings

from .storage import new_session, add_scores, get_questions

# Helper to deduplicate questions while preserving order
def _dedup_qs(qs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seen = set()
    out: list[Dict[str, Any]] = []
    for q in qs or []:
        txt = (q.get("text") if isinstance(q, dict) else str(q)) or ""
        key = " ".join(str(txt).strip().lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out

def _normalize_prefs(p):
    if isinstance(p, dict):
        return p
    if isinstance(p, str):
        s = p.lower()
        out = {}
        if "free" in s or "free_only" in s:
            out["free_only"] = True
        sources = []
        if "youtube" in s:
            sources.append("youtube")
        if sources:
            out["sources"] = sources
        return out
    return {}

class GraphState(TypedDict, total=False):
    user_id: Optional[str]
    session_id: Optional[str]
    query: Optional[str]
    intent: Optional[str]
    slots: Dict[str, Any]
    result: str
    questions: list[Dict[str, Any]]
    resources: list[Dict[str, Any]]
    skills: list[str]
    themes: list[str]
    scores: list[Dict[str, Any]]
    profile: Dict[str, Any]

def supervisor(state: GraphState) -> GraphState:
    """Supervisor node: detect intent from user query.

    Rules for testability:
    - If caller already set `intent`, do NOT overwrite it.
    - Merge any pre-existing `slots` with classified slots (caller-provided keys win).
    """
    # If the caller forces an intent (e.g., for tests), keep it
    if state.get("intent"):
        state.setdefault("slots", {})
        return state

    query = state.get("query") or ""
    result_obj: Intent = classify(query)

    # Set detected intent
    state["intent"] = result_obj.type.value

    # Merge slots: caller-provided slots take precedence over router output
    prev_slots = state.get("slots") or {}
    classified_slots = result_obj.model_dump(exclude_none=True)
    merged = {**classified_slots, **prev_slots} if prev_slots else classified_slots
    state["slots"] = merged
    return state

# --- Intent router for graph conditional edges ---

def route_by_intent(state: GraphState) -> str:
    intent = (state.get("intent") or "HELP").upper()
    if intent == IntentType.INGEST_JD.value:
        return "ingest_jd"
    if intent == IntentType.QGEN_FROM_JD.value:
        return "qgen_from_jd"
    if intent == IntentType.QGEN_TOPIC.value:
        return "qgen_topic"
    if intent == IntentType.EVALUATE_ANSWERS.value:
        return "evaluate_answers"
    if intent == IntentType.RECOMMEND_RESOURCES.value:
        return "recommend_resources"
    if intent == IntentType.SHOW_PROFILE.value:
        return "show_profile"
    if intent == IntentType.UPDATE_PREFS.value:
        return "update_prefs"
    return "help_node"

# --- Stub handlers (to be implemented later) ---

def ingest_jd(state: GraphState) -> GraphState:
    slots = state.get("slots", {}) or {}
    jd_text = (slots.get("jd_text") or "").strip()
    jd_url = (slots.get("jd_url") or "").strip()

    if not jd_text and not jd_url:
        state["result"] = "[INGEST_JD] No JD text or URL provided. Paste the JD text to proceed."
        return state

    # If only a URL is provided, fetch the page and extract text
    if not jd_text and jd_url:
        try:
            fetched = fetch_jd_from_url(jd_url)
            jd_text = fetched.strip()
            # keep slots updated so downstream nodes (qgen_from_jd) can reuse it
            state.setdefault("slots", {})
            state["slots"]["jd_text"] = jd_text
        except Exception as e:
            state["result"] = f"[INGEST_JD] failed to fetch JD from URL: {e}"
            return state

    try:
        session_id = new_session(user_id=state.get("user_id"), jd_text=jd_text)
        state["session_id"] = session_id
        state["result"] = f"JD stored; session {session_id} created"
    except Exception as e:
        state["result"] = f"[INGEST_JD] failed to create session: {e}"

    return state

def qgen_from_jd(state: GraphState) -> GraphState:
    slots = state.get("slots", {}) or {}
    try:
        profile = extract_profile_from_slots(slots)  # JDOpenProfile (role-agnostic)
    except Exception as e:
        state["skills"] = []
        state["questions"] = []
        state["result"] = f"[QGEN_FROM_JD] profile extraction failed: {e}"
        return state

    # Save some of the profile to state for visibility
    state["skills"] = list(getattr(profile, "skills", []) or [])
    state["themes"] = list(getattr(profile, "themes", []) or [])

    # Plan total questions across *themes*, picking concrete *tasks* for prompts
    total_cap = max(5, int(getattr(settings, "total_questions_from_jd", 10)))
    weights: Dict[str, float] = dict(getattr(profile, "weights_by_theme", {}) or {})

    # If no weights, distribute evenly across themes; if no themes, default to General
    themes = list(weights.keys()) or (list(getattr(profile, "themes", []) or []) or ["General"])
    if not weights:
        w = round(1.0 / max(1, len(themes)), 4)
        weights = {t: w for t in themes}

    # Round-robin allocation by weights
    import math
    alloc = {t: 0 for t in themes}
    leftover = total_cap
    for t in themes:
        k = int(math.floor(weights.get(t, 0.0) * total_cap))
        alloc[t] = k
        leftover -= k
    if leftover > 0:
        for t in sorted(themes, key=lambda x: weights.get(x, 0.0), reverse=True):
            if leftover <= 0:
                break
            alloc[t] += 1
            leftover -= 1

    # Build task buckets per theme from profile.tasks (simple keyword overlap assignment)
    tasks = list(getattr(profile, "tasks", []) or [])
    buckets: Dict[str, list[str]] = {t: [] for t in themes}
    for task in tasks:
        low = task.lower()
        best_t = None
        best_score = 0
        for th in themes:
            score = 0
            for w in th.lower().split():
                if w and w in low:
                    score += 1
            if score > best_score:
                best_score = score
                best_t = th
        if best_t is None:
            best_t = max(themes, key=lambda x: weights.get(x, 0.0))
        buckets[best_t].append(task)

    # If no tasks, fall back to skills as prompts
    use_skills = False
    if all(len(v) == 0 for v in buckets.values()):
        use_skills = True
        skills = list(getattr(profile, "skills", []) or [])
        for i, sk in enumerate(skills):
            buckets[themes[i % len(themes)]].append(sk)

    # Generate questions batched per theme (1 LLM call per theme)
    all_qs: list[Dict[str, Any]] = []
    for th in themes:
        need = alloc.get(th, 0)
        if need <= 0:
            continue
        items = buckets.get(th, [])
        if not items:
            continue
        # keep a few distinct items to steer the questions
        seen_local = set()
        focus_items = []
        for it in items:
            k = it.strip().lower()
            if not k or k in seen_local:
                continue
            seen_local.add(k)
            focus_items.append(it)
            if len(focus_items) >= max(3, min(5, need)):
                break

        role = getattr(profile, "role", None) or ""
        subs = ", ".join((getattr(profile, "subroles", []) or [])[:1])
        ctx_bits = [th]
        if role:
            ctx_bits.append(role)
        if subs:
            ctx_bits.append(subs)
        context_str = " | ".join([c for c in ctx_bits if c])

        # Build a single topic string that lists focus items
        topic = f"{th} — focus on: " + "; ".join(focus_items)
        topic = topic + f" — context: {context_str}" if context_str else topic

        sk_slots = dict(slots)
        sk_slots["topic"] = topic
        try:
            qs = generate_from_topic(sk_slots, n=need)
        except Exception:
            qs = []
        if qs:
            all_qs.extend(qs)
        if len(all_qs) >= total_cap:
            break

    # First pass questions
    state["questions"] = all_qs[:total_cap]

    # Top-up: if we allocated to empty themes, reuse the richest bucket to reach the cap
    if len(state["questions"]) < total_cap:
        remaining = total_cap - len(state["questions"])
        # pick the theme with the most items
        richest = None
        max_items = 0
        for th in themes:
            n = len(buckets.get(th, []))
            if n > max_items:
                max_items = n
                richest = th
        if richest and max_items > 0:
            # build a focused topic for the richest theme
            seen_local = set()
            focus_items = []
            for it in buckets[richest]:
                k = it.strip().lower()
                if not k or k in seen_local:
                    continue
                seen_local.add(k)
                focus_items.append(it)
                if len(focus_items) >= min(5, remaining):
                    break
            role = getattr(profile, "role", None) or ""
            subs = ", ".join((getattr(profile, "subroles", []) or [])[:1])
            ctx_bits = [richest]
            if role:
                ctx_bits.append(role)
            if subs:
                ctx_bits.append(subs)
            context_str = " | ".join([c for c in ctx_bits if c])
            topic = f"{richest} — focus on: " + "; ".join(focus_items)
            topic = topic + f" — context: {context_str}" if context_str else topic

            sk_slots = dict(slots)
            sk_slots["topic"] = topic
            try:
                extra = generate_from_topic(sk_slots, n=remaining)
            except Exception:
                extra = []
            if extra:
                state["questions"].extend(extra[:remaining])
                state["questions"] = state["questions"][:total_cap]

    # De-duplicate questions while preserving order
    state["questions"] = _dedup_qs(state["questions"])[:total_cap]

    # Persist to the current session if available
    session_id = state.get("session_id")
    if session_id and state["questions"]:
        try:
            add_questions(int(session_id), state["questions"])
        except Exception as e:
            state["result"] = f"Generated {len(state['questions'])} questions (save failed: {e})"
            return state

    # Result line
    if not state["questions"]:
        state["result"] = "[QGEN_FROM_JD] No questions could be generated from this JD."
    else:
        state["result"] = f"Generated {len(state['questions'])} questions across {len(themes)} theme(s)"
    return state

def qgen_topic(state: GraphState) -> GraphState:
    slots = state.get("slots", {}) or {}
    try:
        questions = generate_from_topic(slots, n=5)
    except Exception as e:
        state["questions"] = []
        state["result"] = f"[QGEN_TOPIC] failed: {e}"
        return state

    state["questions"] = questions  # list of {text, topic, difficulty}
    topic = slots.get("topic") or "your topic"
    state["result"] = f"Generated {len(questions)} questions on {topic}"
    return state

def evaluate_answers(state: GraphState) -> GraphState:
    slots = state.get("slots", {}) or {}
    try:
        scores = evaluate_from_slots(slots)
        # If user provided indices of the questions they answered, attach question text
        qidxs = slots.get("question_indices") or slots.get("qidx") or []
        try:
            qidxs = [int(i) for i in qidxs] if isinstance(qidxs, (list, tuple)) else []
        except Exception:
            qidxs = []

        if qidxs:
            sid = state.get("session_id")
            if sid:
                try:
                    qs = get_questions(int(sid))  # 1-based indexing in CLI
                    for i, s in enumerate(scores):
                        if i < len(qidxs):
                            idx = qidxs[i]
                            if 1 <= idx <= len(qs):
                                s["question_text"] = qs[idx - 1]["text"]
                except Exception:
                    pass
        
        
        # List[dict]: answer, score, feedback, confidence
    except Exception as e:
        state["scores"] = []
        state["result"] = f"[EVALUATE_ANSWERS] failed: {e}"
        return state

    # Persist to SQLite (create a session if missing)
    session_id = state.get("session_id")
    if not session_id:
        # optional: attach jd_text if present, otherwise None
        jd_text = slots.get("jd_text")
        try:
            session_id = new_session(user_id=state.get("user_id"), jd_text=jd_text)
            state["session_id"] = session_id
        except Exception as e:
            # Don't fail the node on storage issues; keep running
            state["result"] = f"[EVALUATE_ANSWERS] scored but failed to create session: {e}"
            session_id = None

    if session_id:
        try:
            # add_scores expects dicts with keys: question_text (optional), answer, score, feedback, confidence
            add_scores(int(session_id), scores)
        except Exception as e:
            state["result"] = f"[EVALUATE_ANSWERS] scored but failed to save: {e}"

    state["scores"] = scores
    n = len(scores)
    avg = round(sum(s.get("score", 0) for s in scores) / n, 2) if n else 0.0
    state["result"] = f"Scored {n} answer(s); avg score = {avg}" + (f" (session {session_id})" if session_id else "")
    return state

def recommend_resources(state: GraphState) -> GraphState:
    slots = state.get("slots", {}) or {}

    # normalize prefs if the router gave a string
    if "prefs" in slots:
        slots = dict(slots)
        slots["prefs"] = _normalize_prefs(slots.get("prefs"))

    # merge stored prefs if none provided in this message
    if not slots.get("prefs"):
        try:
            stored = get_prefs(state.get("user_id"))
            if stored:
                slots = dict(slots)
                slots["prefs"] = stored
        except Exception:
            pass

    try:
        resources = recommend_from_slots(slots, max_results=5)
    except Exception as e:
        state["resources"] = []
        state["result"] = f"[RECOMMEND_RESOURCES] search failed: {e}"
        return state

    state["resources"] = resources
    skill = slots.get("skill") or slots.get("topic") or "your query"
    applied_prefs = slots.get("prefs") or {}
    suffix = " (prefs applied)" if applied_prefs else ""
    state["result"] = f"Found {len(resources)} resources for {skill}{suffix}"
    return state

def show_profile(state: GraphState) -> GraphState:
    slots = state.get("slots", {}) or {}
    summary = summarize_profile(slots, session_id=state.get("session_id"), user_id=state.get("user_id"))
    state["profile"] = summary

    total = summary.get("total_answers", 0)
    avg = summary.get("avg_score", 0.0)
    state["result"] = f"Profile: {total} answers, avg score {avg}"
    return state

def update_prefs(state: GraphState) -> GraphState:
    slots = state.get("slots", {}) or {}
    prefs = _normalize_prefs(slots.get("prefs"))
    try:
        set_prefs(state.get("user_id") or "default", prefs)
        state["result"] = f"Preferences saved: {prefs}"
    except Exception as e:
        state["result"] = f"[UPDATE_PREFS] failed to save: {e}"
    return state

def help_node(state: GraphState) -> GraphState:
    note = state.get("slots", {}).get("notes")
    state["result"] = f"[HELP] need clarification. Note: {note or 'Please specify your goal.'}"
    return state

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("supervisor", supervisor)
    g.add_node("ingest_jd", ingest_jd)
    g.add_node("qgen_from_jd", qgen_from_jd)
    g.add_node("qgen_topic", qgen_topic)
    g.add_node("evaluate_answers", evaluate_answers)
    g.add_node("recommend_resources", recommend_resources)
    g.add_node("show_profile", show_profile)
    g.add_node("update_prefs", update_prefs)
    g.add_node("help_node", help_node)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "ingest_jd": "ingest_jd",
            "qgen_from_jd": "qgen_from_jd",
            "qgen_topic": "qgen_topic",
            "evaluate_answers": "evaluate_answers",
            "recommend_resources": "recommend_resources",
            "show_profile": "show_profile",
            "update_prefs": "update_prefs",
            "help_node": "help_node",
        },
    )

    # All leaves end, except `ingest_jd` which flows into `qgen_from_jd`
    for node in [
        "qgen_from_jd",
        "qgen_topic",
        "evaluate_answers",
        "recommend_resources",
        "show_profile",
        "update_prefs",
        "help_node",
    ]:
        g.add_edge(node, END)

    # After ingesting a JD, automatically generate questions from it
    g.add_edge("ingest_jd", "qgen_from_jd")

    return g.compile()

def _pretty_print(out: Dict[str, Any]):
    print("Intent:", out.get("intent"))
    print("Result:", out.get("result"))
    print("Slots:", out.get("slots"))
    if out.get("themes"): print("Themes:", out.get("themes"))

    qs = out.get("questions") or []
    if qs:
        print("\nQuestions:")
        for i, q in enumerate(qs, 1):
            text = q.get("text") if isinstance(q, dict) else str(q)
            print(f"{i}. {text}")

    resources = out.get("resources") or []
    if resources:
        print("\nResources:")
        for i, r in enumerate(resources, 1):
            title = r.get('title') if isinstance(r, dict) else str(r)
            url = r.get('url') if isinstance(r, dict) else None
            if url:
                print(f"{i}. {title} -> {url}")
            else:
                print(f"{i}. {title}")

    scores = out.get("scores") or []
    if scores:
        print("\nScores:")
        for i, s in enumerate(scores, 1):
            print(f"{i}. {s.get('score')} — {s.get('feedback')}")

    profile = out.get("profile") or {}
    if profile:
        print("\nProfile Summary:")
        print(f"Total answers: {profile.get('total_answers')}")
        print(f"Average score: {profile.get('avg_score')}")
        dist = profile.get('distribution') or {}
        if dist:
            print("Distribution:", dist)
        strengths = profile.get('strengths') or []
        if strengths:
            print("Strengths:")
            for s in strengths:
                print("-", s)
        weaknesses = profile.get('weaknesses') or []
        if weaknesses:
            print("Weaknesses:")
            for w in weaknesses:
                print("-", w)

if __name__ == "__main__":
    app = build_graph()
    user_id = os.environ.get("USER_ID", "anon").strip() or "anon"
    print(f"\n[info] using user_id: {user_id}")
    print("\ninterview_guide CLI — type 'exit' to quit")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("bye!")
            break
        out = app.invoke({"query": q, "user_id": user_id})
        _pretty_print(out)