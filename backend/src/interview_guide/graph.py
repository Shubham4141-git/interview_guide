import os
from typing import TypedDict, Optional, Dict, Any, List
from langgraph.graph import StateGraph, END
from .router import classify
from .intents import Intent, IntentType
from .agents.recommender import recommend_from_slots
from .agents.qgen import generate_from_topic
from .agents.jd_parser import extract_profile_from_slots
from .agents.evaluator import evaluate_from_slots
from .agents.profile import summarize_profile
from .agents.jd_fetcher import fetch_jd_from_url
from .agents.skill_clusterer import cluster_skills
from .storage import add_questions
from .storage.prefs import set_prefs, get_prefs
from .configuration import settings

from .storage import new_session, add_scores, get_questions

_SKILL_FEW_SHOT_EXAMPLE = (
    "REFERENCE ONLY — match tone and depth, but DO NOT reuse these exact questions:\n"
    "Skill: Decision Tree\n"
    "1. What is a decision tree and how do you detect when it is overfitting in production?\n"
    "2. Explain how you select a split criterion when the classes are heavily imbalanced.\n"
    "3. Compare a single decision tree with XGBoost — when would you ship the simpler model?\n"
    "\n"
    "Skill: Python\n"
    "1. What is the difference between == and is in Python?\n"
    "2. Explain the concept of list comprehensions and when you would use them.\n"
    "3. How does Python's garbage collection work?\n"
    "4. What are Python generators and how do they differ from regular functions?\n"
    "5. Describe the difference between *args and **kwargs.\n"
    "\n"
    "Skill: JavaScript\n"
    "6. What is the difference between let, const, and var?\n"
    "7. Explain event bubbling and event capturing in JavaScript.\n"
    "8. How do closures work in JavaScript? Provide an example scenario where they're useful.\n"
    "9. What is the difference between == and === in JavaScript?\n"
    "10. Describe how asynchronous programming works with Promises and async/await.\n"
    "\n"
    "Skill: SQL\n"
    "11. Write a query to find duplicate records in a table.\n"
    "12. Explain the difference between clustered and non-clustered indexes.\n"
    "13. How would you optimize a slow-performing SQL query?\n"
    "14. What is a database transaction and what are ACID properties?\n"
    "15. How do you handle NULL values in SQL queries?\n"
    "\n"
    "Skill: Docker\n"
    "16. How would you reduce the size of a Docker image?\n"
    "17. Explain the difference between COPY and ADD commands in a Dockerfile.\n"
    "18. How do you debug issues in a running Docker container?\n"
    "19. What is the purpose of a .dockerignore file?\n"
    "20. How would you handle logging in containerized applications?\n"
    "\n"
    "Skill: Kubernetes\n"
    "21. How do you troubleshoot a pod that's stuck in Pending state?\n"
    "22. Explain the difference between ConfigMaps and Secrets.\n"
    "23. How would you implement rolling updates for a deployment?\n"
    "24. What are Ingress controllers and how do they work?\n"
    "25. How do you monitor resource usage across your Kubernetes cluster?\n"
    "\n"
    "Skill: AWS\n"
    "26. What's the difference between S3 storage classes?\n"
    "27. How would you design a highly available architecture using AWS services?\n"
    "28. Explain the difference between Application Load Balancer and Network Load Balancer.\n"
    "29. How do you secure data in transit and at rest in AWS?\n"
    "30. What strategies would you use for cost optimization in AWS?\n"
    "\n"
    "Skill: System Design\n"
    "31. How would you design a URL shortener like bit.ly?\n"
    "32. Design a chat application that can handle millions of users.\n"
    "33. How would you implement a distributed cache?\n"
    "34. Design a rate limiting system for an API.\n"
    "35. How would you handle data consistency in a microservices architecture?\n"
    "\n"
    "Skill: Git\n"
    "36. What's the difference between git merge and git rebase?\n"
    "37. How do you resolve merge conflicts in Git?\n"
    "38. Explain Git branching strategies for a team environment.\n"
    "39. How would you undo the last commit in Git?\n"
    "40. What is the difference between git pull and git fetch?\n"
    "\n"
    "Skill: Linux / DevOps\n"
    "41. How do you check system performance and identify bottlenecks?\n"
    "42. Explain the difference between soft and hard links in Linux.\n"
    "43. How would you set up automated backups for a database server?\n"
    "44. What are the different types of load balancing algorithms?\n"
    "45. How do you implement a blue-green deployment strategy?\n"
    "\n"
    "Skill: API Design / REST\n"
    "46. What are the principles of REST API design?\n"
    "47. How do you version your APIs?\n"
    "48. Explain different HTTP status codes and when to use them.\n"
    "49. How would you implement authentication and authorization in APIs?\n"
    "50. What's the difference between GraphQL and REST APIs?\n"
)

_SOFT_SKILL_STOPWORDS = {
    "communication",
    "collaboration",
    "collaborative",
    "teamwork",
    "team player",
    "leadership",
    "stakeholder management",
    "stakeholder",
    "presentation",
    "negotiation",
    "problem solving",
    "problem-solving",
    "adaptability",
    "adaptable",
    "time management",
    "organizational",
    "organization",
    "empathy",
    "interpersonal",
    "communication skills",
    "written communication",
    "verbal communication",
    "people management",
    "mentoring",
    "coaching",
}


def _filter_technical_skills(skills: list[str]) -> list[str]:
    """Drop skills that look like soft skills using a simple stopword list."""
    out: list[str] = []
    for skill in skills:
        low = skill.strip().lower()
        if not low:
            continue
        if any(stop in low for stop in _SOFT_SKILL_STOPWORDS):
            continue
        out.append(skill)
    return out


def _build_skill_topic(skills: List[str], n: int) -> str:
    """Create a reusable few-shot topic string focused strictly on listed skills."""
    if not skills:
        raise ValueError("_build_skill_topic requires at least one skill")
    skill_title = " / ".join(skills)
    bullet_lines = "\n".join(f"- {s}" for s in skills)
    intro = (
        f"SKILL TARGETS: {skill_title}\n"
        "You are a senior practitioner conducting a technical interview.\n"
        "Stay strictly on these skills — no unrelated tools, teams, or HR chatter.\n"
    )
    # guidance = (
    #     "Probe mechanics, workflows, trade-offs, and how the candidate applies the skill.\n"
    #     "Alternate between quick knowledge checks and deeper follow-ups.\n"
    #     "At least half the questions should be direct (definition, mechanism, difference).\n"
    #     "Limit scenario-based questions to at most one per batch and avoid phrases like 'Describe a scenario' or 'Describe a situation'.\n"
    #     "Ensure question stems are unique; do not reuse the same opening phrase across the batch.\n"
    #     "Cover varied angles: include definition/mechanism, implementation/how-to, troubleshooting/debugging, and optimization/best-practice views where possible.\n"
    #     "Favor concise stems such as 'What', 'How does', 'Why', 'When would', or 'Explain'.\n"
    #     "Keep each question concise. Do not provide answers.\n"
    #     "Generate fresh questions — do NOT repeat any examples shown in the reference bank.\n"
    #     "Ensure every question references at least one of the target skills listed below.\n"
    # )
    
    guidance = (
    "Generate direct technical questions that test specific knowledge and understanding of the target skills. "
    "Focus on concise, pointed questions that can be answered clearly and demonstrate expertise.\n\n"
    
    "QUESTION STYLE - PRIORITIZE DIRECT QUESTIONS:\n"
    "• Use short, specific question stems that get straight to the point\n"
    "• Focus on facts, definitions, mechanisms, purposes, and comparisons\n"
    "• Avoid lengthy setups, scenarios, or multi-part questions\n"
    "• Each question should test one clear concept or piece of knowledge\n\n"
    
    # "PREFERRED QUESTION PATTERNS (use varied stems):\n"
    # "• 'What is...', 'What does...', 'What happens...'\n"
    # "• 'How does...', 'How would...', 'How can...'\n"
    # "• 'Why does...', 'Why would...', 'Why is...'\n"
    # "• 'When should...', 'When would...', 'When is...'\n"
    # "• 'Which parameter...', 'Which method...', 'Which approach...'\n"
    # "• 'What causes...', 'What determines...', 'What distinguishes...'\n"
    # "• 'Name the...', 'List the...', 'Identify the...'\n\n"
    
    "CONTENT FOCUS:\n"
    "• 80% should be direct knowledge questions: definitions, mechanisms, purposes, key differences\n"
    "• 20% can be implementation or troubleshooting questions (but keep them direct)\n"
    "• Test understanding of core concepts, parameters, methods, and technical distinctions\n"
    "• Include questions about when to use certain approaches or what specific components do\n\n"
    
    "STRICT RULES:\n"
    "• Every question stem must be unique - no repeated opening phrases in a batch\n"
    "• Do not repeat the same technical concept multiple times\n"
    "• Avoid: 'Explain the process...', 'Walk through...', 'Describe the architecture...', 'Describe a scenario...'\n"
    "• Keep questions short and focused\n"
    "• Never copy provided example questions\n"
    "• Each question must reference specific target skills"
    )
    
    request = (
        f"Target skill list:\n{bullet_lines}\n"
        f"Generate {n} high-signal interview questions (target 3–4) that test concrete ability with these skills.\n"
        "Assume the candidate has hands-on experience and challenge them accordingly.\n"
    )
    return "\n".join([intro, guidance, _SKILL_FEW_SHOT_EXAMPLE.strip(), request]).strip()

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
    skill_groups: List[List[str]]
    themes: list[str]
    scores: list[Dict[str, Any]]
    profile: Dict[str, Any]


def _append_result(state: GraphState, message: str) -> None:
    """Append a short message to the running textual summary."""
    if not message:
        return
    current = (state.get("result") or "").strip()
    if current:
        state["result"] = f"{current}\n{message}"
    else:
        state["result"] = message

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
    skills_all = list(getattr(profile, "skills", []) or [])
    skills_all = _filter_technical_skills(skills_all)
    state["skills"] = list(skills_all)
    state["themes"] = []

    skill_groups = cluster_skills(skills_all) if skills_all else []
    state["skill_groups"] = [list(g) for g in skill_groups]

    base_target = max(5, int(getattr(settings, "total_questions_from_jd", 10)))
    questions: list[Dict[str, Any]] = []
    desired_total = base_target

    if skill_groups:
        group_alloc = [1 for _ in skill_groups]
        total_alloc = sum(group_alloc)
        idx = 0
        while total_alloc < base_target:
            if group_alloc[idx] < 4:
                group_alloc[idx] += 1
                total_alloc += 1
            idx = (idx + 1) % len(group_alloc)
            if idx == 0 and all(a >= 4 for a in group_alloc):
                break
        desired_total = max(base_target, total_alloc)

        for g_idx, group in enumerate(skill_groups):
            if len(questions) >= desired_total:
                break
            need = group_alloc[g_idx]
            topic = _build_skill_topic(group, need)
            skill_slots = dict(slots)
            skill_slots["topic"] = topic
            skill_slots["n"] = str(need)
            try:
                qs = generate_from_topic(skill_slots, n=need)
            except Exception:
                qs = []
            if not qs:
                continue
            for q in qs[:need]:
                q.setdefault("meta", {})
                if isinstance(q["meta"], dict):
                    q["meta"].setdefault("skill_group", list(group))
            questions.extend(qs[:need])

    # If no skills detected or generation failed, fall back to a generic topic using JD text
    if not questions:
        jd_text = (slots.get("jd_text") or "").strip()
        fallback_topic = (
            "Generate practical interview questions for this job description.\n"
            "Focus on key skills and responsibilities mentioned.\n"
            f"JD snippet: {jd_text[:500]}"
        )
        fallback_slots = dict(slots)
        fallback_slots["topic"] = fallback_topic
        fallback_slots["n"] = str(base_target)
        try:
            fallback = generate_from_topic(fallback_slots, n=base_target)
        except Exception:
            fallback = []
        questions.extend(fallback[:base_target])

    final_cap = desired_total if skill_groups else base_target
    state["questions"] = _dedup_qs(questions)[:final_cap]

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
        if skill_groups:
            label = f"{len(skill_groups)} skill cluster(s)"
        else:
            label = "general prompts"
        state["result"] = (
            f"Generated {len(state['questions'])} questions across {label}"
        )
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
    auto_chain = state.pop("auto_chain_from_eval", False)
    slots = state.get("slots", {}) or {}
    if auto_chain and state.get("skip_recommendations"):
        state["resources"] = []
        _append_result(state, "No weaknesses detected; skipping resources.")
        return state
    if not slots.get("skill") and not slots.get("topic"):
        state["resources"] = []
        _append_result(state, "No target skill specified; skipping resources.")
        return state
    state.pop("skip_recommendations", None)

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
        _append_result(state, f"[RECOMMEND_RESOURCES] search failed: {e}")
        return state

    state["resources"] = resources
    skill = slots.get("skill") or slots.get("topic") or "your query"
    applied_prefs = slots.get("prefs") or {}
    suffix = " (prefs applied)" if applied_prefs else ""
    _append_result(state, f"Found {len(resources)} resources for {skill}{suffix}")
    return state

def show_profile(state: GraphState) -> GraphState:
    slots = state.get("slots", {}) or {}
    summary = summarize_profile(slots, session_id=state.get("session_id"), user_id=state.get("user_id"))
    state["profile"] = summary

    total = summary.get("total_answers", 0)
    avg = summary.get("avg_score", 0.0)
    strong = summary.get("strength_skills") or []
    weak = summary.get("weakness_skills") or []

    parts = [f"Profile: {total} answers, avg score {avg}."]
    if strong:
        parts.append(f"Strengths: {', '.join(strong[:3])}.")
    if weak:
        parts.append(f"Focus areas: {', '.join(weak[:3])}.")
    _append_result(state, " ".join(parts))

    auto_chain_from_eval = state.get("auto_chain_from_eval") or (
        state.get("intent") == IntentType.EVALUATE_ANSWERS.value
    )
    state["auto_chain_from_eval"] = auto_chain_from_eval

    # Auto-set the focus skill for downstream recommendations if none provided.
    focus_skill = weak[0] if weak else None
    if focus_skill and not slots.get("skill"):
        slots.setdefault("prefs", slots.get("prefs") or {})
        slots["skill"] = focus_skill
        state["slots"] = slots

    if auto_chain_from_eval:
        if weak:
            state["skip_recommendations"] = False
        else:
            if "skill" in slots and not focus_skill:
                slots = dict(slots)
                slots.pop("skill", None)
                state["slots"] = slots
            state["skip_recommendations"] = True
    else:
        state.pop("skip_recommendations", None)
        state.pop("auto_chain_from_eval", None)
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
    examples = [
        "Ingest this JD: …",
        "Generate questions on system design",
        "Evaluate these answers: …",
        "Show my profile",
        "Recommend resources for Kubernetes weaknesses",
    ]
    message = (
        "I can ingest job descriptions, generate tailored questions, evaluate your answers, "
        "summarize strengths & weaknesses, and recommend learning resources. "
        "Try prompts like:\n- " + "\n- ".join(examples)
    )
    state["result"] = message
    state["examples"] = examples
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
        "recommend_resources",
        "update_prefs",
        "help_node",
    ]:
        g.add_edge(node, END)

    # After ingesting a JD, automatically generate questions from it
    g.add_edge("ingest_jd", "qgen_from_jd")
    # After evaluations, surface profile + resource recommendations automatically
    g.add_edge("evaluate_answers", "show_profile")
    g.add_edge("show_profile", "recommend_resources")

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
