import os
import sys
import pathlib
import streamlit as st

from typing import Any, Dict, List

def _render_questions(qs: List[Dict[str, Any]]):
    if not qs:
        return
    st.markdown("**Questions:**")
    for i, q in enumerate(qs, 1):
        txt = q.get("text") if isinstance(q, dict) else str(q)
        st.write(f"{i}. {txt}")


def _render_resources(items: List[Dict[str, Any]]):
    if not items:
        return
    st.markdown("**Resources:**")
    for i, it in enumerate(items, 1):
        title = it.get("title") or it.get("name") or it.get("text") or f"Item {i}"
        url = it.get("url") or it.get("link")
        if url:
            st.markdown(f"{i}. [{title}]({url})")
        else:
            st.write(f"{i}. {title}")



def _render_scores(scores: List[Dict[str, Any]]):
    if not scores:
        return
    st.markdown("**Scores:**")
    for i, sc in enumerate(scores, 1):
        score = sc.get("score")
        fb = sc.get("feedback") or sc.get("explanation")
        qtxt = sc.get("question_text")
        st.write(f"{i}. {score} — {fb}")
        if qtxt:
            st.caption(qtxt)


# New helpers for distribution and strengths/weaknesses
def _render_distribution(dist: Dict[str, Any]):
    if not dist:
        return
    st.markdown("**Distribution:**")
    for k, v in dist.items():
        st.write(f"- {k}: {v}")


def _render_list(title: str, items: List[Any]):
    if not items:
        return
    st.markdown(f"**{title}:**")
    for it in items:
        st.write(f"- {it}")

# New helper for styled badge lists
def _render_badge_list(title: str, items: List[Any], good: bool = True):
    if not items:
        return
    st.markdown(f"**{title}:**")
    icon = "✅" if good else "⚠️"
    for it in items:
        st.write(f"{icon} {it}")

# Ensure the same import path as CLI: PYTHONPATH=src
ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interview_guide.graph import build_graph

# Initialize graph once
app = build_graph()

st.set_page_config(page_title="Interview Guide", layout="wide")
st.title("📝 Interview Guide Assistant")

# Ensure a default user_id exists in session state
if "user_id" not in st.session_state:
    st.session_state.user_id = os.environ.get("USER_ID", "anon")
# Keep a private list of this browser's user IDs (not global)
if "my_user_ids" not in st.session_state:
    st.session_state.my_user_ids = [st.session_state.user_id]

# Sidebar: user switcher
st.sidebar.markdown("### User")
uid_in = st.sidebar.text_input("User ID", value=st.session_state.user_id, key="user_id_input")
if st.sidebar.button("Switch user"):
    new_id = uid_in.strip()
    if new_id:
        st.session_state.user_id = new_id
        if new_id not in st.session_state.my_user_ids:
            st.session_state.my_user_ids.append(new_id)

# Quick picker of IDs used in THIS browser only
if st.session_state.my_user_ids:
    try:
        current_index = st.session_state.my_user_ids.index(st.session_state.user_id)
    except ValueError:
        current_index = 0
    chosen = st.sidebar.selectbox("Your saved IDs", st.session_state.my_user_ids, index=current_index, key="saved_ids_select")
    if st.sidebar.button("Use selected"):
        st.session_state.user_id = chosen

st.sidebar.markdown(f"**Current User ID:** {st.session_state.user_id}")

# Tabs for cleaner layout
chat_tab, profile_tab, resources_tab, debug_tab = st.tabs(["Chat", "Profile", "Resources", "Debug"])

with chat_tab:
    query = st.text_area("Enter your query or paste a Job Description:", height=200)
    run_clicked = st.button("Run")

with profile_tab:
    st.caption("Run 'Show my profile' or evaluate answers to populate this tab.")

with resources_tab:
    st.caption("Ask for recommendations to populate this tab.")

# We'll store last_out in session so other tabs can read it
if "last_out" not in st.session_state:
    st.session_state.last_out = None

if run_clicked:
    if query and query.strip():
        payload = {"query": query, "user_id": st.session_state.user_id}
        try:
            out = app.invoke(payload)
            if hasattr(out, "model_dump"):
                out = out.model_dump(exclude_none=True)
            elif not isinstance(out, dict):
                out = {"result": str(out)}
            st.session_state.last_out = out
        except Exception as e:
            with chat_tab:
                st.subheader("Error (full traceback)")
                st.exception(e)
                try:
                    st.write("Payload:", payload)
                except Exception:
                    pass
    else:
        with chat_tab:
            st.warning("Please enter a query or JD first.")

# If we have output, render across tabs
out = st.session_state.last_out
if out:
    intent = out.get("intent") or out.get("type")
    result = out.get("result") or out.get("message")
    session_id = out.get("session_id")
    themes = out.get("themes") or out.get("jd_themes")
    questions = out.get("questions") or []
    resources = out.get("resources") or []
    scores = out.get("scores") or []
    profile = out.get("profile") or {}

    with chat_tab:
        st.subheader("Output")
        if intent:
            st.write(f"**Intent:** {intent}")
        if result:
            st.success(result)
        if themes:
            st.write("**Themes:** ", ", ".join(map(str, themes)))
        if session_id:
            st.write(f"**Session:** {session_id}")
        _render_questions(questions)
        _render_scores(scores)

    with resources_tab:
        _render_resources(resources)

    with profile_tab:
        _render_distribution(profile.get("distribution") or out.get("distribution") or {})
        _render_badge_list("Strong skills", profile.get("strength_skills") or [], good=True)
        _render_badge_list("Weak skills", profile.get("weakness_skills") or [], good=False)

    with debug_tab:
        st.json(out)