"""
SQLite storage (minimal)
- creates tables if missing
- save/load sessions, questions, scores
"""
from __future__ import annotations
import os, sqlite3
from typing import Iterable, Dict, Any, List, Optional
from pathlib import Path

DB_PATH = os.environ.get("INTERVIEW_GUIDE_DB", "interview_guide.db")

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  jd_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS questions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER,
  text TEXT NOT NULL,
  topic TEXT,
  difficulty TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER,
  question_text TEXT,
  answer TEXT,
  score INTEGER,
  feedback TEXT,
  confidence REAL,
  skill TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
"""

def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init():
    with _connect() as cx:
        cx.executescript(_SCHEMA)

def _migrate_add_skill_column():
    """Add `skill` column to scores if missing (safe no-op if present)."""
    with _connect() as cx:
        try:
            cur = cx.execute("PRAGMA table_info(scores)")
            cols = {row[1] for row in cur.fetchall()}  # name is at index 1
            if "skill" not in cols:
                cx.execute("ALTER TABLE scores ADD COLUMN skill TEXT")
        except Exception:
            # Best-effort; ignore if pragma/alter fails
            pass

def new_session(user_id: Optional[str] = None, jd_text: Optional[str] = None) -> int:
    user_id = user_id or "anon"
    with _connect() as cx:
        cur = cx.execute(
            "INSERT INTO sessions(user_id, jd_text) VALUES (?, ?)",
            (user_id, jd_text),
        )
        return int(cur.lastrowid)

def add_questions(session_id: int, questions: Iterable[Dict[str, Any]]):
    rows = [
        (session_id, q.get("text"), q.get("topic"), q.get("difficulty"))
        for q in questions
        if q.get("text")
    ]
    if not rows:
        return
    with _connect() as cx:
        cx.executemany(
            "INSERT INTO questions(session_id, text, topic, difficulty) VALUES (?, ?, ?, ?)",
            rows,
        )

def add_scores(session_id: int, scores: Iterable[Dict[str, Any]]):
    rows = [
        (
            session_id,
            s.get("question_text"),  # optional; may be None
            s.get("answer"),
            int(s.get("score", 0)),
            s.get("feedback"),
            float(s.get("confidence", 0.0)),
            s.get("skill"),
        )
        for s in scores
    ]
    if not rows:
        return
    with _connect() as cx:
        cx.executemany(
            "INSERT INTO scores(session_id, question_text, answer, score, feedback, confidence, skill) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

def get_scores_by_skill(session_id: Optional[int] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Simple read: returns all scores (later we’ll aggregate per skill/topic).
    """
    where = []
    args: List[Any] = []
    if session_id is not None:
        where.append("s.session_id = ?")
        args.append(session_id)
    if user_id is not None:
        where.append("sess.user_id = ?")
        args.append(user_id)

    sql = """
    SELECT s.id, s.session_id, s.question_text, s.answer, s.score, s.feedback, s.confidence, s.skill, sess.created_at
    FROM scores s
    JOIN sessions sess ON sess.id = s.session_id
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY s.id DESC"

    with _connect() as cx:
        cur = cx.execute(sql, args)
        return [dict(r) for r in cur.fetchall()]
    
def get_questions(session_id: int) -> List[Dict[str, Any]]:
    with _connect() as cx:
        cur = cx.execute(
            "SELECT id, text, topic, difficulty FROM questions WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def list_sessions(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Most-recent-first list of sessions for a given user."""
    with _connect() as cx:
        cur = cx.execute(
            "SELECT id, user_id, created_at, jd_text FROM sessions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def latest_session_id(user_id: str) -> Optional[int]:
    """Return the latest session id for a user, or None."""
    with _connect() as cx:
        cur = cx.execute(
            "SELECT id FROM sessions WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (user_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


def list_user_ids(limit: int = 50) -> List[str]:
    """Return distinct user IDs ordered by most recent activity."""
    sql = """
    SELECT user_id
    FROM (
        SELECT user_id, MAX(created_at) AS last_seen
        FROM sessions
        WHERE COALESCE(user_id, '') <> ''
        GROUP BY user_id
    )
    ORDER BY datetime(last_seen) DESC
    LIMIT ?
    """
    with _connect() as cx:
        cur = cx.execute(sql, (limit,))
        return [row["user_id"] for row in cur.fetchall()]


# Ensure tables/indexes exist when module is imported
init()
_migrate_add_skill_column()
