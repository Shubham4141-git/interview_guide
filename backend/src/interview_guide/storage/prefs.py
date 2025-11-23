"""
Simple preferences storage on SQLite.
- set_prefs(user_id, prefs_dict)
- get_prefs(user_id) -> dict
"""
from __future__ import annotations
import json, sqlite3, os
from pathlib import Path
from typing import Any, Dict, Optional

DB_PATH = os.environ.get("INTERVIEW_GUIDE_DB", "interview_guide.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_prefs (
  user_id TEXT PRIMARY KEY,
  prefs_json TEXT NOT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(DB_PATH)
    cx.row_factory = sqlite3.Row
    return cx

def init():
    with _connect() as cx:
        cx.executescript(_SCHEMA)

def set_prefs(user_id: str, prefs: Dict[str, Any]) -> None:
    if not user_id:
        user_id = "default"
    init()
    payload = json.dumps(prefs or {}, ensure_ascii=False)
    with _connect() as cx:
        cx.execute(
            """
            INSERT INTO user_prefs(user_id, prefs_json)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              prefs_json=excluded.prefs_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (user_id, payload),
        )

def get_prefs(user_id: Optional[str]) -> Dict[str, Any]:
    if not user_id:
        user_id = "default"
    init()
    with _connect() as cx:
        cur = cx.execute("SELECT prefs_json FROM user_prefs WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["prefs_json"] or "{}")
        except Exception:
            return {}