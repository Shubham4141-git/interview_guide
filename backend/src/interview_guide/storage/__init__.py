"""Storage package: initialize SQLite DB and re-export helpers."""
from . import db as _db

# Initialize schema on first import (safe if tables already exist)
try:  # pragma: no cover
    _db.init()
except Exception:
    pass

# Re-export commonly used functions
init = _db.init
new_session = _db.new_session
add_questions = _db.add_questions
add_scores = _db.add_scores
get_scores_by_skill = _db.get_scores_by_skill
DB_PATH = _db.DB_PATH
get_questions = _db.get_questions
list_sessions = _db.list_sessions
latest_session_id = _db.latest_session_id