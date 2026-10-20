"""
Context Store: on-device SQLite database (no cloud).

Tables
    events    raw events exactly as received (JSON payload kept as text)
    sessions  one row of aggregated features per (user, session)
"""

import json
import sqlite3

from .schema import validate_event

FEATURE_COLUMNS = (
    "click_error_rate", "voice_command_latency_ms", "correction_frequency",
    "gesture_stability", "modality_share_voice", "modality_share_gesture",
    "modality_share_screen", "completion_accuracy",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(user_id, session_id);

CREATE TABLE IF NOT EXISTS sessions (
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    session_index INTEGER NOT NULL,
    start_ts TEXT NOT NULL,
    n_events INTEGER NOT NULL,
    click_error_rate REAL, voice_command_latency_ms REAL, correction_frequency REAL,
    gesture_stability REAL, modality_share_voice REAL, modality_share_gesture REAL,
    modality_share_screen REAL, completion_accuracy REAL,
    label_decline INTEGER,
    PRIMARY KEY (user_id, session_id)
);
"""


class ContextStore:
    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)

    def close(self):
        self.conn.close()

    # ---- raw events -------------------------------------------------
    def add_events(self, events):
        rows = []
        for e in events:
            validate_event(e)
            rows.append((e["user_id"], e["session_id"], e["timestamp"],
                         e["source"], json.dumps(e["payload"])))
        with self.conn:
            self.conn.executemany(
                "INSERT INTO events (user_id, session_id, timestamp, source, payload) "
                "VALUES (?,?,?,?,?)", rows)
        return len(rows)

    def get_events(self, user_id, session_id):
        cur = self.conn.execute(
            "SELECT * FROM events WHERE user_id=? AND session_id=? ORDER BY timestamp",
            (user_id, session_id))
        return [{"user_id": r["user_id"], "session_id": r["session_id"],
                 "timestamp": r["timestamp"], "source": r["source"],
                 "payload": json.loads(r["payload"])} for r in cur]

    def list_session_ids(self, user_id):
        """Session ids of a user, ordered by their first event time."""
        cur = self.conn.execute(
            "SELECT session_id, MIN(timestamp) AS t FROM events WHERE user_id=? "
            "GROUP BY session_id ORDER BY t", (user_id,))
        return [(r["session_id"], r["t"]) for r in cur]

    def list_users(self):
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT user_id FROM events ORDER BY user_id")]

    # ---- session features -------------------------------------------
    def upsert_session(self, user_id, session_id, session_index, start_ts,
                       n_events, features, label=None):
        cols = ", ".join(FEATURE_COLUMNS)
        marks = ", ".join("?" for _ in FEATURE_COLUMNS)
        with self.conn:
            self.conn.execute(
                f"INSERT OR REPLACE INTO sessions (user_id, session_id, session_index, "
                f"start_ts, n_events, {cols}, label_decline) VALUES (?,?,?,?,?,{marks},?)",
                (user_id, session_id, session_index, start_ts, n_events,
                 *[features[c] for c in FEATURE_COLUMNS], label))

    def get_user_sessions(self, user_id):
        cur = self.conn.execute(
            "SELECT * FROM sessions WHERE user_id=? ORDER BY session_index", (user_id,))
        return [dict(r) for r in cur]

    def count(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
