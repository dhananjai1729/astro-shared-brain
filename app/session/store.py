"""Short-term memory (SQLite): recent turns per session, a profile cache for when the graph is down,
and an outbox of memory writes that could not reach the graph."""
import json
import sqlite3
import threading
from datetime import datetime, timezone


class SessionStore:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, session_id TEXT,
                    role TEXT, content TEXT, ts TEXT);
                CREATE INDEX IF NOT EXISTS idx_msg ON messages(user_id, session_id, id);
                CREATE TABLE IF NOT EXISTS profile_cache(user_id TEXT PRIMARY KEY, data TEXT);
                CREATE TABLE IF NOT EXISTS outbox(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, payload TEXT,
                    attempts INTEGER DEFAULT 0, created_at TEXT);
            """)

    def add_turn(self, user_id: str, session_id: str, role: str, content: str):
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO messages(user_id,session_id,role,content,ts) VALUES(?,?,?,?,?)",
                (user_id, session_id, role, content, datetime.now(timezone.utc).isoformat()))

    def recent(self, user_id: str, session_id: str, n: int) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT role, content FROM messages WHERE user_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
                (user_id, session_id, n)).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def cache_profile(self, user_id: str, profile: dict):
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO profile_cache VALUES(?,?)", (user_id, json.dumps(profile)))

    def cached_profile(self, user_id: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT data FROM profile_cache WHERE user_id=?", (user_id,)).fetchone()
        return json.loads(row[0]) if row else None

    # outbox
    def enqueue(self, user_id: str, payload: dict):
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO outbox(user_id,payload,created_at) VALUES(?,?,?)",
                (user_id, json.dumps(payload), datetime.now(timezone.utc).isoformat()))

    def pending(self, limit: int = 50) -> list[tuple[int, str, dict]]:
        with self.lock:
            rows = self.conn.execute("SELECT id,user_id,payload FROM outbox ORDER BY id LIMIT ?", (limit,)).fetchall()
        return [(i, u, json.loads(p)) for i, u, p in rows]

    def pending_count(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]

    def ack(self, outbox_id: int):
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM outbox WHERE id=?", (outbox_id,))
