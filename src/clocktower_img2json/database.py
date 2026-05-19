from __future__ import annotations

import sqlite3
from pathlib import Path

DATA_DIR = Path("/app/data")
DB_PATH = DATA_DIR / "metadata.db"


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the audit tables used by the editor dashboard."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scripts (
                uuid TEXT PRIMARY KEY,
                creator TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edit_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                script_uuid TEXT,
                edited_by TEXT,
                edited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                change_summary TEXT,
                FOREIGN KEY (script_uuid) REFERENCES scripts(uuid)
            )
            """
        )
        conn.commit()


def create_script_record(
    script_uuid: str,
    creator: str | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """Insert or replace the audit record for a newly created script."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scripts (uuid, creator) VALUES (?, ?)",
            (script_uuid, creator),
        )
        conn.commit()


def script_record_exists(script_uuid: str, db_path: Path = DB_PATH) -> bool:
    """Return whether a script audit record exists."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM scripts WHERE uuid = ?",
            (script_uuid,),
        ).fetchone()
    return row is not None


def log_script_edit(
    script_uuid: str,
    edited_by: str | None = None,
    change_summary: str | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """Append a script edit event to the audit history table."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO edit_history (script_uuid, edited_by, change_summary)
            VALUES (?, ?, ?)
            """,
            (script_uuid, edited_by, change_summary),
        )
        conn.commit()
