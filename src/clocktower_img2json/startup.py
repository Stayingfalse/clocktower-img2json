from __future__ import annotations

import json
import logging
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("/app/data")
DB_PATH = DATA_DIR / "scripts.db"
OFFICIAL_ROLES_PATH = DATA_DIR / "official_roles.json"
ROLES_URL = (
    "https://github.com/ThePandemoniumInstitute/botc-release"
    "/raw/main/resources/data/roles.json"
)


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the scripts database and table if they do not already exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scripts (
                uuid        TEXT PRIMARY KEY,
                name        TEXT,
                custom_data TEXT
            )
            """
        )
        conn.commit()
    logger.info("Database initialised at %s", db_path)


def refresh_official_roles(
    roles_path: Path = OFFICIAL_ROLES_PATH,
    roles_url: str = ROLES_URL,
) -> None:
    """Download the latest official roles JSON and persist it locally.

    If the network request fails and a cached copy already exists, the cached
    copy is kept and the error is logged as a warning.  If there is no cached
    copy at all, the exception is re-raised so the caller can decide how to
    handle a cold-start failure.
    """
    roles_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(
            roles_url,
            headers={"User-Agent": "clocktower-img2json/0.1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
            raw = response.read()
        # Validate the payload is parseable JSON before overwriting the cache.
        json.loads(raw)
        roles_path.write_bytes(raw)
        logger.info("Official roles refreshed and saved to %s", roles_path)
    except Exception as exc:
        if roles_path.exists():
            logger.warning(
                "Failed to refresh official roles (%s); using cached file at %s",
                exc,
                roles_path,
            )
        else:
            logger.error(
                "Failed to fetch official roles and no local cache exists: %s", exc
            )
            raise


def get_official_roles(roles_path: Path = OFFICIAL_ROLES_PATH) -> list:
    """Return the official roles as a parsed list read from the local cache file."""
    with roles_path.open("r", encoding="utf-8") as f:
        return json.load(f)
