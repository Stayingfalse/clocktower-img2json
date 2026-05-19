from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

from .database import DB_PATH, DATA_DIR, init_db

logger = logging.getLogger(__name__)

OFFICIAL_ROLES_PATH = DATA_DIR / "official_roles.json"
ROLES_URL = (
    "https://github.com/ThePandemoniumInstitute/botc-release"
    "/raw/main/resources/data/roles.json"
)


def refresh_official_roles(
    roles_path: Path = OFFICIAL_ROLES_PATH,
    roles_url: str = ROLES_URL,
) -> None:
    """Download the latest official roles JSON and persist it locally."""
    roles_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(
            roles_url,
            headers={"User-Agent": "clocktower-img2json/0.1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
            raw = response.read()
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
                "Failed to fetch official roles and no local cache exists: %s",
                exc,
            )
            raise


def get_official_roles(roles_path: Path = OFFICIAL_ROLES_PATH) -> list:
    """Return the official roles as a parsed list read from the local cache file."""
    with roles_path.open("r", encoding="utf-8") as f:
        return json.load(f)
