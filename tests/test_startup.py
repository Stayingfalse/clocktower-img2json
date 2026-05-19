import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clocktower_img2json.startup import (
    get_official_roles,
    init_db,
    refresh_official_roles,
)


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

def test_init_db_creates_table():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "subdir" / "scripts.db"
        init_db(db_path=db_path)

        assert db_path.exists()
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='scripts'"
            ).fetchall()
        assert rows == [("scripts",)]


def test_init_db_column_schema():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "scripts.db"
        init_db(db_path=db_path)

        with sqlite3.connect(db_path) as conn:
            info = conn.execute("PRAGMA table_info(scripts)").fetchall()
        col_names = [row[1] for row in info]
        assert col_names == ["uuid", "name", "custom_data"]


def test_init_db_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "scripts.db"
        init_db(db_path=db_path)
        init_db(db_path=db_path)  # should not raise


# ---------------------------------------------------------------------------
# refresh_official_roles
# ---------------------------------------------------------------------------

SAMPLE_ROLES = [{"id": "imp", "name": "Imp", "team": "demon", "ability": "..."}]


def test_refresh_official_roles_success():
    with tempfile.TemporaryDirectory() as tmp:
        roles_path = Path(tmp) / "official_roles.json"
        payload = json.dumps(SAMPLE_ROLES).encode()

        mock_response = MagicMock()
        mock_response.read.return_value = payload
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("clocktower_img2json.startup.urllib.request.urlopen", return_value=mock_response):
            refresh_official_roles(roles_path=roles_path)

        assert roles_path.exists()
        assert json.loads(roles_path.read_bytes()) == SAMPLE_ROLES


def test_refresh_official_roles_network_failure_uses_cache():
    with tempfile.TemporaryDirectory() as tmp:
        roles_path = Path(tmp) / "official_roles.json"
        # Pre-populate cache
        roles_path.write_text(json.dumps(SAMPLE_ROLES), encoding="utf-8")

        with patch(
            "clocktower_img2json.startup.urllib.request.urlopen",
            side_effect=OSError("network down"),
        ):
            refresh_official_roles(roles_path=roles_path)  # should not raise

        # Cache must still be intact
        assert json.loads(roles_path.read_text()) == SAMPLE_ROLES


def test_refresh_official_roles_network_failure_no_cache_raises():
    with tempfile.TemporaryDirectory() as tmp:
        roles_path = Path(tmp) / "official_roles.json"
        assert not roles_path.exists()

        with patch(
            "clocktower_img2json.startup.urllib.request.urlopen",
            side_effect=OSError("network down"),
        ):
            with pytest.raises(OSError):
                refresh_official_roles(roles_path=roles_path)


# ---------------------------------------------------------------------------
# get_official_roles
# ---------------------------------------------------------------------------

def test_get_official_roles_returns_parsed_list():
    with tempfile.TemporaryDirectory() as tmp:
        roles_path = Path(tmp) / "official_roles.json"
        roles_path.write_text(json.dumps(SAMPLE_ROLES), encoding="utf-8")

        result = get_official_roles(roles_path=roles_path)

    assert isinstance(result, list)
    assert result[0]["id"] == "imp"
