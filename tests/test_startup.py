import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clocktower_img2json.database import (
    create_script_record,
    init_db,
    log_script_edit,
    script_record_exists,
)
from clocktower_img2json.startup import get_official_roles, refresh_official_roles


SAMPLE_ROLES = [{"id": "imp", "name": "Imp", "team": "demon", "ability": "..."}]


def test_init_db_creates_audit_tables():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "subdir" / "metadata.db"
        init_db(db_path=db_path)

        assert db_path.exists()
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        assert rows == [("edit_history",), ("scripts",), ("sqlite_sequence",)]


def test_init_db_column_schema():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "metadata.db"
        init_db(db_path=db_path)

        with sqlite3.connect(db_path) as conn:
            scripts_info = conn.execute("PRAGMA table_info(scripts)").fetchall()
            edits_info = conn.execute("PRAGMA table_info(edit_history)").fetchall()

        assert [row[1] for row in scripts_info] == ["uuid", "creator", "created_at"]
        assert [row[1] for row in edits_info] == [
            "id",
            "script_uuid",
            "edited_by",
            "edited_at",
            "change_summary",
        ]


def test_create_script_record_and_exists():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "metadata.db"
        init_db(db_path=db_path)

        create_script_record("deadbeef", creator="tester", db_path=db_path)

        assert script_record_exists("deadbeef", db_path=db_path) is True
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT uuid, creator FROM scripts WHERE uuid = ?", ("deadbeef",)).fetchone()
        assert row == ("deadbeef", "tester")


def test_log_script_edit_appends_audit_row():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "metadata.db"
        init_db(db_path=db_path)
        create_script_record("deadbeef", creator="tester", db_path=db_path)

        log_script_edit(
            "deadbeef",
            edited_by="alice",
            change_summary="Updated team assignments",
            db_path=db_path,
        )

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT script_uuid, edited_by, change_summary FROM edit_history WHERE script_uuid = ?",
                ("deadbeef",),
            ).fetchone()
        assert row == ("deadbeef", "alice", "Updated team assignments")


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
        roles_path.write_text(json.dumps(SAMPLE_ROLES), encoding="utf-8")

        with patch(
            "clocktower_img2json.startup.urllib.request.urlopen",
            side_effect=OSError("network down"),
        ):
            refresh_official_roles(roles_path=roles_path)

        assert json.loads(roles_path.read_text()) == SAMPLE_ROLES


def test_refresh_official_roles_network_failure_no_cache_raises():
    with tempfile.TemporaryDirectory() as tmp:
        roles_path = Path(tmp) / "official_roles.json"

        with patch(
            "clocktower_img2json.startup.urllib.request.urlopen",
            side_effect=OSError("network down"),
        ):
            with pytest.raises(OSError):
                refresh_official_roles(roles_path=roles_path)


def test_get_official_roles_returns_parsed_list():
    with tempfile.TemporaryDirectory() as tmp:
        roles_path = Path(tmp) / "official_roles.json"
        roles_path.write_text(json.dumps(SAMPLE_ROLES), encoding="utf-8")

        result = get_official_roles(roles_path=roles_path)

    assert isinstance(result, list)
    assert result[0]["id"] == "imp"
