"""Tests for filesystem-first GET /api/script and audited POST updates."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from clocktower_img2json.api import create_app


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
SAMPLE_SCRIPT = [{"id": "_meta", "name": "Test Script"}, {"id": "washerwoman"}]
UPDATED_SCRIPT = [{"id": "_meta", "name": "Updated Script"}, {"id": "imp"}]


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "metadata.db"
    with patch("clocktower_img2json.api.init_db"), patch("clocktower_img2json.api.refresh_official_roles"):
        app = create_app(
            storage_dir=str(tmp_path),
            db_path=db_path,
            frontend_dir=str(FRONTEND_DIR),
        )
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
                change_summary TEXT
            )
            """
        )
        conn.commit()
    return TestClient(app, raise_server_exceptions=True), tmp_path, db_path


def _seed_script(tmp_path: Path, db_path: Path, uid: str, script: list) -> None:
    (tmp_path / uid).mkdir(parents=True, exist_ok=True)
    (tmp_path / uid / "script.json").write_text(json.dumps(script), encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO scripts (uuid, creator) VALUES (?, ?)", (uid, "tester"))
        conn.commit()


def test_get_script_reads_script_json_from_disk(client):
    tc, tmp_path, db_path = client
    uid = "aa11bb22"
    _seed_script(tmp_path, db_path, uid, SAMPLE_SCRIPT)

    response = tc.get(f"/api/script/{uid}")

    assert response.status_code == 200
    assert response.json() == SAMPLE_SCRIPT


def test_get_script_404_when_file_missing(client):
    tc, _, _ = client
    response = tc.get("/api/script/deadbeef")
    assert response.status_code == 404


def test_update_script_requires_existing_metadata_record(client):
    tc, tmp_path, _ = client
    uid = "cc33dd44"
    (tmp_path / uid).mkdir(parents=True, exist_ok=True)
    (tmp_path / uid / "script.json").write_text(json.dumps(SAMPLE_SCRIPT), encoding="utf-8")

    response = tc.post(f"/api/script/{uid}/update", json=UPDATED_SCRIPT)

    assert response.status_code == 404
    assert response.json()["detail"] == "Script metadata not found"


def test_update_script_overwrites_file(client):
    tc, tmp_path, db_path = client
    uid = "ee55ff66"
    _seed_script(tmp_path, db_path, uid, SAMPLE_SCRIPT)

    response = tc.post(f"/api/script/{uid}/update?edited_by=Builder", json=UPDATED_SCRIPT)

    assert response.status_code == 200
    saved = json.loads((tmp_path / uid / "script.json").read_text())
    assert saved == UPDATED_SCRIPT


def test_update_script_logs_edit_history(client):
    tc, tmp_path, db_path = client
    uid = "aabb1122"
    _seed_script(tmp_path, db_path, uid, SAMPLE_SCRIPT)

    tc.post(f"/api/script/{uid}/update?edited_by=Alice", json=UPDATED_SCRIPT)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT script_uuid, edited_by, change_summary FROM edit_history WHERE script_uuid = ?",
            (uid,),
        ).fetchone()
    assert row is not None
    assert row[0] == uid
    assert row[1] == "Alice"
    assert row[2] == f"Updated script with {len(UPDATED_SCRIPT)} entries"


def test_update_script_non_list_body_returns_400(client):
    tc, tmp_path, db_path = client
    uid = "ccdd3344"
    _seed_script(tmp_path, db_path, uid, SAMPLE_SCRIPT)

    response = tc.post(
        f"/api/script/{uid}/update",
        content=json.dumps({"not": "a list"}),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400


def test_update_script_bad_identifier_returns_400(client):
    tc, _, _ = client
    response = tc.post("/api/script/../../etc/update", json=UPDATED_SCRIPT)
    assert response.status_code in (400, 404, 422)
