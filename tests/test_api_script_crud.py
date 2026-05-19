"""Tests for GET /api/script/{uuid} and POST /api/script/{uuid}/update."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from clocktower_img2json.api import create_app


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "scripts.db"
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

    with patch("clocktower_img2json.api.init_db"), \
         patch("clocktower_img2json.api.refresh_official_roles"):
        app = create_app(storage_dir=str(tmp_path), db_path=db_path)

    return TestClient(app, raise_server_exceptions=True), tmp_path, db_path


def _seed(db_path: Path, uid: str, name: str, script: list) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO scripts (uuid, name, custom_data) VALUES (?, ?, ?)",
            (uid, name, json.dumps(script)),
        )
        conn.commit()


SAMPLE_SCRIPT = [
    {"id": "_meta", "name": "Test Script"},
    {"id": "washerwoman"},
]


# ---------------------------------------------------------------------------
# GET /api/script/{uuid_str}
# ---------------------------------------------------------------------------

def test_get_script_returns_json(client):
    tc, tmp_path, db_path = client
    uid = "aa11bb22"
    _seed(db_path, uid, "Test Script", SAMPLE_SCRIPT)

    response = tc.get(f"/api/script/{uid}")

    assert response.status_code == 200
    assert response.json() == SAMPLE_SCRIPT


def test_get_script_404_when_missing(client):
    tc, tmp_path, db_path = client

    response = tc.get("/api/script/deadbeef")

    assert response.status_code == 404


def test_get_script_400_bad_identifier(client):
    tc, tmp_path, db_path = client

    response = tc.get("/api/script/../etc/passwd")

    assert response.status_code in (400, 404, 422)


# ---------------------------------------------------------------------------
# POST /api/script/{uuid_str}/update
# ---------------------------------------------------------------------------

UPDATED_SCRIPT = [
    {"id": "_meta", "name": "Updated Script"},
    {"id": "imp"},
]


def test_update_script_returns_ok(client):
    tc, tmp_path, db_path = client
    uid = "cc33dd44"
    _seed(db_path, uid, "Updated Script", SAMPLE_SCRIPT)
    (tmp_path / uid).mkdir(parents=True, exist_ok=True)

    response = tc.post(f"/api/script/{uid}/update", json=UPDATED_SCRIPT)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["uuid"] == uid


def test_update_script_overwrites_file(client):
    tc, tmp_path, db_path = client
    uid = "ee55ff66"
    _seed(db_path, uid, "Script", SAMPLE_SCRIPT)
    (tmp_path / uid).mkdir(parents=True, exist_ok=True)

    tc.post(f"/api/script/{uid}/update", json=UPDATED_SCRIPT)

    saved = json.loads((tmp_path / uid / "script.json").read_text())
    assert saved == UPDATED_SCRIPT


def test_update_script_syncs_db(client):
    tc, tmp_path, db_path = client
    uid = "aabb1122"
    _seed(db_path, uid, "Script", SAMPLE_SCRIPT)
    (tmp_path / uid).mkdir(parents=True, exist_ok=True)

    tc.post(f"/api/script/{uid}/update", json=UPDATED_SCRIPT)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT custom_data FROM scripts WHERE uuid = ?", (uid,)
        ).fetchone()
    assert row is not None
    assert json.loads(row[0]) == UPDATED_SCRIPT


def test_update_script_non_list_body_returns_400(client):
    tc, tmp_path, db_path = client
    uid = "ccdd3344"
    _seed(db_path, uid, "Script", SAMPLE_SCRIPT)

    response = tc.post(
        f"/api/script/{uid}/update",
        content=json.dumps({"not": "a list"}),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400


def test_update_script_bad_identifier_returns_400(client):
    tc, tmp_path, db_path = client

    response = tc.post(
        "/api/script/../../etc/update",
        json=UPDATED_SCRIPT,
    )

    assert response.status_code in (400, 404, 422)
