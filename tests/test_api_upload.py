"""Tests for the /api/upload endpoint and /script/* asset routes."""
from __future__ import annotations

import io
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from clocktower_img2json.api import create_app
from clocktower_img2json.data import OfficialRole


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int = 200, height: int = 300) -> bytes:
    """Return the bytes of a simple synthetic PNG image."""
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    # Add a dark rectangle in the body to give OpenCV contours to find
    top_cutoff = int(height * 0.15)
    cv2.rectangle(img, (5, top_cutoff + 10), (180, top_cutoff + 80), (30, 30, 30), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def client(tmp_path):
    """Create a TestClient backed by a temporary storage directory."""
    db_path = tmp_path / "scripts.db"
    # Pre-create the DB so the upload can write to it without needing /app/data
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


# ---------------------------------------------------------------------------
# POST /api/upload — basic happy path
# ---------------------------------------------------------------------------

SAMPLE_OFFICIAL_ROLES = [
    OfficialRole(id="washerwoman", name="Washerwoman", team="townsfolk", ability="..."),
]


def _ocr_side_effect(call_responses: list[str]):
    it = iter(call_responses)

    def _inner(*_a, **_kw):
        return next(it, "")

    return _inner


def test_upload_returns_uuid_and_script(client):
    tc, tmp_path, db_path = client
    png_bytes = _make_png_bytes()

    # Patch OCR: first call = top section (script name), subsequent = row name+ability
    # Patch process_script_image to return a controlled result
    fake_icon = np.zeros((60, 60, 3), dtype=np.uint8)
    fake_rows = [
        {"raw_name": "Washerwoman", "ability": "You start knowing something.", "icon_crop": fake_icon},
        {"raw_name": "My Homebrew", "ability": "Custom ability.", "icon_crop": fake_icon},
    ]

    with patch("clocktower_img2json.api.process_script_image", return_value=("Test Script", fake_rows)), \
         patch("clocktower_img2json.api.get_official_roles", return_value=SAMPLE_OFFICIAL_ROLES):
        response = tc.post(
            "/api/upload",
            files={"image": ("script.png", io.BytesIO(png_bytes), "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "uuid" in data
    assert "script" in data
    assert isinstance(data["uuid"], str)
    assert len(data["uuid"]) == 8


def test_upload_script_has_meta_block(client):
    tc, tmp_path, db_path = client
    png_bytes = _make_png_bytes()
    fake_icon = np.zeros((60, 60, 3), dtype=np.uint8)

    with patch("clocktower_img2json.api.process_script_image", return_value=("My Script", [])), \
         patch("clocktower_img2json.api.get_official_roles", return_value=[]):
        response = tc.post(
            "/api/upload",
            files={"image": ("script.png", io.BytesIO(png_bytes), "image/png")},
        )

    assert response.status_code == 200
    script = response.json()["script"]
    assert script[0] == {"id": "_meta", "name": "My Script"}


def test_upload_official_role_appended_as_id_only(client):
    tc, tmp_path, db_path = client
    png_bytes = _make_png_bytes()
    fake_icon = np.zeros((60, 60, 3), dtype=np.uint8)
    fake_rows = [{"raw_name": "Washerwoman", "ability": "...", "icon_crop": fake_icon}]

    with patch("clocktower_img2json.api.process_script_image", return_value=("Script", fake_rows)), \
         patch("clocktower_img2json.api.get_official_roles", return_value=SAMPLE_OFFICIAL_ROLES):
        response = tc.post(
            "/api/upload",
            files={"image": ("script.png", io.BytesIO(png_bytes), "image/png")},
        )

    script = response.json()["script"]
    role_entries = [e for e in script if e.get("id") != "_meta"]
    assert role_entries == [{"id": "washerwoman"}]


def test_upload_homebrew_role_has_full_schema(client):
    tc, tmp_path, db_path = client
    png_bytes = _make_png_bytes()
    fake_icon = np.zeros((60, 60, 3), dtype=np.uint8)
    fake_rows = [{"raw_name": "My Homebrew", "ability": "Does something.", "icon_crop": fake_icon}]

    with patch("clocktower_img2json.api.process_script_image", return_value=("Script", fake_rows)), \
         patch("clocktower_img2json.api.get_official_roles", return_value=[]):
        response = tc.post(
            "/api/upload",
            files={"image": ("script.png", io.BytesIO(png_bytes), "image/png")},
        )

    script = response.json()["script"]
    role_entries = [e for e in script if e.get("id") != "_meta"]
    assert len(role_entries) == 1
    role = role_entries[0]
    assert role["id"] == "my-homebrew"
    assert role["name"] == "My Homebrew"
    assert role["ability"] == "Does something."
    assert role["team"] == "townsfolk"
    assert "image" in role
    assert role["image"].startswith("/script-assets/")
    assert role["image"].endswith("script.my-homebrew.png")


def test_upload_saves_script_json_to_disk(client):
    tc, tmp_path, db_path = client
    png_bytes = _make_png_bytes()
    fake_icon = np.zeros((60, 60, 3), dtype=np.uint8)

    with patch("clocktower_img2json.api.process_script_image", return_value=("Saved Script", [])), \
         patch("clocktower_img2json.api.get_official_roles", return_value=[]):
        response = tc.post(
            "/api/upload",
            files={"image": ("script.png", io.BytesIO(png_bytes), "image/png")},
        )

    uid = response.json()["uuid"]
    script_path = tmp_path / uid / "script.json"
    assert script_path.exists()
    saved = json.loads(script_path.read_text())
    assert saved[0]["name"] == "Saved Script"


def test_upload_inserts_into_db(client):
    tc, tmp_path, db_path = client
    png_bytes = _make_png_bytes()

    with patch("clocktower_img2json.api.process_script_image", return_value=("DB Script", [])), \
         patch("clocktower_img2json.api.get_official_roles", return_value=[]):
        response = tc.post(
            "/api/upload",
            files={"image": ("script.png", io.BytesIO(png_bytes), "image/png")},
        )

    uid = response.json()["uuid"]
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT uuid, name FROM scripts WHERE uuid=?", (uid,)).fetchone()
    assert row is not None
    assert row[0] == uid
    assert row[1] == "DB Script"


def test_upload_empty_file_returns_400(client):
    tc, tmp_path, db_path = client
    with patch("clocktower_img2json.api.process_script_image", return_value=("X", [])), \
         patch("clocktower_img2json.api.get_official_roles", return_value=[]):
        response = tc.post(
            "/api/upload",
            files={"image": ("empty.png", io.BytesIO(b""), "image/png")},
        )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /script/{uuid}/script.json
# ---------------------------------------------------------------------------

def _seed_script(tmp_path: Path, uid: str, script: list) -> None:
    d = tmp_path / uid
    d.mkdir(parents=True, exist_ok=True)
    (d / "script.json").write_text(json.dumps(script), encoding="utf-8")


def test_get_script_json_returns_file(client):
    tc, tmp_path, _ = client
    uid = "abc12345"
    script = [{"id": "_meta", "name": "The Script"}]
    _seed_script(tmp_path, uid, script)

    response = tc.get(f"/script/{uid}/script.json")
    assert response.status_code == 200
    assert response.json() == script


def test_get_script_json_404(client):
    tc, tmp_path, _ = client
    response = tc.get("/script/nonexistent/script.json")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /script/{uuid}/scriptlogo.png
# ---------------------------------------------------------------------------

def test_get_scriptlogo_returns_file(client):
    tc, tmp_path, _ = client
    uid = "logo1234"
    d = tmp_path / uid
    d.mkdir(parents=True, exist_ok=True)
    logo = np.full((150, 600, 3), 64, dtype=np.uint8)
    cv2.imwrite(str(d / "scriptlogo.png"), logo)

    response = tc.get(f"/script/{uid}/scriptlogo.png")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_get_scriptlogo_404(client):
    tc, tmp_path, _ = client
    response = tc.get("/script/missing/scriptlogo.png")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /script/{uuid}/{asset_name}  — wildcard homebrew icons
# ---------------------------------------------------------------------------

def test_get_asset_returns_file(client):
    tc, tmp_path, _ = client
    uid = "asset123"
    d = tmp_path / uid
    d.mkdir(parents=True, exist_ok=True)
    icon = np.zeros((60, 60, 3), dtype=np.uint8)
    cv2.imwrite(str(d / "script.my-role.png"), icon)

    response = tc.get(f"/script/{uid}/script.my-role.png")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_get_asset_404(client):
    tc, tmp_path, _ = client
    response = tc.get("/script/abc12345/script.nobody.png")
    assert response.status_code == 404


def test_get_asset_path_traversal_blocked(client):
    tc, tmp_path, _ = client
    # Attempt to escape the storage dir via a path-traversal asset name
    response = tc.get("/script/abc12345/..%2F..%2Fetc%2Fpasswd")
    # FastAPI URL-decodes before routing; the resolved path should not exist and
    # the server should return 400 (traversal detected) or 404 (path not found).
    assert response.status_code in (400, 404)
