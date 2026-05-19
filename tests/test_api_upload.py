"""Tests for the upload endpoint, dashboard pages, and /script/* asset routes."""
from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from clocktower_img2json.api import create_app
from clocktower_img2json.data import OfficialRole


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def _make_png_bytes(width: int = 200, height: int = 300) -> bytes:
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    top_cutoff = int(height * 0.15)
    cv2.rectangle(img, (5, top_cutoff + 10), (180, top_cutoff + 80), (30, 30, 30), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


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


SAMPLE_OFFICIAL_ROLES = [
    OfficialRole(id="washerwoman", name="Washerwoman", team="townsfolk", ability="..."),
]


def _seed_script(tmp_path: Path, uid: str, script: list) -> None:
    d = tmp_path / uid
    d.mkdir(parents=True, exist_ok=True)
    (d / "script.json").write_text(json.dumps(script), encoding="utf-8")


def test_index_page_serves_upload_dashboard(client):
    tc, _, _ = client
    response = tc.get("/")
    assert response.status_code == 200
    assert "Process &amp; Open Dashboard" in response.text


def test_edit_page_serves_dashboard(client):
    tc, _, _ = client
    response = tc.get("/dashboard/edit.html")
    assert response.status_code == 200
    assert "Save Changes" in response.text


def test_upload_returns_uuid_and_script(client):
    tc, _, _ = client
    png_bytes = _make_png_bytes()
    fake_icon = np.zeros((60, 60, 3), dtype=np.uint8)
    fake_rows = [
        {"raw_name": "Washerwoman", "ability": "You start knowing something.", "icon_crop": fake_icon},
        {"raw_name": "My Homebrew", "ability": "Custom ability.", "icon_crop": fake_icon},
    ]

    with patch("clocktower_img2json.api.process_script_image", return_value=("Test Script", fake_rows)), patch(
        "clocktower_img2json.api.get_official_roles", return_value=SAMPLE_OFFICIAL_ROLES
    ):
        response = tc.post(
            "/api/upload",
            files={"image": ("script.png", io.BytesIO(png_bytes), "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["uuid"]) == 8
    assert data["script"][0] == {"id": "_meta", "name": "Test Script"}


def test_upload_saves_script_json_and_logo_to_disk(client):
    tc, tmp_path, _ = client
    png_bytes = _make_png_bytes()

    with patch("clocktower_img2json.api.process_script_image", return_value=("Saved Script", [])), patch(
        "clocktower_img2json.api.get_official_roles", return_value=[]
    ):
        response = tc.post(
            "/api/upload",
            files={"image": ("script.png", io.BytesIO(png_bytes), "image/png")},
        )

    uid = response.json()["uuid"]
    assert (tmp_path / uid / "script.json").exists()
    assert (tmp_path / uid / "scriptlogo.png").exists()


def test_upload_records_only_metadata_in_db(client):
    tc, _, db_path = client
    png_bytes = _make_png_bytes()

    with patch("clocktower_img2json.api.process_script_image", return_value=("DB Script", [])), patch(
        "clocktower_img2json.api.get_official_roles", return_value=[]
    ):
        response = tc.post(
            "/api/upload",
            files={"image": ("script.png", io.BytesIO(png_bytes), "image/png")},
            data={"creator": "Uploader"},
        )

    uid = response.json()["uuid"]
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT uuid, creator FROM scripts WHERE uuid=?", (uid,)).fetchone()
    assert row == (uid, "Uploader")


def test_upload_homebrew_role_uses_script_asset_route(client):
    tc, _, _ = client
    png_bytes = _make_png_bytes()
    fake_icon = np.zeros((60, 60, 3), dtype=np.uint8)
    fake_rows = [{"raw_name": "My Homebrew", "ability": "Does something.", "icon_crop": fake_icon}]

    with patch("clocktower_img2json.api.process_script_image", return_value=("Script", fake_rows)), patch(
        "clocktower_img2json.api.get_official_roles", return_value=[]
    ):
        response = tc.post(
            "/api/upload",
            files={"image": ("script.png", io.BytesIO(png_bytes), "image/png")},
        )

    role = response.json()["script"][1]
    assert role["image"].startswith("/script/")
    assert role["image"].endswith("script.my-homebrew.png")


def test_upload_empty_file_returns_400(client):
    tc, _, _ = client
    response = tc.post(
        "/api/upload",
        files={"image": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert response.status_code == 400


def test_get_script_json_returns_file(client):
    tc, tmp_path, _ = client
    uid = "abc12345"
    script = [{"id": "_meta", "name": "The Script"}]
    _seed_script(tmp_path, uid, script)

    response = tc.get(f"/script/{uid}/script.json")
    assert response.status_code == 200
    assert response.json() == script


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


def test_get_asset_path_traversal_blocked(client):
    tc, _, _ = client
    response = tc.get("/script/abc12345/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in (400, 404)
