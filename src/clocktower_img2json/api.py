from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .converter import convert_image_bytes_to_script
from .data import get_official_roles
from .ocv_processor import process_script_image
from .startup import DB_PATH, init_db, refresh_official_roles


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    refresh_official_roles()
    yield


def _slugify(text: str) -> str:
    """Convert a role name to a filesystem-safe lowercase ASCII slug."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned[:50]


_SAFE_UUID_RE = re.compile(r"^[a-zA-Z0-9\-]{1,64}$")
_SAFE_ASSET_RE = re.compile(r"^[a-zA-Z0-9_.\-]{1,128}$")


def create_app(storage_dir: str = "storage", db_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="clocktower-img2json", version="0.1.0", lifespan=_lifespan)
    storage_path = Path(storage_dir).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)
    _db_path: Path = db_path if db_path is not None else DB_PATH

    app.mount("/assets", StaticFiles(directory=str(storage_path)), name="assets")

    @app.post("/scripts/from-upload")
    async def convert_upload(
        request: Request,
        image: UploadFile = File(...),
        script_name: str | None = Form(default=None),
        author: str | None = Form(default=None),
    ):
        base_url = str(request.base_url).rstrip("/")
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        result = convert_image_bytes_to_script(
            image_bytes=image_bytes,
            storage_dir=storage_path,
            public_base_url=base_url,
            source_name=image.filename or "upload.png",
            script_name_override=script_name,
            author_override=author,
        )
        return {
            "uuid": result.request_id,
            "json_url": f"{base_url}/scripts/{result.request_id}.json",
            "source_image_url": f"{base_url}/assets/{result.request_id}/original.png",
            "homebrew_images": result.image_urls,
            "script": result.script,
        }

    @app.get("/scripts/{request_id}.json")
    def get_script_json(request_id: str):
        try:
            request_uuid = uuid.UUID(request_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid UUID") from exc

        script_path = (storage_path / str(request_uuid) / "script.json").resolve()
        if storage_path not in script_path.parents:
            raise HTTPException(status_code=400, detail="Invalid path")
        if not script_path.exists():
            raise HTTPException(status_code=404, detail="Script not found")
        with script_path.open("r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))

    @app.get("/health")
    def health():
        return {"ok": True}

    # ------------------------------------------------------------------
    # POST /api/upload — ingest a script image
    # ------------------------------------------------------------------
    @app.post("/api/upload")
    async def upload_script(image: UploadFile = File(...)):
        uid = uuid.uuid4().hex[:8]
        upload_dir = storage_path / uid
        upload_dir.mkdir(parents=True, exist_ok=True)

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        source_filename = image.filename or "upload.png"
        source_path = upload_dir / source_filename
        source_path.write_bytes(image_bytes)

        script_name, rows = process_script_image(str(source_path), str(upload_dir))

        official_roles = get_official_roles()
        official_by_name: dict[str, str] = {
            r.name.strip().lower(): r.id for r in official_roles
        }

        script: list = [{"id": "_meta", "name": script_name}]

        for row in rows:
            search_key = row["raw_name"].strip().lower()
            if search_key in official_by_name:
                script.append({"id": official_by_name[search_key]})
            else:
                raw_name: str = row["raw_name"]
                safe_id = _slugify(raw_name) or f"homebrew-{uid}"
                icon_filename = f"script.{safe_id}.png"
                icon_path = upload_dir / icon_filename
                cv2.imwrite(str(icon_path), row["icon_crop"])
                script.append(
                    {
                        "id": safe_id,
                        "name": raw_name,
                        "ability": row["ability"],
                        "team": "townsfolk",
                        "image": f"/script-assets/{uid}/{icon_filename}",
                    }
                )

        script_json_path = upload_dir / "script.json"
        with script_json_path.open("w", encoding="utf-8") as f:
            json.dump(script, f, indent=2, ensure_ascii=False)

        with sqlite3.connect(_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scripts (uuid, name, custom_data) VALUES (?, ?, ?)",
                (uid, script_name, json.dumps(script, ensure_ascii=False)),
            )
            conn.commit()

        return {"uuid": uid, "script": script}

    # ------------------------------------------------------------------
    # GET /script/{uuid_str}/script.json
    # ------------------------------------------------------------------
    @app.get("/script/{uuid_str}/script.json")
    def get_script_json_file(uuid_str: str):
        if not _SAFE_UUID_RE.match(uuid_str):
            raise HTTPException(status_code=400, detail="Invalid identifier")
        safe_uid = os.path.basename(uuid_str)
        path = (storage_path / safe_uid / "script.json").resolve()
        if storage_path.resolve() not in path.parents:
            raise HTTPException(status_code=400, detail="Invalid path")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Script not found")
        return FileResponse(str(path), media_type="application/json")

    # ------------------------------------------------------------------
    # GET /script/{uuid_str}/scriptlogo.png
    # ------------------------------------------------------------------
    @app.get("/script/{uuid_str}/scriptlogo.png")
    def get_script_logo(uuid_str: str):
        if not _SAFE_UUID_RE.match(uuid_str):
            raise HTTPException(status_code=400, detail="Invalid identifier")
        safe_uid = os.path.basename(uuid_str)
        path = (storage_path / safe_uid / "scriptlogo.png").resolve()
        if storage_path.resolve() not in path.parents:
            raise HTTPException(status_code=400, detail="Invalid path")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Logo not found")
        return FileResponse(str(path), media_type="image/png")

    # ------------------------------------------------------------------
    # GET /script/{uuid_str}/{asset_name}  — homebrew icon assets
    # ------------------------------------------------------------------
    @app.get("/script/{uuid_str}/{asset_name}")
    def get_script_asset(uuid_str: str, asset_name: str):
        if not _SAFE_UUID_RE.match(uuid_str):
            raise HTTPException(status_code=400, detail="Invalid identifier")
        if not _SAFE_ASSET_RE.match(asset_name) or asset_name.startswith("."):
            raise HTTPException(status_code=400, detail="Invalid asset name")
        safe_uid = os.path.basename(uuid_str)
        safe_name = os.path.basename(asset_name)
        path = (storage_path / safe_uid / safe_name).resolve()
        if storage_path.resolve() not in path.parents:
            raise HTTPException(status_code=400, detail="Invalid path")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(str(path), media_type="image/png")

    return app


app = create_app()
