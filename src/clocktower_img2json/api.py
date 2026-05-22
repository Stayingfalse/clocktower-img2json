from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import jsonschema
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from .converter import convert_image_bytes_to_script
from .data import get_official_roles
from .database import DB_PATH, create_script_record, init_db, log_script_edit, script_record_exists
from .startup import refresh_official_roles

_SAFE_UUID_RE = re.compile(r"^[a-zA-Z0-9\-]{1,64}$")
_SAFE_ASSET_RE = re.compile(r"^[a-zA-Z0-9_.\-]{1,128}$")
_LOGO_WIDTH = 600
_LOGO_HEIGHT = 150
_LOGO_BG = "#20252f"
_LOGO_FG = "#f7f7fb"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db(app.state.db_path)
    refresh_official_roles()
    yield



def _slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned[:50]



def _safe_uuid(uuid_str: str) -> str:
    if not _SAFE_UUID_RE.match(uuid_str):
        raise HTTPException(status_code=400, detail="Invalid identifier")
    return os.path.basename(uuid_str)



def _existing_script_dir(storage_path: Path, uuid_str: str) -> Path:
    safe_uid = _safe_uuid(uuid_str)
    if not storage_path.exists():
        raise HTTPException(status_code=404, detail="Script not found")
    for child in storage_path.iterdir():
        if child.is_dir() and child.name == safe_uid:
            return child
    raise HTTPException(status_code=404, detail="Script not found")



def _existing_script_file(script_dir: Path, filename: str, not_found_detail: str) -> Path:
    for child in script_dir.iterdir():
        if child.is_file() and child.name == filename:
            return child
    raise HTTPException(status_code=404, detail=not_found_detail)



def _frontend_path(frontend_root: Path, relative_path: str) -> Path:
    path = (frontend_root / relative_path).resolve()
    if frontend_root.resolve() not in path.parents and path != frontend_root.resolve():
        raise HTTPException(status_code=400, detail="Invalid frontend path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return path



def _resolve_frontend_dir(frontend_dir: str | None) -> Path:
    if frontend_dir is not None:
        return Path(frontend_dir).resolve()

    candidates: list[Path] = []

    env_frontend = os.getenv("CLOCKTOWER_FRONTEND_DIR")
    if env_frontend:
        candidates.append(Path(env_frontend).resolve())

    module_path = Path(__file__).resolve()
    candidates.extend(
        [
            module_path.parents[2] / "frontend",  # source checkout layout
            module_path.parents[1] / "frontend",  # package-included frontend layout
            Path.cwd() / "frontend",  # runtime working directory layout
            Path("/app/frontend"),  # container layout
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]



def _ensure_script_logo(script_name: str, output_dir: Path) -> None:
    logo_path = output_dir / "scriptlogo.png"
    if logo_path.exists():
        return

    img = Image.new("RGB", (_LOGO_WIDTH, _LOGO_HEIGHT), color=_LOGO_BG)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=40)
    except OSError:
        font = ImageFont.load_default()

    text = script_name or "Custom Script"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((_LOGO_WIDTH - text_w) // 2, (_LOGO_HEIGHT - text_h) // 2),
        text,
        fill=_LOGO_FG,
        font=font,
    )
    img.save(logo_path)


def _rewrite_dashboard_script_assets(script: list[object], script_dir: Path, uid: str) -> list[object]:
    rewritten: list[object] = []
    images_dir = script_dir / "images"

    for entry in script:
        if not isinstance(entry, dict):
            rewritten.append(entry)
            continue
        if entry.get("id") == "_meta":
            rewritten.append(entry)
            continue

        role_id = str(entry.get("id", "")).strip()
        if not role_id:
            rewritten.append(entry)
            continue

        icon_filename = f"script.{role_id}.png"
        generated_icon = images_dir / f"{role_id}.png"
        icon_path = script_dir / icon_filename
        if generated_icon.exists() and not icon_path.exists():
            generated_icon.replace(icon_path)

        new_entry = dict(entry)
        new_entry["image"] = f"/script/{uid}/{icon_filename}"
        rewritten.append(new_entry)

    return rewritten



def create_app(
    storage_dir: str = "storage",
    db_path: Path | None = None,
    frontend_dir: str | None = None,
) -> FastAPI:
    app = FastAPI(title="clocktower-img2json", version="0.1.0", lifespan=_lifespan)
    storage_path = Path(storage_dir).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)

    app.state.storage_path = storage_path
    app.state.db_path = db_path if db_path is not None else DB_PATH
    app.state.frontend_path = _resolve_frontend_dir(frontend_dir)

    # Ensure frontend directory exists to prevent startup crashes
    if not app.state.frontend_path.exists():
        print(f"CRITICAL WARNING: Frontend directory not found at {app.state.frontend_path}!")

    app.mount("/assets", StaticFiles(directory=str(storage_path)), name="assets")
    app.mount("/dashboard", StaticFiles(directory=str(app.state.frontend_path), html=True), name="frontend")

    @app.get("/")
    async def root_redirect():
        return RedirectResponse(url="/dashboard/index.html")

    @app.get("/index.html", response_class=HTMLResponse)
    def index_page():
        return FileResponse(_frontend_path(app.state.frontend_path, "index.html"))

    @app.get("/dashboard/edit.html", response_class=HTMLResponse)
    def dashboard_page():
        return FileResponse(_frontend_path(app.state.frontend_path, "edit.html"))

    @app.get("/script/{uuid_str}/", response_class=HTMLResponse)
    def pretty_dashboard_page(uuid_str: str):
        _safe_uuid(uuid_str)
        return FileResponse(_frontend_path(app.state.frontend_path, "edit.html"))

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

        try:
            result = convert_image_bytes_to_script(
                image_bytes=image_bytes,
                storage_dir=storage_path,
                public_base_url=base_url,
                source_name=image.filename or "upload.png",
                script_name_override=script_name,
                author_override=author,
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
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

    @app.get("/api/official-roles")
    def official_roles():
        return [
            {
                "id": role.id,
                "name": role.name,
                "team": role.team,
                "ability": role.ability,
            }
            for role in get_official_roles()
        ]

    @app.post("/api/upload")
    async def upload_script(
        request: Request,
        image: UploadFile = File(...),
        creator: str | None = Form(default=None),
    ):
        uid = uuid.uuid4().hex[:8]
        upload_dir = storage_path / uid
        upload_dir.mkdir(parents=True, exist_ok=True)

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        source_filename = image.filename or "upload.png"
        source_path = upload_dir / source_filename
        source_path.write_bytes(image_bytes)

        try:
            result = convert_image_bytes_to_script(
                image_bytes=image_bytes,
                storage_dir=storage_path,
                public_base_url=str(request.base_url).rstrip("/"),
                source_name=source_filename,
                request_id=uid,
            )
        except jsonschema.ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"OCR produced a script that does not meet the minimum requirements: {exc.message}",
            ) from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
        script = _rewrite_dashboard_script_assets(result.script, upload_dir, uid)
        if not script or not (isinstance(script[0], dict) and script[0].get("id") == "_meta"):
            script.insert(0, {"id": "_meta", "name": "Custom Script"})
        script_name = str(script[0].get("name", "Custom Script")) if isinstance(script[0], dict) else "Custom Script"
        _ensure_script_logo(script_name, upload_dir)

        script_json_path = upload_dir / "script.json"
        with script_json_path.open("w", encoding="utf-8") as f:
            json.dump(script, f, indent=2, ensure_ascii=False)

        create_script_record(uid, creator=creator, db_path=app.state.db_path)
        return {"uuid": uid, "script": script}

    @app.get("/api/script/{uuid_str}")
    def get_script(uuid_str: str):
        script_dir = _existing_script_dir(storage_path, uuid_str)
        script_path = _existing_script_file(script_dir, "script.json", "Script not found")
        with script_path.open("r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))

    @app.post("/api/script/{uuid_str}/update")
    async def update_script(
        uuid_str: str,
        request: Request,
        edited_by: str | None = None,
    ):
        safe_uid = _safe_uuid(uuid_str)
        script_dir = _existing_script_dir(storage_path, safe_uid)
        script_path = _existing_script_file(script_dir, "script.json", "Script not found")
        if not script_record_exists(safe_uid, db_path=app.state.db_path):
            raise HTTPException(status_code=404, detail="Script metadata not found")

        try:
            updated_script = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

        if not isinstance(updated_script, list):
            raise HTTPException(status_code=400, detail="Request body must be a JSON array")

        with script_path.open("w", encoding="utf-8") as f:
            json.dump(updated_script, f, indent=2, ensure_ascii=False)

        log_script_edit(
            safe_uid,
            edited_by=(edited_by or "anonymous").strip() or "anonymous",
            change_summary=f"Updated script with {len(updated_script)} entries",
            db_path=app.state.db_path,
        )
        return {
            "status": "ok",
            "message": "Script updated successfully",
            "uuid": safe_uid,
        }

    @app.get("/script/{uuid_str}/script.json")
    def get_script_json_file(uuid_str: str):
        script_dir = _existing_script_dir(storage_path, uuid_str)
        path = _existing_script_file(script_dir, "script.json", "Script not found")
        return FileResponse(str(path), media_type="application/json")

    @app.get("/script/{uuid_str}/scriptlogo.png")
    def get_script_logo(uuid_str: str):
        script_dir = _existing_script_dir(storage_path, uuid_str)
        path = _existing_script_file(script_dir, "scriptlogo.png", "Logo not found")
        return FileResponse(str(path), media_type="image/png")

    @app.get("/script/{uuid_str}/{asset_name}")
    def get_script_asset(uuid_str: str, asset_name: str):
        if not _SAFE_ASSET_RE.match(asset_name) or asset_name.startswith("."):
            raise HTTPException(status_code=400, detail="Invalid asset name")
        safe_name = os.path.basename(asset_name)
        script_dir = _existing_script_dir(storage_path, uuid_str)
        path = _existing_script_file(script_dir, safe_name, "Asset not found")
        return FileResponse(str(path), media_type="image/png")

    return app


app = create_app()
