from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .converter import convert_image_bytes_to_script


def create_app(storage_dir: str = "storage") -> FastAPI:
    app = FastAPI(title="clocktower-img2json", version="0.1.0")
    storage_path = Path(storage_dir).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)

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

    return app


app = create_app()
