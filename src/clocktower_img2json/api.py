from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from .converter import convert_image_bytes_to_script, convert_image_to_script


class ConvertRequest(BaseModel):
    image_url: HttpUrl
    script_name: str | None = None
    author: str | None = None


def create_app(storage_dir: str = "storage") -> FastAPI:
    app = FastAPI(title="clocktower-img2json", version="0.1.0")
    storage_path = Path(storage_dir).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)

    app.mount("/assets", StaticFiles(directory=str(storage_path)), name="assets")

    @app.post("/scripts/from-image")
    def convert(req: ConvertRequest, request: Request):
        base_url = str(request.base_url).rstrip("/")
        result = convert_image_to_script(
            image_url=str(req.image_url),
            storage_dir=storage_path,
            public_base_url=base_url,
            script_name_override=req.script_name,
            author_override=req.author,
        )
        return {
            "uuid": result.request_id,
            "json_url": f"{base_url}/scripts/{result.request_id}.json",
            "source_image_url": f"{base_url}/assets/{result.request_id}/original.png",
            "homebrew_images": result.image_urls,
            "script": result.script,
        }

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
        script_path = storage_path / request_id / "script.json"
        if not script_path.exists():
            raise HTTPException(status_code=404, detail="Script not found")
        with script_path.open("r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


app = create_app()
