from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import jsonschema
import pytesseract
from PIL import Image

from .data import get_official_role_maps, get_script_schema
from .parser import OCRLine, crop_icon_for_role, parse_script_lines, save_icon, slugify_role_id


@dataclass
class ConversionResult:
    request_id: str
    script: list
    script_path: Path
    image_path: Path
    image_urls: dict[str, str]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _extract_lines(image: Image.Image) -> list[OCRLine]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], OCRLine] = {}

    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf_raw = data.get("conf", ["-1"] * n)[i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1

        if not text or conf < 20:
            continue

        block = int(data["block_num"][i])
        par = int(data["par_num"][i])
        line_num = int(data["line_num"][i])
        key = (block, par, line_num)

        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        x0, y0, x1, y1 = left, top, left + width, top + height

        existing = lines.get(key)
        if existing is None:
            lines[key] = OCRLine(text=text, x0=x0, y0=y0, x1=x1, y1=y1)
        else:
            lines[key] = OCRLine(
                text=f"{existing.text} {text}".strip(),
                x0=min(existing.x0, x0),
                y0=min(existing.y0, y0),
                x1=max(existing.x1, x1),
                y1=max(existing.y1, y1),
            )

    return sorted(lines.values(), key=lambda l: (l.y0, l.x0))


def convert_image_bytes_to_script(
    image_bytes: bytes,
    storage_dir: Path,
    public_base_url: str,
    source_name: str = "original.png",
    script_name_override: str | None = None,
    author_override: str | None = None,
) -> ConversionResult:
    request_id = str(uuid.uuid4())
    output_dir = storage_dir / request_id
    images_dir = output_dir / "images"
    _ensure_dir(images_dir)

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    _ = source_name
    original_path = output_dir / "original.png"
    image.save(original_path, format="PNG")

    by_id, by_name = get_official_role_maps()
    lines = _extract_lines(image)
    script_name, author, roles = parse_script_lines(lines, by_name)

    if script_name_override:
        script_name = script_name_override
    if author_override:
        author = author_override

    script: list = []
    if script_name:
        meta = {"id": "_meta", "name": script_name}
        if author:
            meta["author"] = author
        script.append(meta)

    image_urls: dict[str, str] = {}

    for role in roles:
        if role.official:
            script.append(role.official.id)
            continue

        role_id = slugify_role_id(role.name)
        team = role.team or "townsfolk"
        role_obj = {
            "id": role_id,
            "name": role.name,
            "team": team,
            "ability": role.ability or "",
        }

        if role.bbox:
            icon = crop_icon_for_role(image, role.bbox)
            icon_path = images_dir / f"{role_id}.png"
            save_icon(icon, icon_path)
            icon_url = f"{public_base_url.rstrip('/')}/assets/{request_id}/images/{role_id}.png"
            role_obj["image"] = icon_url
            image_urls[role_id] = icon_url

        script.append(role_obj)

    schema = get_script_schema()
    jsonschema.validate(script, schema)

    script_path = output_dir / "script.json"
    with script_path.open("w", encoding="utf-8") as f:
        json.dump(script, f, indent=2, ensure_ascii=False)

    return ConversionResult(
        request_id=request_id,
        script=script,
        script_path=script_path,
        image_path=original_path,
        image_urls=image_urls,
    )
