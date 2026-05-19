from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import jsonschema
import pytesseract
import requests
from PIL import Image

from .data import get_official_role_maps, get_script_schema
from .parser import OCRLine, crop_icon_for_role, parse_script_lines, save_icon, slugify_role_id

logger = logging.getLogger(__name__)

_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


@dataclass
class ConversionResult:
    request_id: str
    script: list
    script_path: Path
    image_path: Path
    image_urls: dict[str, str]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _extract_lines_local(image: Image.Image) -> list[OCRLine]:
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


def _extract_json_payload(content: object) -> str:
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    else:
        text = ""
    if not text:
        raise ValueError("Missing model output")
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def _extract_lines_deepseek(image: Image.Image) -> list[OCRLine] | None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    model = os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-chat")
    endpoint = os.getenv("DEEPSEEK_API_URL", _DEEPSEEK_URL)

    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Read this Blood on the Clocktower script image and return only valid JSON."
                            " Output an array. Each entry must have text,x0,y0,x1,y1."
                            " text is the exact OCR text for one full line."
                            " Coordinates are pixel ints within the image."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            }
        ],
    }

    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("No choices returned")
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    text = _extract_json_payload(content)
    raw_lines = json.loads(text)
    if not isinstance(raw_lines, list):
        raise ValueError("Model output must be a JSON list")

    lines: list[OCRLine] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            continue
        text_value = str(raw_line.get("text", "")).strip()
        if not text_value:
            continue
        try:
            x0 = int(raw_line.get("x0"))
            y0 = int(raw_line.get("y0"))
            x1 = int(raw_line.get("x1"))
            y1 = int(raw_line.get("y1"))
        except (TypeError, ValueError):
            continue
        lines.append(OCRLine(text=text_value, x0=x0, y0=y0, x1=x1, y1=y1))

    if not lines:
        raise ValueError("DeepSeek returned no valid OCR lines")
    return sorted(lines, key=lambda l: (l.y0, l.x0))


def _extract_lines(image: Image.Image) -> list[OCRLine]:
    try:
        deepseek_lines = _extract_lines_deepseek(image)
        if deepseek_lines:
            return deepseek_lines
    except Exception as exc:
        msg = str(exc)
        if "400" in msg or "Bad Request" in msg:
            logger.warning(
                "DeepSeek OCR failed (400 Bad Request); falling back to local OCR."
                " The configured model may not support vision inputs."
                " Set DEEPSEEK_OCR_MODEL to a vision-capable model (e.g. deepseek-vl2). Error: %s",
                exc,
            )
        else:
            logger.warning("DeepSeek OCR failed; falling back to local OCR: %s", exc)
    return _extract_lines_local(image)


def convert_image_bytes_to_script(
    image_bytes: bytes,
    storage_dir: Path,
    public_base_url: str,
    source_name: str = "original.png",
    script_name_override: str | None = None,
    author_override: str | None = None,
    request_id: str | None = None,
) -> ConversionResult:
    request_id = request_id or str(uuid.uuid4())
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
