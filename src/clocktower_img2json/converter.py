from __future__ import annotations

import json
import logging
import os
import uuid
from base64 import b64encode
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import jsonschema
import pytesseract
import requests
from PIL import Image

from .data import get_official_role_maps, get_script_schema, normalize_name
from .parser import OCRLine, ParsedRole, crop_icon_for_role, parse_script_lines, save_icon, slugify_role_id

logger = logging.getLogger(__name__)

_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_PROMPT = (
    "You are processing a Blood on the Clocktower custom script image."
    " Return ONLY a valid JSON object with this shape:"
    ' {"script_name": string|null, "author": string|null, "roles": [...]}'
    ' where each role is {"name": string, "team": string|null, "ability": string|null,'
    ' "x": int, "y": int, "width": int, "height": int,'
    ' "icon_x": int|null, "icon_y": int|null, "icon_width": int|null, "icon_height": int|null}.'
    " Extract every visible role across all columns."
    " Do NOT return a finished script array and do NOT infer missing roles."
    " Team should be one of townsfolk/outsider/minion/demon/traveller when visible, otherwise null."
    " Ability should contain only this role's ability text; do not merge adjacent roles."
    " Use image pixel coordinates for role text and icon/token when visible."
    " Return only the JSON object, no explanation, no markdown fences."
)


@dataclass
class ConversionResult:
    request_id: str
    script: list
    script_path: Path
    image_path: Path
    image_urls: dict[str, str]


@dataclass
class GeminiRoleObservation:
    name: str
    team: str | None
    ability: str | None
    bbox: tuple[int, int, int, int] | None
    icon_bbox: tuple[int, int, int, int] | None


@dataclass
class GeminiObservation:
    script_name: str | None
    author: str | None
    roles: list[GeminiRoleObservation]
    lines: list[OCRLine]


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


def _extract_lines(image: Image.Image) -> list[OCRLine]:
    """Extract OCR lines using local pytesseract."""
    return _extract_lines_local(image)


def _extract_json_payload(content: object) -> str:
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
            and ((part.get("type") == "text") or ("text" in part and "inline_data" not in part))
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


def _extract_bbox(payload: dict[str, object], prefix: str = "") -> tuple[int, int, int, int] | None:
    try:
        x = int(payload[f"{prefix}x"])
        y = int(payload[f"{prefix}y"])
        width = int(payload[f"{prefix}width"])
        height = int(payload[f"{prefix}height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return (x, y, x + width, y + height)


def _parse_gemini_observations(payload: object) -> GeminiObservation:
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object")

    roles: list[GeminiRoleObservation] = []
    raw_roles = payload.get("roles")
    if isinstance(raw_roles, list):
        for raw_role in raw_roles:
            if not isinstance(raw_role, dict):
                continue
            name = str(raw_role.get("name", "")).strip()
            if not name:
                continue
            bbox = _extract_bbox(raw_role)
            icon_bbox = _extract_bbox(raw_role, prefix="icon_")
            roles.append(
                GeminiRoleObservation(
                    name=name,
                    team=str(raw_role.get("team")).strip() if raw_role.get("team") else None,
                    ability=str(raw_role.get("ability")).strip() if raw_role.get("ability") else None,
                    bbox=bbox,
                    icon_bbox=icon_bbox,
                )
            )

    lines: list[OCRLine] = []
    raw_lines = payload.get("lines")
    if isinstance(raw_lines, list):
        for raw_line in raw_lines:
            if not isinstance(raw_line, dict):
                continue
            text = str(raw_line.get("text", "")).strip()
            if not text:
                continue
            bbox = _extract_bbox(raw_line)
            if bbox is None:
                continue
            icon_bbox = _extract_bbox(raw_line, prefix="icon_")
            lines.append(
                OCRLine(
                    text=text,
                    x0=bbox[0],
                    y0=bbox[1],
                    x1=bbox[2],
                    y1=bbox[3],
                    icon_bbox=icon_bbox,
                )
            )

    if not roles and not lines:
        raise ValueError("Model output must include a roles array or lines array")

    lines.sort(key=lambda line: (line.y0, line.x0))
    return GeminiObservation(
        script_name=str(payload.get("script_name")).strip() if payload.get("script_name") else None,
        author=str(payload.get("author")).strip() if payload.get("author") else None,
        roles=roles,
        lines=lines,
    )


def _extract_embedded_json(image_bytes: bytes) -> list | None:
    """Try to extract a script JSON array embedded in PNG metadata text chunks.

    Blood on the Clocktower script tools (e.g. clocktower.online) often embed
    the script JSON directly into the exported PNG as a tEXt/iTXt chunk.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        candidates: dict[str, str] = {}
        info = getattr(image, "info", {}) or {}
        candidates.update({k: v for k, v in info.items() if isinstance(v, str)})
        text_attr = getattr(image, "text", {}) or {}
        candidates.update({k: v for k, v in text_attr.items() if isinstance(v, str)})

        for value in candidates.values():
            value = value.strip()
            if not (value.startswith("[") or value.startswith("{")):
                continue
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list) and len(parsed) > 0:
                    logger.debug("Found embedded JSON in PNG metadata (%d entries)", len(parsed))
                    return parsed
            except json.JSONDecodeError:
                continue
    except Exception as exc:
        logger.debug("Could not read embedded JSON from image: %s", exc)
    return None


def _redact_gemini_payload(payload: dict) -> dict:
    redacted = json.loads(json.dumps(payload))
    for content in redacted.get("contents", []):
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                continue
            inline_data = part.get("inline_data")
            if isinstance(inline_data, dict) and "data" in inline_data:
                inline_data["data"] = "<redacted>"
    return redacted


def _redact_request_body_for_log(request_body: str | None, fallback_payload: dict) -> str:
    if request_body:
        try:
            parsed = json.loads(request_body)
            if isinstance(parsed, dict):
                return json.dumps(_redact_gemini_payload(parsed), ensure_ascii=False)
        except Exception:
            pass
    return json.dumps(_redact_gemini_payload(fallback_payload), ensure_ascii=False)


def _extract_gemini_observations(image_bytes: bytes, embedded_hint: list | None = None) -> GeminiObservation | None:
    """Call Gemini for observed text/icon boxes and convert the response into OCR lines."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    endpoint = os.getenv("GEMINI_API_URL", _GEMINI_API_URL).rstrip("/")

    prompt = _GEMINI_PROMPT
    if embedded_hint:
        try:
            hint_json = json.dumps(embedded_hint, ensure_ascii=False)
            prompt = (
                f"The image metadata contains this embedded JSON hint: {hint_json}"
                f"\nUse it to help verify your extraction, but treat the image as the primary source."
                f"\n\n{prompt}"
            )
        except Exception:
            pass

    image_b64 = b64encode(image_bytes).decode("ascii")
    payload = {
        "generationConfig": {"temperature": 0},
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    },
                ],
            }
        ],
    }
    url = f"{endpoint}/{model}:generateContent?key={api_key}"

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise ValueError("No candidates returned")
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else {}
        parts = content.get("parts") if isinstance(content, dict) else None
        text = _extract_json_payload(parts)
        parsed = json.loads(text)
        return _parse_gemini_observations(parsed)
    except requests.HTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        request_body = None
        response_body = None
        if exc.response is not None:
            request = getattr(exc.response, "request", None)
            if request is not None:
                request_body_raw = getattr(request, "body", None)
                if isinstance(request_body_raw, bytes):
                    request_body = request_body_raw.decode("utf-8", errors="replace")
                elif request_body_raw is not None:
                    request_body = str(request_body_raw)
            response_body = getattr(exc.response, "text", None)
        request_body = _redact_request_body_for_log(request_body, payload)
        if response_body is None:
            response_body = str(exc)

        logger.warning(
            "Gemini observation extraction failed with HTTP %s; falling back to local OCR."
            " Request body sent: %s"
            " Response body received: %s",
            status_code,
            request_body,
            response_body,
        )
        return None
    except Exception as exc:
        logger.warning("Gemini observation extraction failed; falling back to local OCR: %s", exc)
        return None


def _extract_script_gemini(image_bytes: bytes, embedded_hint: list | None = None) -> GeminiObservation | None:
    return _extract_gemini_observations(image_bytes, embedded_hint=embedded_hint)


_TEAM_ALIASES = {
    "townsfolk": "townsfolk",
    "outsider": "outsider",
    "outsiders": "outsider",
    "minion": "minion",
    "minions": "minion",
    "demon": "demon",
    "demons": "demon",
    "traveller": "traveller",
    "travellers": "traveller",
}


def _normalize_team_label(value: str | None) -> str | None:
    if not value:
        return None
    return _TEAM_ALIASES.get(normalize_name(value))


def _build_roles_from_gemini_observations(
    observation: GeminiObservation,
    official_by_name: dict,
) -> tuple[str | None, str | None, list[ParsedRole], str]:
    if observation.roles:
        deduped: list[ParsedRole] = []
        seen: set[str] = set()
        for role in observation.roles:
            key = normalize_name(role.name)
            if not key or key in seen:
                continue
            seen.add(key)
            official = official_by_name.get(key)
            deduped.append(
                ParsedRole(
                    name=role.name,
                    team=official.team if official else _normalize_team_label(role.team),
                    ability=(role.ability or "").strip(),
                    bbox=role.bbox,
                    icon_bbox=role.icon_bbox,
                    official=official,
                )
            )
        return observation.script_name, observation.author, deduped, "roles"

    script_name, author, roles = parse_script_lines(observation.lines, official_by_name)
    if observation.script_name:
        script_name = observation.script_name
    if observation.author:
        author = observation.author
    return script_name, author, roles, "lines"


def _role_names_preview(roles: list[ParsedRole], limit: int = 8) -> str:
    if not roles:
        return "none"
    names = [role.name for role in roles[:limit]]
    suffix = ", ..." if len(roles) > limit else ""
    return ", ".join(names) + suffix


def _apply_meta_overrides(script: list, name_override: str | None, author_override: str | None) -> None:
    """Apply script name / author overrides to the _meta entry, inserting one if absent."""
    if not (name_override or author_override):
        return
    if script and isinstance(script[0], dict) and script[0].get("id") == "_meta":
        if name_override:
            script[0]["name"] = name_override
        if author_override:
            script[0]["author"] = author_override
    elif name_override:
        meta: dict = {"id": "_meta", "name": name_override}
        if author_override:
            meta["author"] = author_override
        script.insert(0, meta)


def _crop_icon(image: Image.Image, role_bbox: tuple[int, int, int, int], icon_bbox: tuple[int, int, int, int] | None) -> Image.Image:
    if icon_bbox is None:
        return crop_icon_for_role(image, role_bbox)

    x0, y0, x1, y1 = icon_bbox
    if x1 <= x0 or y1 <= y0:
        return crop_icon_for_role(image, role_bbox)

    role_height = max(32, role_bbox[3] - role_bbox[1])
    icon_size = max(x1 - x0, y1 - y0)
    target_size = max(icon_size + 28, int(role_height * 2.5), 72)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    crop_x0 = int(round(cx - (target_size / 2)))
    crop_y0 = int(round(cy - (target_size / 2)))
    crop_x1 = crop_x0 + target_size
    crop_y1 = crop_y0 + target_size

    if crop_x0 < 0:
        crop_x1 -= crop_x0
        crop_x0 = 0
    if crop_y0 < 0:
        crop_y1 -= crop_y0
        crop_y0 = 0
    if crop_x1 > image.width:
        delta = crop_x1 - image.width
        crop_x0 = max(0, crop_x0 - delta)
        crop_x1 = image.width
    if crop_y1 > image.height:
        delta = crop_y1 - image.height
        crop_y0 = max(0, crop_y0 - delta)
        crop_y1 = image.height

    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return crop_icon_for_role(image, role_bbox)

    return image.crop((crop_x0, crop_y0, crop_x1, crop_y1))


def _build_script_from_roles(
    *,
    image: Image.Image,
    request_id: str,
    public_base_url: str,
    images_dir: Path,
    script_name: str | None,
    author: str | None,
    roles: list,
) -> tuple[list, dict[str, str]]:
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
            icon = _crop_icon(image, role.bbox, role.icon_bbox)
            icon_path = images_dir / f"{role_id}.png"
            save_icon(icon, icon_path)
            icon_url = f"{public_base_url.rstrip('/')}/assets/{request_id}/images/{role_id}.png"
            role_obj["image"] = icon_url
            image_urls[role_id] = icon_url

        script.append(role_obj)

    return script, image_urls


def _script_role_count(script: list) -> int:
    count = 0
    for entry in script:
        if isinstance(entry, dict) and entry.get("id") == "_meta":
            continue
        count += 1
    return count


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
    normalized_image_buffer = BytesIO()
    image.save(normalized_image_buffer, format="PNG")
    normalized_image_bytes = normalized_image_buffer.getvalue()
    original_path.write_bytes(normalized_image_bytes)

    _, by_name = get_official_role_maps()
    schema = get_script_schema()

    # --- 1. Check for embedded JSON in the original image bytes ---
    embedded_json = _extract_embedded_json(image_bytes)
    validated_embedded_json: list | None = None
    if embedded_json:
        logger.info("Embedded JSON found in image metadata (%d entries)", len(embedded_json))
        try:
            jsonschema.validate(embedded_json, schema)
            validated_embedded_json = embedded_json
        except jsonschema.ValidationError:
            logger.info("Embedded JSON failed schema validation; cannot use as direct fallback")

    # --- 2. Try Google AI Studio Gemini with the normalized image bytes ---
    gemini_observations = _extract_gemini_observations(normalized_image_bytes, embedded_hint=embedded_json)
    if gemini_observations is not None and (gemini_observations.roles or gemini_observations.lines):
        logger.info(
            "Gemini extracted %d role observations and %d line observations",
            len(gemini_observations.roles),
            len(gemini_observations.lines),
        )
        script_name, author, roles, gemini_source = _build_roles_from_gemini_observations(
            gemini_observations,
            by_name,
        )
        logger.info(
            "Gemini heuristic role build (%s) produced %d roles: %s",
            gemini_source,
            len(roles),
            _role_names_preview(roles),
        )
        if script_name_override:
            script_name = script_name_override
        if author_override:
            author = author_override

        script, image_urls = _build_script_from_roles(
            image=image,
            request_id=request_id,
            public_base_url=public_base_url,
            images_dir=images_dir,
            script_name=script_name,
            author=author,
            roles=roles,
        )
        try:
            jsonschema.validate(script, schema)
            if validated_embedded_json is not None:
                gemini_role_count = _script_role_count(script)
                embedded_role_count = _script_role_count(validated_embedded_json)
                if embedded_role_count >= 4 and gemini_role_count < max(3, embedded_role_count // 2):
                    logger.warning(
                        "Gemini script looked sparse (%d roles vs %d embedded; source=%s; roles=%s); using embedded JSON fallback.",
                        gemini_role_count,
                        embedded_role_count,
                        gemini_source,
                        _role_names_preview(roles),
                    )
                else:
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
            else:
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
        except jsonschema.ValidationError as exc:
            logger.warning(
                "Gemini-built script failed schema validation (%d roles from %s: %s); falling back to embedded/local OCR: %s",
                len(roles),
                gemini_source,
                _role_names_preview(roles),
                exc.message,
            )

    # --- 3. Use embedded JSON directly if it passes schema validation ---
    if validated_embedded_json:
        logger.info("Using embedded JSON as script directly")
        _apply_meta_overrides(validated_embedded_json, script_name_override, author_override)
        script_path = output_dir / "script.json"
        with script_path.open("w", encoding="utf-8") as f:
            json.dump(validated_embedded_json, f, indent=2, ensure_ascii=False)
        return ConversionResult(
            request_id=request_id,
            script=validated_embedded_json,
            script_path=script_path,
            image_path=original_path,
            image_urls={},
        )

    # --- 4. Fall back to local pytesseract OCR ---
    lines = _extract_lines(image)
    script_name, author, roles = parse_script_lines(lines, by_name)

    if script_name_override:
        script_name = script_name_override
    if author_override:
        author = author_override

    script, image_urls = _build_script_from_roles(
        image=image,
        request_id=request_id,
        public_base_url=public_base_url,
        images_dir=images_dir,
        script_name=script_name,
        author=author,
        roles=roles,
    )

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
