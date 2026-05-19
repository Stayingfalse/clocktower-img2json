from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .data import OfficialRole, normalize_name

TEAM_HEADERS = {
    "townsfolk": "townsfolk",
    "outsiders": "outsider",
    "minions": "minion",
    "demons": "demon",
    "travellers": "traveller",
}


@dataclass
class OCRLine:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class ParsedRole:
    name: str
    team: str | None
    ability: str
    bbox: tuple[int, int, int, int] | None
    official: OfficialRole | None


def slugify_role_id(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in name)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if cleaned:
        return cleaned[:50]
    return f"homebrew-{uuid.uuid4().hex[:12]}"


def _clean_line(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_role_name(text: str) -> bool:
    if not text or len(text) > 50:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    if not words or len(words) > 5:
        return False
    if len(words) == 1 and words[0].isupper():
        return False
    starts_upper = sum(1 for w in words if w[0].isupper())
    return starts_upper >= max(1, len(words) - 1)


def parse_script_lines(lines: list[OCRLine], official_by_name: dict[str, OfficialRole]) -> tuple[str | None, str | None, list[ParsedRole]]:
    normalized = [OCRLine(_clean_line(line.text), line.x0, line.y0, line.x1, line.y1) for line in lines]
    normalized = [line for line in normalized if line.text]

    script_name: str | None = None
    author: str | None = None
    if normalized:
        top = normalized[:8]
        for line in top:
            lowered = line.text.lower()
            if " by " in lowered and not script_name:
                parts = re.split(r"\bby\b", line.text, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    script_name = parts[0].strip(" -") or None
                    author = parts[1].strip(" -") or None
                    break
        if not script_name:
            script_name = top[0].text

    roles: list[ParsedRole] = []
    current_team: str | None = None
    i = 0
    while i < len(normalized):
        line = normalized[i]
        team_key = normalize_name(line.text)
        if team_key in TEAM_HEADERS:
            current_team = TEAM_HEADERS[team_key]
            i += 1
            continue

        is_header = _looks_like_role_name(line.text)
        official = official_by_name.get(normalize_name(line.text)) if is_header else None
        if is_header:
            ability_parts: list[str] = []
            j = i + 1
            while j < len(normalized):
                nxt = normalized[j]
                nxt_team_key = normalize_name(nxt.text)
                if nxt_team_key in TEAM_HEADERS:
                    break
                if _looks_like_role_name(nxt.text) and not nxt.text.lower().startswith(("you ", "each ", "if ", "when ", "once ", "all ")):
                    break
                ability_parts.append(nxt.text)
                j += 1

            ability_text = " ".join(ability_parts).strip()
            role_team = official.team if official else current_team
            roles.append(
                ParsedRole(
                    name=line.text,
                    team=role_team,
                    ability=ability_text,
                    bbox=(line.x0, line.y0, line.x1, line.y1),
                    official=official,
                )
            )
            i = j
            continue

        i += 1

    deduped: list[ParsedRole] = []
    seen = set()
    for role in roles:
        key = normalize_name(role.name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(role)

    return script_name, author, deduped


def crop_icon_for_role(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    text_height = max(32, y1 - y0)
    icon_size = max(80, int(text_height * 2.2))
    pad = 12

    crop_x1 = max(1, x0 - pad)
    crop_x0 = max(0, crop_x1 - icon_size)
    center_y = (y0 + y1) // 2
    crop_y0 = max(0, center_y - icon_size // 2)
    crop_y1 = min(image.height, crop_y0 + icon_size)

    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return image.crop((max(0, x0 - 100), max(0, y0 - 30), min(image.width, x0), min(image.height, y1 + 60)))
    return image.crop((crop_x0, crop_y0, crop_x1, crop_y1))


def save_icon(icon: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    icon.convert("RGBA").save(out_path, format="PNG")
