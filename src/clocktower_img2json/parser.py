from __future__ import annotations

import re
import uuid
from collections import deque
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
    icon_bbox: tuple[int, int, int, int] | None = None


@dataclass
class ParsedRole:
    name: str
    team: str | None
    ability: str
    bbox: tuple[int, int, int, int] | None
    icon_bbox: tuple[int, int, int, int] | None
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
    normalized = [
        OCRLine(_clean_line(line.text), line.x0, line.y0, line.x1, line.y1, line.icon_bbox)
        for line in lines
    ]
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
        if is_header and current_team is None and official is None:
            i += 1
            continue
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
                    icon_bbox=line.icon_bbox,
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
    remove_background(icon).save(out_path, format="PNG")


def remove_background(icon: Image.Image, threshold: int = 44) -> Image.Image:
    image = icon.convert("RGBA")
    width, height = image.size
    if width < 2 or height < 2:
        return image

    pixels = image.load()
    sample_size = max(1, min(8, width // 4, height // 4))
    corner_pixels: list[tuple[int, int, int]] = []
    boxes = [
        (0, 0, sample_size, sample_size),
        (max(0, width - sample_size), 0, width, sample_size),
        (0, max(0, height - sample_size), sample_size, height),
        (max(0, width - sample_size), max(0, height - sample_size), width, height),
    ]
    for x0, y0, x1, y1 in boxes:
        for x in range(x0, x1):
            for y in range(y0, y1):
                r, g, b, _ = pixels[x, y]
                corner_pixels.append((r, g, b))

    if not corner_pixels:
        return image

    bg_r = sum(r for r, _, _ in corner_pixels) / len(corner_pixels)
    bg_g = sum(g for _, g, _ in corner_pixels) / len(corner_pixels)
    bg_b = sum(b for _, _, b in corner_pixels) / len(corner_pixels)
    threshold_sq = threshold * threshold

    def _near_bg(x: int, y: int) -> bool:
        r, g, b, _ = pixels[x, y]
        dist_sq = (r - bg_r) ** 2 + (g - bg_g) ** 2 + (b - bg_b) ** 2
        return dist_sq <= threshold_sq

    visited = [[False] * width for _ in range(height)]
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if _near_bg(x, 0):
            queue.append((x, 0))
        if _near_bg(x, height - 1):
            queue.append((x, height - 1))
    for y in range(height):
        if _near_bg(0, y):
            queue.append((0, y))
        if _near_bg(width - 1, y):
            queue.append((width - 1, y))

    while queue:
        x, y = queue.popleft()
        if visited[y][x]:
            continue
        visited[y][x] = True
        r, g, b, _ = pixels[x, y]
        pixels[x, y] = (r, g, b, 0)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if visited[ny][nx] or not _near_bg(nx, ny):
                continue
            queue.append((nx, ny))

    return image
