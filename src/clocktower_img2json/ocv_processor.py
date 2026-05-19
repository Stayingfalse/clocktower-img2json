from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

_DEFAULT_SCRIPT_NAME = "Custom Script"
_LOGO_WIDTH = 600
_LOGO_HEIGHT = 150
_LOGO_BG = (64, 64, 64)      # dark gray (RGB)
_LOGO_FG = (255, 255, 255)   # white
_MIN_CONTOUR_HEIGHT = 30
_MIN_CONTOUR_WIDTH = 100


def _make_placeholder_logo(script_name: str, storage_dir: Path) -> None:
    """Create a 600×150 dark-gray placeholder image with white script name text."""
    img = Image.new("RGB", (_LOGO_WIDTH, _LOGO_HEIGHT), color=_LOGO_BG)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=40)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), script_name, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (_LOGO_WIDTH - text_w) // 2
    y = (_LOGO_HEIGHT - text_h) // 2
    draw.text((x, y), script_name, fill=_LOGO_FG, font=font)

    logo_path = storage_dir / "scriptlogo.png"
    img.save(logo_path)


def process_script_image(
    image_path: str,
    storage_dir: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Process a Blood on the Clocktower script image with OpenCV and PyTesseract.

    Parameters
    ----------
    image_path:
        Path to the input script image (any format supported by OpenCV).
    storage_dir:
        Directory in which derived files (e.g. scriptlogo.png) are written.

    Returns
    -------
    tuple[str, list[dict]]
        A 2-tuple of ``(script_name, rows)`` where each row dict contains:
        ``'raw_name'``, ``'ability'``, and ``'icon_crop'`` (a NumPy BGR array).
    """
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load image and extract script name from top 15 %
    # ------------------------------------------------------------------
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Could not read image at: {image_path}")

    height, width = bgr.shape[:2]
    top_cutoff = max(1, int(height * 0.15))

    top_section = bgr[:top_cutoff, :]
    top_rgb = cv2.cvtColor(top_section, cv2.COLOR_BGR2RGB)
    top_pil = Image.fromarray(top_rgb)
    top_text = pytesseract.image_to_string(top_pil).strip()

    first_line = ""
    for raw_line in top_text.splitlines():
        stripped = raw_line.strip()
        if stripped:
            first_line = stripped
            break

    if len(first_line) >= 3:
        script_name = first_line
    else:
        script_name = _DEFAULT_SCRIPT_NAME
        _make_placeholder_logo(script_name, storage_path)

    # ------------------------------------------------------------------
    # 2. Process the body: grayscale → inverted binary threshold → contours
    # ------------------------------------------------------------------
    body = bgr[top_cutoff:, :]
    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ------------------------------------------------------------------
    # 3. Sort contours top-to-bottom and filter by minimum size
    # ------------------------------------------------------------------
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    rows: list[dict[str, Any]] = []

    for contour in sorted_contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if ch <= _MIN_CONTOUR_HEIGHT or cw <= _MIN_CONTOUR_WIDTH:
            continue

        # ------------------------------------------------------------------
        # 4. OCR the segment; first line → name, rest → ability
        # ------------------------------------------------------------------
        segment = body[cy : cy + ch, cx : cx + cw]
        seg_pil = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        ocr_text = pytesseract.image_to_string(seg_pil).strip()

        lines = [ln.strip() for ln in ocr_text.replace("\r\n", "\n").split("\n") if ln.strip()]
        raw_name = lines[0] if lines else ""
        ability = " ".join(lines[1:]) if len(lines) > 1 else ""

        # ------------------------------------------------------------------
        # 5. Crop a square thumbnail from the leftmost edge of the row segment
        # ------------------------------------------------------------------
        square_size = ch  # use row height as the side length of the square
        icon_crop = body[cy : cy + square_size, cx : cx + square_size]

        rows.append(
            {
                "raw_name": raw_name,
                "ability": ability,
                "icon_crop": icon_crop,
            }
        )

    return script_name, rows
