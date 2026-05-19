"""Tests for ocv_processor.process_script_image."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from unittest.mock import patch

from clocktower_img2json.ocv_processor import (
    _DEFAULT_SCRIPT_NAME,
    _MIN_CONTOUR_HEIGHT,
    _MIN_CONTOUR_WIDTH,
    _make_placeholder_logo,
    process_script_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_test_image(path: Path, width: int = 400, height: int = 600) -> np.ndarray:
    """Create a simple synthetic script image and save it."""
    img = np.full((height, width, 3), 200, dtype=np.uint8)  # light gray background

    # Draw a dark "role row" block in the body section (below top 15 %)
    top_cutoff = int(height * 0.15)
    row_y = top_cutoff + 40
    cv2.rectangle(img, (10, row_y), (width - 10, row_y + 60), (50, 50, 50), -1)

    cv2.imwrite(str(path), img)
    return img


# ---------------------------------------------------------------------------
# _make_placeholder_logo
# ---------------------------------------------------------------------------

def test_make_placeholder_logo_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        storage = Path(tmp)
        _make_placeholder_logo("Test Script", storage)
        logo = storage / "scriptlogo.png"
        assert logo.exists()
        # Check dimensions
        import cv2 as _cv2
        img = _cv2.imread(str(logo))
        assert img is not None
        h, w = img.shape[:2]
        assert w == 600
        assert h == 150


# ---------------------------------------------------------------------------
# process_script_image — invalid path
# ---------------------------------------------------------------------------

def test_process_script_image_invalid_path():
    with pytest.raises(ValueError, match="Could not read image"):
        process_script_image("/nonexistent/image.png", "/tmp")


# ---------------------------------------------------------------------------
# process_script_image — short/empty OCR name → default + placeholder
# ---------------------------------------------------------------------------

def test_process_script_image_short_name_produces_placeholder():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "script.png"
        _write_test_image(img_path)

        # Force pytesseract to return an empty string (no real Tesseract in CI)
        with patch("clocktower_img2json.ocv_processor.pytesseract.image_to_string", return_value=""):
            script_name, rows = process_script_image(str(img_path), tmp)

        assert script_name == _DEFAULT_SCRIPT_NAME
        assert (Path(tmp) / "scriptlogo.png").exists()


# ---------------------------------------------------------------------------
# process_script_image — OCR returns a usable name
# ---------------------------------------------------------------------------

def test_process_script_image_uses_first_ocr_line_as_name():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "script.png"
        _write_test_image(img_path)

        ocr_responses = iter(["My Script\nSome extra line", "Role Name\nAbility text"])

        with patch(
            "clocktower_img2json.ocv_processor.pytesseract.image_to_string",
            side_effect=lambda *_a, **_kw: next(ocr_responses),
        ):
            script_name, _ = process_script_image(str(img_path), tmp)

        assert script_name == "My Script"
        assert not (Path(tmp) / "scriptlogo.png").exists()


# ---------------------------------------------------------------------------
# process_script_image — return structure
# ---------------------------------------------------------------------------

def test_process_script_image_return_type():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "script.png"
        _write_test_image(img_path, width=500, height=600)

        # First call = top-section OCR (returns a valid name), subsequent calls = row OCR
        call_count = {"n": 0}

        def mock_ocr(img, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "Awesome Script"
            return "The Imp\nEach night*, choose a player: they die."

        with patch("clocktower_img2json.ocv_processor.pytesseract.image_to_string", side_effect=mock_ocr):
            script_name, rows = process_script_image(str(img_path), tmp)

        assert isinstance(script_name, str)
        assert isinstance(rows, list)
        for row in rows:
            assert "raw_name" in row
            assert "ability" in row
            assert "icon_crop" in row
            assert isinstance(row["icon_crop"], np.ndarray)


# ---------------------------------------------------------------------------
# process_script_image — rows with OCR text are parsed correctly
# ---------------------------------------------------------------------------

def test_process_script_image_row_parsing():
    """Row OCR: first line → raw_name, rest → ability."""
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "script.png"

        # Build an image with a high-contrast row that OpenCV can detect
        height, width = 400, 600
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        top_cutoff = int(height * 0.15)
        row_y = top_cutoff + 20
        # Big dark block that exceeds the size thresholds
        cv2.rectangle(img, (5, row_y), (550, row_y + 80), (0, 0, 0), -1)
        cv2.imwrite(str(img_path), img)

        responses = iter([
            "Script Title",                          # top-section OCR
            "Washerwoman\nYou start knowing a Townsfolk.",  # row OCR
        ])

        with patch(
            "clocktower_img2json.ocv_processor.pytesseract.image_to_string",
            side_effect=lambda *_a, **_kw: next(responses),
        ):
            script_name, rows = process_script_image(str(img_path), tmp)

        assert script_name == "Script Title"
        assert len(rows) >= 1
        assert rows[0]["raw_name"] == "Washerwoman"
        assert rows[0]["ability"] == "You start knowing a Townsfolk."
