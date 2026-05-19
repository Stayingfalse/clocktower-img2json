from __future__ import annotations

from unittest.mock import Mock, patch

from PIL import Image

from clocktower_img2json.converter import _extract_lines


def _sample_tesseract_payload() -> dict:
    return {
        "text": ["Washerwoman"],
        "conf": ["85"],
        "block_num": [1],
        "par_num": [1],
        "line_num": [1],
        "left": [10],
        "top": [20],
        "width": [120],
        "height": [18],
    }


def test_extract_lines_uses_local_ocr_when_deepseek_key_missing():
    image = Image.new("RGB", (300, 200), color="white")
    with patch.dict("os.environ", {}, clear=True), patch(
        "clocktower_img2json.converter.pytesseract.image_to_data",
        return_value=_sample_tesseract_payload(),
    ):
        lines = _extract_lines(image)
    assert len(lines) == 1
    assert lines[0].text == "Washerwoman"


def test_extract_lines_uses_deepseek_when_available():
    image = Image.new("RGB", (300, 200), color="white")
    deepseek_body = {
        "choices": [
            {
                "message": {
                    "content": '[{"text":"Pixie","x0":12,"y0":34,"x1":98,"y1":54}]',
                }
            }
        ]
    }
    mock_response = Mock()
    mock_response.json.return_value = deepseek_body
    mock_response.raise_for_status.return_value = None

    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ), patch(
        "clocktower_img2json.converter.pytesseract.image_to_data",
        side_effect=AssertionError("local OCR should not be used"),
    ):
        lines = _extract_lines(image)

    assert len(lines) == 1
    assert lines[0].text == "Pixie"


def test_extract_lines_falls_back_to_local_when_deepseek_fails():
    image = Image.new("RGB", (300, 200), color="white")
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post",
        side_effect=RuntimeError("deepseek unavailable"),
    ), patch(
        "clocktower_img2json.converter.pytesseract.image_to_data",
        return_value=_sample_tesseract_payload(),
    ):
        lines = _extract_lines(image)

    assert len(lines) == 1
    assert lines[0].text == "Washerwoman"


def test_extract_lines_logs_vision_hint_on_400(caplog):
    import logging
    import requests as req_module

    image = Image.new("RGB", (300, 200), color="white")
    http_error = req_module.exceptions.HTTPError("400 Client Error: Bad Request for url: https://api.deepseek.com/chat/completions")

    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post",
        side_effect=http_error,
    ), patch(
        "clocktower_img2json.converter.pytesseract.image_to_data",
        return_value=_sample_tesseract_payload(),
    ), caplog.at_level(logging.WARNING, logger="clocktower_img2json.converter"):
        lines = _extract_lines(image)

    assert len(lines) == 1
    assert lines[0].text == "Washerwoman"
    assert any("vision" in record.message.lower() for record in caplog.records)
