from __future__ import annotations

import io
import json
import logging
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from clocktower_img2json.converter import (
    _extract_embedded_json,
    _extract_lines,
    _extract_script_deepseek,
)


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


def _png_bytes_with_json(script: list) -> bytes:
    """Create a PNG with embedded JSON script in a tEXt metadata chunk."""
    img = Image.new("RGB", (50, 50), color="white")
    meta = PngInfo()
    meta.add_text("JSON", json.dumps(script))
    buf = BytesIO()
    img.save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()


# --- _extract_lines (always local OCR) ---

def test_extract_lines_uses_local_ocr_when_deepseek_key_missing():
    image = Image.new("RGB", (300, 200), color="white")
    with patch.dict("os.environ", {}, clear=True), patch(
        "clocktower_img2json.converter.pytesseract.image_to_data",
        return_value=_sample_tesseract_payload(),
    ):
        lines = _extract_lines(image)
    assert len(lines) == 1
    assert lines[0].text == "Washerwoman"


def test_extract_lines_always_uses_local_ocr_even_with_deepseek_key():
    """_extract_lines is now local-only; DeepSeek operates at a higher level."""
    image = Image.new("RGB", (300, 200), color="white")
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.pytesseract.image_to_data",
        return_value=_sample_tesseract_payload(),
    ):
        lines = _extract_lines(image)
    assert len(lines) == 1
    assert lines[0].text == "Washerwoman"


# --- _extract_script_deepseek ---

def test_extract_script_deepseek_returns_none_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        result = _extract_script_deepseek("https://example.com/assets/abc/original.png")
    assert result is None


def test_extract_script_deepseek_returns_script_list():
    deepseek_body = {
        "choices": [
            {
                "message": {
                    "content": '[{"id": "_meta", "name": "Test Script"}, "washerwoman"]',
                }
            }
        ]
    }
    mock_response = Mock()
    mock_response.json.return_value = deepseek_body
    mock_response.raise_for_status.return_value = None

    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ):
        result = _extract_script_deepseek("https://example.com/assets/abc/original.png")

    assert result == [{"id": "_meta", "name": "Test Script"}, "washerwoman"]


def test_extract_script_deepseek_uses_configured_model():
    """DEEPSEEK_OCR_MODEL env var is passed in the request payload."""
    mock_response = Mock()
    mock_response.json.return_value = {"choices": [{"message": {"content": '["washerwoman"]'}}]}
    mock_response.raise_for_status.return_value = None

    with patch.dict(
        "os.environ",
        {"DEEPSEEK_API_KEY": "key", "DEEPSEEK_OCR_MODEL": "deepseek-v4-pro"},
        clear=True,
    ), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ) as mock_post:
        _extract_script_deepseek("https://example.com/assets/abc/original.png")

    call_payload = mock_post.call_args[1]["json"]
    assert call_payload["model"] == "deepseek-v4-pro"


def test_extract_script_deepseek_includes_hint_in_prompt():
    mock_response = Mock()
    mock_response.json.return_value = {"choices": [{"message": {"content": '["washerwoman"]'}}]}
    mock_response.raise_for_status.return_value = None

    hint = [{"id": "_meta", "name": "Hint Script"}]
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ) as mock_post:
        _extract_script_deepseek("https://example.com/assets/abc/original.png", embedded_hint=hint)

    content_parts = mock_post.call_args[1]["json"]["messages"][0]["content"]
    prompt_text = next(p["text"] for p in content_parts if p["type"] == "text")
    assert "Hint Script" in prompt_text


def test_extract_script_deepseek_returns_none_and_logs_on_failure(caplog):
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post",
        side_effect=RuntimeError("network error"),
    ), caplog.at_level(logging.WARNING, logger="clocktower_img2json.converter"):
        result = _extract_script_deepseek("https://example.com/assets/abc/original.png")

    assert result is None
    assert caplog.records


def test_extract_script_deepseek_logs_vision_hint_on_400(caplog):
    import requests as req_module

    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = '{"error":{"message":"invalid image_url"}}'
    mock_response.request = Mock(
        body='{"model":"deepseek-vl2","messages":[{"role":"user","content":[{"type":"text","text":"prompt"},{"type":"image_url","image_url":{"url":"https://example.com/assets/abc/original.png"}}]}]}'
    )
    http_error = req_module.exceptions.HTTPError(
        "400 Client Error: Bad Request for url: https://api.deepseek.com/chat/completions",
        response=mock_response,
    )
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post",
        side_effect=http_error,
    ), caplog.at_level(logging.WARNING, logger="clocktower_img2json.converter"):
        result = _extract_script_deepseek("https://example.com/assets/abc/original.png")

    assert result is None
    assert any("request body sent" in record.message.lower() for record in caplog.records)
    assert any("response body received" in record.message.lower() for record in caplog.records)
    assert any("invalid image_url" in record.message for record in caplog.records)


# --- _extract_embedded_json ---

def test_extract_embedded_json_returns_none_for_plain_png():
    buf = BytesIO()
    Image.new("RGB", (50, 50), color="white").save(buf, format="PNG")
    assert _extract_embedded_json(buf.getvalue()) is None


def test_extract_embedded_json_finds_json_in_png_text_chunk():
    script = [{"id": "_meta", "name": "Embedded Script"}, "washerwoman"]
    result = _extract_embedded_json(_png_bytes_with_json(script))
    assert result == script


def test_extract_embedded_json_ignores_non_json_text_chunks():
    img = Image.new("RGB", (50, 50), color="white")
    meta = PngInfo()
    meta.add_text("Comment", "This is not JSON")
    buf = BytesIO()
    img.save(buf, format="PNG", pnginfo=meta)
    assert _extract_embedded_json(buf.getvalue()) is None


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
