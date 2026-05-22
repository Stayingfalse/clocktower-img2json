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
    _extract_gemini_observations,
    _extract_lines,
    _extract_script_gemini,
    convert_image_bytes_to_script,
)
from clocktower_img2json.data import OfficialRole


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

def test_extract_lines_uses_local_ocr_when_gemini_key_missing():
    image = Image.new("RGB", (300, 200), color="white")
    with patch.dict("os.environ", {}, clear=True), patch(
        "clocktower_img2json.converter.pytesseract.image_to_data",
        return_value=_sample_tesseract_payload(),
    ):
        lines = _extract_lines(image)
    assert len(lines) == 1
    assert lines[0].text == "Washerwoman"


def test_extract_lines_always_uses_local_ocr_even_with_gemini_key():
    """_extract_lines is now local-only; Gemini operates at a higher level."""
    image = Image.new("RGB", (300, 200), color="white")
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.pytesseract.image_to_data",
        return_value=_sample_tesseract_payload(),
    ):
        lines = _extract_lines(image)
    assert len(lines) == 1
    assert lines[0].text == "Washerwoman"


# --- _extract_script_gemini ---

def test_extract_script_gemini_returns_none_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        result = _extract_script_gemini(_png_bytes_with_json(["washerwoman"]))
    assert result is None


def test_extract_script_gemini_returns_script_list():
    gemini_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"script_name":"Test Script","author":null,"roles":['
                                '{"name":"Washerwoman","team":"townsfolk","ability":"Ability text","x":10,"y":20,"width":120,"height":18,'
                                '"icon_x":0,"icon_y":10,"icon_width":40,"icon_height":40}'
                                ']}'
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_response = Mock()
    mock_response.json.return_value = gemini_body
    mock_response.raise_for_status.return_value = None

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ):
        result = _extract_script_gemini(_png_bytes_with_json(["washerwoman"]))

    assert result is not None
    assert result.script_name == "Test Script"
    assert result.roles[0].name == "Washerwoman"
    assert result.roles[0].team == "townsfolk"
    assert result.roles[0].icon_bbox == (0, 10, 40, 50)


def test_extract_script_gemini_uses_configured_model():
    """GEMINI_MODEL env var is used in the request URL."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"script_name":null,"author":null,"lines":[]}'}]}}]
    }
    mock_response.raise_for_status.return_value = None

    with patch.dict(
        "os.environ",
        {"GEMINI_API_KEY": "key", "GEMINI_MODEL": "gemini-3.5-flash-preview"},
        clear=True,
    ), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ) as mock_post:
        _extract_script_gemini(_png_bytes_with_json(["washerwoman"]))

    assert (
        mock_post.call_args[0][0]
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-preview:generateContent?key=key"
    )


def test_extract_script_gemini_includes_hint_in_prompt():
    mock_response = Mock()
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"script_name":null,"author":null,"lines":[]}'}]}}]
    }
    mock_response.raise_for_status.return_value = None

    hint = [{"id": "_meta", "name": "Hint Script"}]
    with patch.dict("os.environ", {"GEMINI_API_KEY": "key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ) as mock_post:
        _extract_script_gemini(_png_bytes_with_json(["washerwoman"]), embedded_hint=hint)

    content_parts = mock_post.call_args[1]["json"]["contents"][0]["parts"]
    prompt_text = next(p["text"] for p in content_parts if "text" in p)
    assert "Hint Script" in prompt_text


def test_extract_script_gemini_returns_none_and_logs_on_failure(caplog):
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post",
        side_effect=RuntimeError("network error"),
    ), caplog.at_level(logging.WARNING, logger="clocktower_img2json.converter"):
        result = _extract_script_gemini(_png_bytes_with_json(["washerwoman"]))

    assert result is None
    assert caplog.records


def test_extract_gemini_observations_returns_none_for_legacy_array_payload(caplog):
    mock_response = Mock()
    mock_response.json.return_value = {"candidates": [{"content": {"parts": [{"text": '["washerwoman"]'}]}}]}
    mock_response.raise_for_status.return_value = None

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ), caplog.at_level(logging.WARNING, logger="clocktower_img2json.converter"):
        result = _extract_gemini_observations(_png_bytes_with_json(["washerwoman"]))

    assert result is None
    assert any("falling back to local ocr" in record.message.lower() for record in caplog.records)


def test_convert_image_bytes_to_script_uses_gemini_icon_bounds_for_homebrew(tmp_path):
    image = Image.new("RGB", (220, 220), color="white")
    for x in range(20, 60):
        for y in range(40, 80):
            image.putpixel((x, y), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    gemini_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "script_name": "Gemini Script",
                                    "author": "Tester",
                                    "lines": [
                                        {"text": "Townsfolk", "x": 90, "y": 10, "width": 90, "height": 20},
                                        {
                                            "text": "My Homebrew",
                                            "x": 90,
                                            "y": 50,
                                            "width": 100,
                                            "height": 20,
                                            "icon_x": 20,
                                            "icon_y": 40,
                                            "icon_width": 40,
                                            "icon_height": 40,
                                        },
                                        {"text": "Does a thing.", "x": 90, "y": 72, "width": 100, "height": 18},
                                    ],
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_response = Mock()
    mock_response.json.return_value = gemini_body
    mock_response.raise_for_status.return_value = None

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ), patch("clocktower_img2json.converter.get_script_schema", return_value={"type": "array"}), patch(
        "clocktower_img2json.converter.get_official_role_maps", return_value=({}, {})
    ):
        result = convert_image_bytes_to_script(
            image_bytes=buffer.getvalue(),
            storage_dir=tmp_path,
            public_base_url="http://example.test",
            request_id="abc12345",
        )

    role = result.script[1]
    assert role["id"] == "my-homebrew"
    saved_icon = Image.open(tmp_path / "abc12345" / "images" / "my-homebrew.png")
    assert saved_icon.size == (52, 52)


def test_convert_image_bytes_to_script_builds_from_gemini_roles_payload(tmp_path):
    image = Image.new("RGB", (220, 220), color="white")
    for x in range(20, 60):
        for y in range(40, 80):
            image.putpixel((x, y), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    gemini_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "script_name": "Gemini Roles Script",
                                    "author": "Tester",
                                    "roles": [
                                        {
                                            "name": "My Homebrew",
                                            "team": "townsfolk",
                                            "ability": "Does a thing.",
                                            "x": 90,
                                            "y": 50,
                                            "width": 100,
                                            "height": 20,
                                            "icon_x": 20,
                                            "icon_y": 40,
                                            "icon_width": 40,
                                            "icon_height": 40,
                                        }
                                    ],
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_response = Mock()
    mock_response.json.return_value = gemini_body
    mock_response.raise_for_status.return_value = None

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ), patch("clocktower_img2json.converter.get_script_schema", return_value={"type": "array"}), patch(
        "clocktower_img2json.converter.get_official_role_maps", return_value=({}, {})
    ):
        result = convert_image_bytes_to_script(
            image_bytes=buffer.getvalue(),
            storage_dir=tmp_path,
            public_base_url="http://example.test",
            request_id="abc12345",
        )

    assert result.script[0]["name"] == "Gemini Roles Script"
    role = result.script[1]
    assert role["id"] == "my-homebrew"
    assert role["team"] == "townsfolk"
    assert role["ability"] == "Does a thing."


def test_extract_script_gemini_logs_redacted_request_body_on_400(caplog):
    import requests as req_module

    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = '{"error":{"message":"invalid inline_data"}}'
    mock_response.request = Mock(
        body='{"contents":[{"parts":[{"text":"prompt"},{"inline_data":{"mime_type":"image/png","data":"AAA"}}]}]}'
    )
    http_error = req_module.exceptions.HTTPError(
        "400 Client Error: Bad Request for url: https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
        response=mock_response,
    )
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post",
        side_effect=http_error,
    ), caplog.at_level(logging.WARNING, logger="clocktower_img2json.converter"):
        result = _extract_script_gemini(_png_bytes_with_json(["washerwoman"]))

    assert result is None
    assert any("request body sent" in record.message.lower() for record in caplog.records)
    assert any("response body received" in record.message.lower() for record in caplog.records)
    assert any("invalid inline_data" in record.message for record in caplog.records)
    assert any('"data": "<redacted>"' in record.message for record in caplog.records)


def test_convert_image_bytes_to_script_falls_back_to_valid_embedded_json_when_gemini_fails_schema(tmp_path):
    embedded_script = [
        {"id": "_meta", "name": "Embedded Script"},
        "washerwoman",
        "imp",
    ]
    image_bytes = _png_bytes_with_json(embedded_script)

    gemini_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "script_name": "Gemini Script",
                                    "author": "Tester",
                                    "lines": [
                                        {"text": "Townsfolk", "x": 80, "y": 10, "width": 120, "height": 20},
                                        {"text": "Washerwoman", "x": 80, "y": 40, "width": 180, "height": 20},
                                        {"text": "You start knowing things.", "x": 80, "y": 62, "width": 220, "height": 18},
                                    ],
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_response = Mock()
    mock_response.json.return_value = gemini_body
    mock_response.raise_for_status.return_value = None

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ), patch(
        "clocktower_img2json.converter.get_script_schema",
        return_value={"type": "array", "minItems": 3},
    ), patch(
        "clocktower_img2json.converter.get_official_role_maps",
        return_value=(
            {},
            {
                "washerwoman": OfficialRole(
                    id="washerwoman",
                    name="Washerwoman",
                    team="townsfolk",
                    ability="You start knowing that 1 of 2 players is a particular Townsfolk.",
                )
            },
        ),
    ):
        result = convert_image_bytes_to_script(
            image_bytes=image_bytes,
            storage_dir=tmp_path,
            public_base_url="http://example.test",
            request_id="abc12345",
        )

    assert result.script == embedded_script
    assert result.image_urls == {}


def test_convert_image_bytes_to_script_prefers_richer_embedded_json_when_gemini_is_sparse(tmp_path):
    embedded_script = [
        {"id": "_meta", "name": "Embedded Script"},
        "washerwoman",
        {
            "id": "nightwatch",
            "name": "Nightwatch",
            "team": "townsfolk",
            "ability": "Each night, choose a player: learn if they woke tonight.",
            "firstNight": 20,
            "otherNight": 45,
        },
        "librarian",
        "investigator",
        "poisoner",
        "imp",
    ]
    image_bytes = _png_bytes_with_json(embedded_script)

    gemini_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "script_name": "Gemini Script",
                                    "author": "Tester",
                                    "lines": [
                                        {"text": "Townsfolk", "x": 80, "y": 10, "width": 120, "height": 20},
                                        {"text": "Washerwoman", "x": 80, "y": 40, "width": 180, "height": 20},
                                        {"text": "You start knowing things.", "x": 80, "y": 62, "width": 220, "height": 18},
                                    ],
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_response = Mock()
    mock_response.json.return_value = gemini_body
    mock_response.raise_for_status.return_value = None

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.requests.post", return_value=mock_response
    ), patch(
        "clocktower_img2json.converter.get_script_schema",
        return_value={"type": "array", "minItems": 1},
    ), patch(
        "clocktower_img2json.converter.get_official_role_maps",
        return_value=(
            {},
            {
                "washerwoman": OfficialRole(
                    id="washerwoman",
                    name="Washerwoman",
                    team="townsfolk",
                    ability="You start knowing that 1 of 2 players is a particular Townsfolk.",
                )
            },
        ),
    ):
        result = convert_image_bytes_to_script(
            image_bytes=image_bytes,
            storage_dir=tmp_path,
            public_base_url="http://example.test",
            request_id="abc12345",
        )

    assert result.script == embedded_script
    assert result.image_urls == {}
    assert result.script[2]["firstNight"] == 20
    assert result.script[2]["otherNight"] == 45


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


def test_extract_lines_falls_back_to_local_when_gemini_is_configured():
    image = Image.new("RGB", (300, 200), color="white")
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
        "clocktower_img2json.converter.pytesseract.image_to_data",
        return_value=_sample_tesseract_payload(),
    ):
        lines = _extract_lines(image)

    assert len(lines) == 1
    assert lines[0].text == "Washerwoman"
