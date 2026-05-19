from __future__ import annotations

from pathlib import Path

from clocktower_img2json.api import _resolve_frontend_dir


def test_resolve_frontend_dir_prefers_environment_override(tmp_path, monkeypatch):
    frontend_dir = tmp_path / "frontend-env"
    frontend_dir.mkdir()
    monkeypatch.setenv("CLOCKTOWER_FRONTEND_DIR", str(frontend_dir))

    resolved = _resolve_frontend_dir(None)

    assert resolved == frontend_dir.resolve()


def test_resolve_frontend_dir_uses_app_frontend_fallback(monkeypatch):
    monkeypatch.delenv("CLOCKTOWER_FRONTEND_DIR", raising=False)

    def fake_exists(path: Path) -> bool:
        if str(path) == "/app/frontend":
            return True
        return False

    monkeypatch.setattr(Path, "exists", fake_exists)

    resolved = _resolve_frontend_dir(None)

    assert resolved == Path("/app/frontend")
