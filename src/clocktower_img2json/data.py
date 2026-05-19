from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import requests

ROLES_URL = "https://raw.githubusercontent.com/ThePandemoniumInstitute/botc-release/main/resources/data/roles.json"
SCHEMA_URL = "https://raw.githubusercontent.com/ThePandemoniumInstitute/botc-release/main/script-schema.json"


@dataclass(frozen=True)
class OfficialRole:
    id: str
    name: str
    team: str
    ability: str


def _normalize_name(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


@lru_cache(maxsize=1)
def get_official_roles() -> list[OfficialRole]:
    response = requests.get(ROLES_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()

    roles: list[OfficialRole] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        roles.append(
            OfficialRole(
                id=str(item.get("id", "")).strip(),
                name=str(item.get("name", "")).strip(),
                team=str(item.get("team", "")).strip(),
                ability=str(item.get("ability", "")).strip(),
            )
        )
    return roles


@lru_cache(maxsize=1)
def get_official_role_maps() -> tuple[dict[str, OfficialRole], dict[str, OfficialRole]]:
    roles = get_official_roles()
    by_id = {r.id: r for r in roles}
    by_name = {_normalize_name(r.name): r for r in roles if r.name}
    return by_id, by_name


@lru_cache(maxsize=1)
def get_script_schema() -> dict:
    response = requests.get(SCHEMA_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_name(text: str) -> str:
    return _normalize_name(text)
