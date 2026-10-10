"""Tiny JSON / NPZ io helpers shared by scripts."""

import json
from pathlib import Path
from typing import Any


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
