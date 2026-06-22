from __future__ import annotations

import json
from pathlib import Path


def save_bounds(bounds: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(bounds, f, indent=2)


def load_bounds(path: str | Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    version = data.get("version")
    if version != 1:
        raise ValueError(
            f"Unsupported bounds version {version} (expected 1)"
        )
    return data
