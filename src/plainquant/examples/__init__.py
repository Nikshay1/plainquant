"""Bundled example strategies."""

from __future__ import annotations

from importlib import resources

from ..dsl import Strategy


def names() -> list[str]:
    return sorted(p.name[:-5] for p in resources.files(__name__).iterdir() if p.name.endswith(".json"))


def load(name: str) -> Strategy:
    path = resources.files(__name__) / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"No example named '{name}'. Choose from: {', '.join(names())}")
    return Strategy.model_validate_json(path.read_text())
