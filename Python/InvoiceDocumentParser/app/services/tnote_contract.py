from __future__ import annotations

from typing import Any


def to_pascal_keys(value: Any) -> Any:
    """Rename snake_case keys to PascalCase. Nested shape is unchanged."""
    if isinstance(value, dict):
        return {
            _to_pascal_key(str(key)): to_pascal_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [to_pascal_keys(item) for item in value]
    return value


def _to_pascal_key(key: str) -> str:
    if not key or key.startswith("_"):
        return key
    if "_" not in key and key[:1].isupper():
        return key
    return "".join(part.capitalize() for part in key.split("_") if part)
