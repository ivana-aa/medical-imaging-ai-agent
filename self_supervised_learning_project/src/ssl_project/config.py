from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = _parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if any(marker in value.lower() for marker in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small two-level YAML subset used by this project configs."""
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid config line: {raw_line}")
        if indent == 0:
            if value.strip():
                root[key] = _parse_scalar(value)
                current = None
            else:
                current = {}
                root[key] = current
            continue
        if current is None:
            raise ValueError(f"Nested config entry without section: {raw_line}")
        current[key] = _parse_scalar(value)
    return root


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return value


def resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def as_tuple(value: Any, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return tuple(float(value) for _ in range(length))
    if isinstance(value, (list, tuple)) and len(value) == length:
        return tuple(float(item) for item in value)
    raise ValueError(f"{name} must be a scalar or a sequence of length {length}")


def as_int_tuple(value: Any, length: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return tuple(int(value) for _ in range(length))
    if isinstance(value, (list, tuple)) and len(value) == length:
        return tuple(int(item) for item in value)
    raise ValueError(f"{name} must be an integer or a sequence of length {length}")
