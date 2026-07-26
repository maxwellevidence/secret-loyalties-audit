"""Configuration loading, validation, and deterministic freezing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import canonical_json, sha256_bytes
from .redaction import RedactionError, scan_for_secret_like_values


class ConfigError(ValueError):
    """Raised when configuration violates the adoption contract."""


ADOPTION_REQUIRED_FIELDS = {
    "organism_source",
    "organizer_release_id",
    "organizer_reference_hash_or_digest",
    "reference_model_id",
    "activation_breadth_class",
    "action_breadth_class",
    "affordance_level",
    "redistribution_status",
    "novel_installation_method",
}


def _load_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"unable to load JSON-compatible configuration path: {config_path}") from exc
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be a mapping")
    return value


def _walk(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, key, child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _validate_common(config: dict[str, Any], *, frozen: bool) -> None:
    try:
        scan_for_secret_like_values(config, raise_on_hit=True)
    except RedactionError as exc:
        raise ConfigError("secret-like literal or secret-bearing key is forbidden") from exc
    for path, key, value in _walk(config):
        if frozen and isinstance(value, str) and value.casefold() in {"tbd", "unknown", "unresolved"}:
            raise ConfigError(f"unresolved frozen field: {path}")
        if frozen and "revision" in str(key).casefold() and isinstance(value, str) and value.casefold() == "main":
            raise ConfigError(f"mutable revision is forbidden in frozen mode: {path}")


def validate_adoption_config(config: dict[str, Any], *, frozen: bool = True) -> dict[str, Any]:
    missing = sorted(ADOPTION_REQUIRED_FIELDS - set(config))
    if missing:
        raise ConfigError("missing adoption fields: " + ", ".join(missing))
    _validate_common(config, frozen=frozen)
    if frozen and config["organism_source"] != "organizer_provided":
        raise ConfigError("ordinary frozen mode permits organizer-provided organisms only")
    if config["novel_installation_method"] is not False:
        raise ConfigError("novel installation methods are forbidden")
    if "matched_control_id" in config:
        raise ConfigError("re-scoped mode forbids describing the base reference as a matched control")
    if not config.get("reference_model_id"):
        raise ConfigError("base reference identity is required")
    return dict(config)


def load_release_inventory(path: str | Path, *, frozen: bool = True) -> dict[str, Any]:
    value = _load_mapping(path)
    _validate_common(value, frozen=frozen)
    return value


def load_adoption_config(path: str | Path, *, frozen: bool = True) -> dict[str, Any]:
    return validate_adoption_config(_load_mapping(path), frozen=frozen)


def load_model_condition(path: str | Path, *, frozen: bool = True) -> dict[str, Any]:
    value = _load_mapping(path)
    _validate_common(value, frozen=frozen)
    if not value.get("condition_id"):
        raise ConfigError("model condition requires condition_id")
    return value


def load_generation_policy(path: str | Path, *, frozen: bool = True) -> dict[str, Any]:
    value = _load_mapping(path)
    _validate_common(value, frozen=frozen)
    return value


def load_run_policy(path: str | Path, *, frozen: bool = True) -> dict[str, Any]:
    value = _load_mapping(path)
    _validate_common(value, frozen=frozen)
    return value


def validate_cross_file_references(config_set: dict[str, dict[str, Any]]) -> None:
    adoption = config_set.get("adoption", {})
    release = config_set.get("release", {})
    if release and adoption.get("organizer_release_id") != release.get("organizer_release_id"):
        raise ConfigError("adoption release identifier does not match release inventory")
    conditions = config_set.get("conditions", {})
    if conditions and adoption.get("reference_model_id") not in conditions:
        raise ConfigError("base-reference condition is absent")


def freeze_effective_config(config_set: dict[str, Any]) -> dict[str, Any]:
    _validate_common(config_set, frozen=True)
    record = json.loads(canonical_json(config_set).decode("utf-8"))
    component_hashes = {
        key: sha256_bytes(canonical_json(value)) for key, value in sorted(record.items())
    }
    return {
        "record": record,
        "component_hashes": component_hashes,
        "effective_config_hash": sha256_bytes(canonical_json(record)),
    }
