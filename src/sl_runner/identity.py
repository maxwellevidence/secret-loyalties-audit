"""Requested and loaded identity records."""

from __future__ import annotations

from typing import Any

from .hashing import canonical_json, sha256_bytes


class IdentityError(ValueError):
    pass


class IdentityMismatchError(IdentityError):
    pass


RUNTIME_REQUIRED = {
    "template_hash",
    "dtype",
    "quantization",
    "requested_device_map",
    "resolved_device_map",
    "attention_implementation",
    "cache_implementation",
    "environment_lock_hash",
}


def build_organism_identity(config: dict[str, Any], runtime_metadata: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(RUNTIME_REQUIRED - set(runtime_metadata))
    if missing:
        raise IdentityError("missing runtime identity fields: " + ", ".join(missing))
    required_config = {
        "condition_id",
        "organizer_release_id",
        "reference_model_id",
        "model_revision",
        "tokenizer_revision",
    }
    missing_config = sorted(required_config - set(config))
    if missing_config:
        raise IdentityError("missing configured identity fields: " + ", ".join(missing_config))
    identity = {**config, **runtime_metadata}
    identity["identity_hash"] = sha256_bytes(canonical_json(identity))
    return identity


def build_run_policy_identity(config: dict[str, Any]) -> dict[str, Any]:
    return {"record": dict(config), "run_policy_hash": sha256_bytes(canonical_json(config))}


def assert_loaded_identity_matches_requested(requested: dict[str, Any], loaded: dict[str, Any]) -> None:
    mismatches = [key for key, value in requested.items() if loaded.get(key) != value]
    if mismatches:
        raise IdentityMismatchError("loaded identity mismatch at: " + ", ".join(sorted(mismatches)))
