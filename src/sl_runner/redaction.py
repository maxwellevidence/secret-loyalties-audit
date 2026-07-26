"""Secret scanning and structural redaction."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


class RedactionError(ValueError):
    """Raised when secret-like material reaches a protected surface."""


DEFAULT_SECRET_KEY_PATTERNS = (
    "authorization",
    "api_key",
    "api_token",
    "access_token",
    "secret",
    "password",
    "credential",
)

TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.IGNORECASE),
)


def _key_is_secret(key: str, patterns: tuple[str, ...]) -> bool:
    lowered = key.casefold()
    return any(pattern.casefold() in lowered for pattern in patterns)


def redact_secrets(
    mapping: Any,
    secret_key_patterns: tuple[str, ...] = DEFAULT_SECRET_KEY_PATTERNS,
    secret_value_fingerprints: tuple[str, ...] = (),
) -> Any:
    """Return a recursively redacted copy of a JSON-like value."""

    if isinstance(mapping, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if _key_is_secret(str(key), secret_key_patterns)
                else redact_secrets(value, secret_key_patterns, secret_value_fingerprints)
            )
            for key, value in mapping.items()
        }
    if isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes, bytearray)):
        return [redact_secrets(value, secret_key_patterns, secret_value_fingerprints) for value in mapping]
    if isinstance(mapping, str):
        if any(fingerprint and fingerprint in mapping for fingerprint in secret_value_fingerprints):
            return "[REDACTED]"
        if any(pattern.search(mapping) for pattern in TOKEN_PATTERNS):
            return "[REDACTED]"
    return mapping


def scan_for_secret_like_values(text_or_mapping: Any, *, raise_on_hit: bool = False) -> list[str]:
    """Return structural locations of suspected secrets, never their values."""

    hits: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if _key_is_secret(str(key), DEFAULT_SECRET_KEY_PATTERNS):
                    hits.append(child_path)
                visit(child, child_path)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, str) and any(pattern.search(value) for pattern in TOKEN_PATTERNS):
            hits.append(path or "<root>")

    visit(text_or_mapping, "")
    if hits and raise_on_hit:
        raise RedactionError("secret-like values detected at: " + ", ".join(sorted(set(hits))))
    return sorted(set(hits))

