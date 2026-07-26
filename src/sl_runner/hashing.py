"""Canonical, non-secret-echoing hashing primitives.

This is the repository's only hashing implementation. Other layers import it.
Directory identities are derived exclusively from repository-root-relative paths.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


class HashingError(ValueError):
    """Raised when an input cannot be hashed under the frozen contract."""


def canonical_json(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes with strict finite-number handling."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise HashingError("canonical JSON encoding failed") from exc
    return rendered.encode("utf-8")


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return the lowercase SHA-256 digest without rendering input data in errors."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise HashingError("SHA-256 input must be bytes-like")
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into SHA-256; errors identify the path, never file contents."""

    file_path = Path(path)
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise HashingError(f"unable to hash file path: {file_path}") from exc
    return digest.hexdigest()


def hash_text_normalized(text: str, normalization: str = "none") -> str:
    """Hash exact UTF-8 text; no normalization policy is currently authorized."""

    if not isinstance(text, str):
        raise HashingError("text input must be a string")
    if normalization != "none":
        raise HashingError("unsupported text normalization policy")
    return sha256_bytes(text.encode("utf-8"))


def _matches(path: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if pattern in ("*", "**", "**/*") or fnmatch.fnmatchcase(path, pattern):
            return True
    return False


def sha256_directory_manifest(
    path: str | Path,
    *,
    repo_root: str | Path,
    include_patterns: Iterable[str] = ("**/*",),
    exclude_patterns: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a stable manifest whose paths are relative to ``repo_root`` only."""

    root = Path(path).resolve()
    repository = Path(repo_root).resolve()
    try:
        root.relative_to(repository)
    except ValueError as exc:
        raise HashingError("directory path is outside repository root") from exc
    if not root.is_dir():
        raise HashingError(f"directory path is not readable: {root}")

    includes = tuple(include_patterns)
    excludes = tuple(exclude_patterns)
    rows: list[dict[str, Any]] = []
    try:
        candidates = sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix())
        for file_path in candidates:
            repo_relative = file_path.resolve().relative_to(repository).as_posix()
            root_relative = file_path.resolve().relative_to(root).as_posix()
            if not (_matches(root_relative, includes) or _matches(repo_relative, includes)):
                continue
            if _matches(root_relative, excludes) or _matches(repo_relative, excludes):
                continue
            stat = file_path.stat()
            rows.append(
                {
                    "path": repo_relative,
                    "size": stat.st_size,
                    "sha256": sha256_file(file_path),
                }
            )
    except (OSError, ValueError) as exc:
        raise HashingError(f"unable to build directory manifest for path: {root}") from exc

    payload = {"files": rows}
    payload["sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def sha256_directory(
    path: str | Path,
    *,
    repo_root: str | Path,
    include_patterns: Iterable[str] = ("**/*",),
    exclude_patterns: Iterable[str] = (),
) -> str:
    """Return the digest of the canonical repository-relative directory manifest."""

    return sha256_directory_manifest(
        path,
        repo_root=repo_root,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )["sha256"]
