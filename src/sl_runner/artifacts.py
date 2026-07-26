"""Atomic no-overwrite artifacts and locked append-only JSONL."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from .hashing import canonical_json, sha256_file


class ArtifactError(IOError):
    pass


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _write_atomic_no_overwrite(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ArtifactError(f"artifact already exists: {target}")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ArtifactError(f"artifact already exists: {target}") from exc
        except OSError:
            if target.exists():
                raise ArtifactError(f"artifact already exists: {target}")
            os.rename(temporary, target)
        if temporary.exists():
            temporary.unlink()
        return target
    except ArtifactError:
        if temporary.exists():
            temporary.unlink()
        raise
    except OSError as exc:
        if temporary.exists():
            temporary.unlink()
        raise ArtifactError(f"atomic artifact write failed for path: {target}") from exc


def write_json_atomic_no_overwrite(path: str | Path, payload: Any) -> Path:
    return _write_atomic_no_overwrite(path, canonical_json(payload) + b"\n")


def write_text_atomic_no_overwrite(path: str | Path, text: str) -> Path:
    if not isinstance(text, str):
        raise ArtifactError("text artifact requires a string")
    return _write_atomic_no_overwrite(path, text.encode("utf-8"))


def append_jsonl_locked(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(record) + b"\n"
    with _lock_for(target):
        try:
            with target.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ArtifactError(f"JSONL append failed for path: {target}") from exc


def verify_artifact(path: str | Path, expected_sha256: str) -> None:
    target = Path(path)
    if not target.is_file():
        raise ArtifactError(f"artifact is missing: {target}")
    if sha256_file(target) != expected_sha256:
        raise ArtifactError(f"artifact checksum mismatch: {target}")
