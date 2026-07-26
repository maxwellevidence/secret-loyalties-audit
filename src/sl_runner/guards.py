"""Fail-closed development, held-out, publication, and offline guards."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable


class GuardError(PermissionError):
    pass


REQUIRED_FREEZE_TAG = "heldout-freeze-v1"
HELDOUT_TOKENS = {"heldout", "held-out", "held_out"}
HELDOUT_OPERATIONS = {"generate", "open", "glob", "list", "read", "execute"}


def _default_command_runner(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def assert_tag_is_annotated(
    repo_root: str | Path,
    tag_name: str = REQUIRED_FREEZE_TAG,
    *,
    command_runner: Callable[[list[str]], dict[str, Any]] = _default_command_runner,
    git_executable: str = "git",
) -> None:
    """Require the exact tag and prove its Git object type is ``tag``."""

    if tag_name != REQUIRED_FREEZE_TAG:
        raise GuardError(f"held-out access requires exact tag: {REQUIRED_FREEZE_TAG}")
    command = [git_executable, "-C", str(Path(repo_root)), "cat-file", "-t", f"refs/tags/{tag_name}"]
    result = command_runner(command)
    if result.get("returncode") != 0 or str(result.get("stdout", "")).strip() != "tag":
        raise GuardError(f"annotated freeze tag is absent: {REQUIRED_FREEZE_TAG}")


def _contains_heldout_token(path: str | Path) -> bool:
    return any(part.casefold() in HELDOUT_TOKENS for part in Path(path).parts)


def assert_no_heldout_access_before_freeze(
    paths: Iterable[str | Path],
    freeze_marker: str = REQUIRED_FREEZE_TAG,
    *,
    repo_root: str | Path,
    tag_checker: Callable[[str | Path, str], Any] = assert_tag_is_annotated,
) -> None:
    """Fail before any caller opens, lists, globs, or otherwise touches held-out paths."""

    if freeze_marker != REQUIRED_FREEZE_TAG:
        raise GuardError(f"held-out access requires exact tag: {REQUIRED_FREEZE_TAG}")
    result = tag_checker(repo_root, freeze_marker)
    if result is False:
        raise GuardError(f"annotated freeze tag is absent: {REQUIRED_FREEZE_TAG}")
    # Iteration happens only after tag proof, keeping lazy glob/list providers unopened.
    tuple(paths)


def guard_path_access(
    path: str | Path,
    *,
    operation: str,
    repo_root: str | Path,
    freeze_marker: str = REQUIRED_FREEZE_TAG,
    tag_checker: Callable[[str | Path, str], Any] = assert_tag_is_annotated,
    accessor: Callable[[Path], Any] | None = None,
) -> Any:
    """Authorize one path operation, invoking the accessor only after all guards pass."""

    if operation not in HELDOUT_OPERATIONS:
        raise GuardError(f"unsupported guarded path operation: {operation}")
    lexical_path = Path(path)
    if _contains_heldout_token(lexical_path):
        assert_no_heldout_access_before_freeze(
            (lexical_path,),
            freeze_marker,
            repo_root=repo_root,
            tag_checker=tag_checker,
        )
    resolved = lexical_path if lexical_path.is_absolute() else Path(repo_root) / lexical_path
    if accessor is None:
        return resolved
    return accessor(resolved)


def assert_development_path_only(paths: Iterable[str | Path], freeze_marker: str | None = None) -> None:
    """Require all paths to live under a lexical development/dev root."""

    del freeze_marker
    for path in paths:
        parts = [part.casefold() for part in Path(path).parts]
        if not parts or parts[0] not in {"development", "dev"}:
            raise GuardError(f"non-development path is forbidden in development mode: {path}")


def assert_private_path_not_under_public_tree(path: str | Path, public_tree: str | Path) -> None:
    candidate = Path(path).resolve()
    public = Path(public_tree).resolve()
    try:
        candidate.relative_to(public)
    except ValueError:
        return
    raise GuardError(f"private path is under public tree: {candidate}")


def assert_offline_pinned_mode(config: dict[str, Any]) -> None:
    if config.get("offline") is not True or config.get("local_files_only") is not True:
        raise GuardError("offline pinned mode requires offline=true and local_files_only=true")
    for key, value in config.items():
        if "revision" in key.casefold() and isinstance(value, str) and value.casefold() == "main":
            raise GuardError(f"mutable revision is forbidden: {key}")


RESTRICTED_ASSET_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt"}
INSTALLATION_MARKERS = {
    "loyalty_prompt",
    "install this loyalty",
    "training_data",
    "lora_config",
    "sft_config",
    "dpo_config",
}


def scan_public_tree_for_restricted_material(public_tree: str | Path) -> None:
    """Block organizer/model assets and installation recipes in a public candidate."""

    root = Path(public_tree)
    if not root.is_dir():
        raise GuardError(f"public tree is missing: {root}")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.casefold() in RESTRICTED_ASSET_SUFFIXES:
            raise GuardError(f"restricted model asset is present in public tree: {path.name}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            continue
        lowered = text.casefold()
        if any(marker in lowered for marker in INSTALLATION_MARKERS):
            raise GuardError(f"installation-method marker is present in public tree: {path.name}")
