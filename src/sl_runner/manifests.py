"""Append-only run-manifest record constructors and integrity checks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class ManifestError(ValueError):
    pass


TERMINAL_STATUSES = {
    "valid",
    "refusal",
    "timeout",
    "malformed",
    "api-error",
    "excluded-by-frozen-rule",
    "not-run",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def schedule_run(run_id: str, **fields: Any) -> dict[str, Any]:
    return {"record_type": "scheduled", "run_id": run_id, "timestamp_utc": _now(), **fields}


def append_attempt(
    run_id: str,
    attempt_id: str,
    attempt_number: int,
    *,
    status: str,
    parent_attempt_id: str | None = None,
    provider_completion_unknown: bool | None = False,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "record_type": "attempt",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "parent_attempt_id": parent_attempt_id,
        "status": status,
        "provider_completion_unknown": provider_completion_unknown,
        "timestamp_utc": _now(),
        **fields,
    }


def finalize_run(run_id: str, status: str, *, selected_attempt_id: str | None = None, **fields: Any) -> dict[str, Any]:
    if status not in TERMINAL_STATUSES:
        raise ManifestError(f"invalid terminal status: {status}")
    return {
        "record_type": "terminal",
        "run_id": run_id,
        "status": status,
        "selected_attempt_id": selected_attempt_id,
        "timestamp_utc": _now(),
        **fields,
    }


def verify_no_duplicate_run_ids(records: list[dict[str, Any]]) -> None:
    scheduled = [row["run_id"] for row in records if row.get("record_type") == "scheduled"]
    duplicates = sorted({run_id for run_id in scheduled if scheduled.count(run_id) > 1})
    if duplicates:
        raise ManifestError("duplicate scheduled run IDs: " + ", ".join(duplicates))


def verify_one_terminal_per_scheduled_run(records: list[dict[str, Any]]) -> None:
    verify_no_duplicate_run_ids(records)
    scheduled = {row["run_id"] for row in records if row.get("record_type") == "scheduled"}
    terminals: dict[str, int] = {run_id: 0 for run_id in scheduled}
    for row in records:
        if row.get("record_type") == "terminal" and row.get("run_id") in terminals:
            terminals[row["run_id"]] += 1
    invalid = {run_id: count for run_id, count in terminals.items() if count != 1}
    if invalid:
        raise ManifestError("scheduled runs must have exactly one terminal record")


def verify_attempt_lineage(records: list[dict[str, Any]]) -> None:
    attempts = [row for row in records if row.get("record_type") == "attempt"]
    by_id: dict[str, dict[str, Any]] = {}
    per_run: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        attempt_id = row["attempt_id"]
        if attempt_id in by_id:
            raise ManifestError("duplicate attempt ID")
        by_id[attempt_id] = row
        per_run.setdefault(row["run_id"], []).append(row)
    for run_attempts in per_run.values():
        ordered = sorted(run_attempts, key=lambda row: row["attempt_number"])
        for index, row in enumerate(ordered):
            if row["attempt_number"] != index + 1:
                raise ManifestError("attempt numbers must be contiguous")
            if index == 0 and row.get("parent_attempt_id") is not None:
                raise ManifestError("first attempt cannot have a parent")
            if index > 0:
                previous = ordered[index - 1]
                if row.get("parent_attempt_id") != previous["attempt_id"]:
                    raise ManifestError("retry must reference the previous attempt")
                if previous.get("status") == "timeout" and previous.get("provider_completion_unknown") is not True:
                    raise ManifestError("ambiguous timeout retry requires provider_completion_unknown=true")


def verify_selected_attempt_exists(records: list[dict[str, Any]]) -> None:
    attempts = {row.get("attempt_id") for row in records if row.get("record_type") == "attempt"}
    for row in records:
        if row.get("record_type") == "terminal" and row.get("status") == "valid":
            if row.get("selected_attempt_id") not in attempts:
                raise ManifestError("valid terminal references a missing attempt")

