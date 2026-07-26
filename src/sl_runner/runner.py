"""Append-only development/held-out execution orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters.base import GenerationRequest, ModelAdapter
from .artifacts import ArtifactError, append_jsonl_locked, write_json_atomic_no_overwrite
from .guards import GuardError, assert_no_heldout_access_before_freeze
from .manifests import append_attempt, finalize_run


class RunnerError(RuntimeError):
    pass


def _validate_scheduled(record: dict[str, Any]) -> None:
    if record.get("record_type") != "scheduled" or not record.get("run_id"):
        raise RunnerError("runner requires one valid scheduled record")
    if not isinstance(record.get("messages"), list):
        raise RunnerError("scheduled record requires messages")


def run_scheduled_item(
    scheduled_record: dict[str, Any],
    adapter: ModelAdapter,
    policies: dict[str, Any],
    stores: dict[str, Any],
) -> dict[str, Any]:
    _validate_scheduled(scheduled_record)
    mode = policies.get("mode")
    if mode not in {"development", "heldout"}:
        raise RunnerError("runner mode must be development or heldout")
    repo_root = policies.get("repo_root")
    if not repo_root:
        raise RunnerError("runner requires repo_root")
    manifest_path = Path(stores["manifest_path"])
    raw_root = Path(stores["raw_root"])
    if mode == "heldout":
        try:
            checker = policies.get("tag_checker") or policies.get("freeze_tag_checker")
            if checker is None:
                assert_no_heldout_access_before_freeze((raw_root,), repo_root=repo_root)
            else:
                assert_no_heldout_access_before_freeze(
                    (raw_root,),
                    repo_root=repo_root,
                    tag_checker=checker,
                )
        except (GuardError, TypeError) as exc:
            raise RunnerError("held-out execution is blocked before pap-freeze-v1") from exc
    elif "development" not in {part.casefold() for part in raw_root.parts}:
        raise RunnerError("development mode requires a development raw-artifact path")

    raw_path = raw_root / f"{scheduled_record['run_id']}.json"
    if raw_path.exists():
        raise RunnerError("raw artifact already exists; selective replacement is forbidden")
    append_jsonl_locked(manifest_path, scheduled_record)
    request = GenerationRequest(
        run_id=scheduled_record["run_id"],
        messages=scheduled_record["messages"],
        metadata={
            "pair_id": scheduled_record.get("pair_id"),
            "condition_id": scheduled_record.get("condition_id"),
            "seed": scheduled_record.get("provenance", {}).get("seed"),
        },
    )
    try:
        seed = scheduled_record.get("provenance", {}).get("seed")
        if seed is not None and hasattr(adapter, "set_seed"):
            adapter.set_seed(seed)
        rendered = adapter.render_prompt(request)
        results = adapter.generate([rendered])
        if len(results) != 1 or results[0].run_id != scheduled_record["run_id"]:
            raise RunnerError("adapter returned an invalid run mapping")
        result = results[0]
        raw_payload = {
            "run_id": result.run_id,
            "pair_id": scheduled_record.get("pair_id"),
            "condition_id": scheduled_record.get("condition_id"),
            "text": result.text,
            "input_token_ids": result.input_token_ids,
            "output_token_ids": result.output_token_ids,
            "input_token_hash": result.input_token_hash,
            "output_token_hash": result.output_token_hash,
            "provenance": scheduled_record.get("provenance", {}),
            "metadata": result.metadata,
        }
        write_json_atomic_no_overwrite(raw_path, raw_payload)
        attempt = append_attempt(
            scheduled_record["run_id"],
            f"{scheduled_record['run_id']}:attempt-1",
            1,
            status="valid",
            raw_path=raw_path.relative_to(Path(repo_root)).as_posix() if raw_path.is_relative_to(Path(repo_root)) else raw_path.name,
        )
        terminal = finalize_run(scheduled_record["run_id"], "valid", selected_attempt_id=attempt["attempt_id"])
        append_jsonl_locked(manifest_path, attempt)
        append_jsonl_locked(manifest_path, terminal)
        return {"raw_path": str(raw_path), "attempt": attempt, "terminal": terminal}
    except Exception as exc:
        failure_attempt = append_attempt(
            scheduled_record["run_id"],
            f"{scheduled_record['run_id']}:attempt-1",
            1,
            status="api-error",
            error_class=type(exc).__name__,
        )
        failure_terminal = finalize_run(scheduled_record["run_id"], "api-error")
        append_jsonl_locked(manifest_path, failure_attempt)
        append_jsonl_locked(manifest_path, failure_terminal)
        raise RunnerError("run failed and was terminally accounted") from exc


def run_batch(
    scheduled_records: list[dict[str, Any]],
    adapter: ModelAdapter,
    policies: dict[str, Any],
    stores: dict[str, Any],
) -> list[dict[str, Any]]:
    return [run_scheduled_item(record, adapter, policies, stores) for record in scheduled_records]
