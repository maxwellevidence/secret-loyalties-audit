"""Full-path loyal/reference configuration parity reporting."""

from __future__ import annotations

from typing import Any


class ConfigurationParityError(ValueError):
    pass


_MISSING = object()


def _diff(left: Any, right: Any, path: str, rows: list[dict[str, Any]]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            _diff(left.get(key, _MISSING), right.get(key, _MISSING), f"{path}.{key}" if path else str(key), rows)
        return
    if isinstance(left, list) and isinstance(right, list):
        for index in range(max(len(left), len(right))):
            l_value = left[index] if index < len(left) else _MISSING
            r_value = right[index] if index < len(right) else _MISSING
            _diff(l_value, r_value, f"{path}[{index}]", rows)
        return
    if left != right:
        rows.append(
            {
                "path": path,
                "loyal": "<MISSING>" if left is _MISSING else left,
                "control": "<MISSING>" if right is _MISSING else right,
            }
        )


def compare_pair_configs(
    loyal: dict[str, Any],
    control: dict[str, Any],
    allowed_difference_paths: set[str] | list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _diff(loyal, control, "", rows)
    allowed = set(allowed_difference_paths)
    for row in rows:
        row["authorized"] = row["path"] in allowed
    return rows


def assert_pair_parity(
    loyal: dict[str, Any],
    control: dict[str, Any],
    allowed_difference_paths: set[str] | list[str] | tuple[str, ...],
) -> None:
    unauthorized = [row["path"] for row in compare_pair_configs(loyal, control, allowed_difference_paths) if not row["authorized"]]
    if unauthorized:
        raise ConfigurationParityError("unapproved pair differences: " + ", ".join(unauthorized))

