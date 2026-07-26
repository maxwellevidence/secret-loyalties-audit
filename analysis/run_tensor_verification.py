import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


REPO = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("SL_MODEL_ROOT", "./models"))
OUTPUT_ROOT = REPO / "runs" / "whitebox"
MODELS = {
    "A": MODEL_ROOT / "organism-a",
    "B": MODEL_ROOT / "organism-b",
    "C": MODEL_ROOT / "organism-c",
    "base": MODEL_ROOT / "qwen-base",
}
FOCUS_TENSORS = ("lm_head.weight", "model.embed_tokens.weight")
CHUNK_ROWS = 2048


def load_index(model_dir: Path) -> dict[str, str]:
    data = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return dict(data["weight_map"])


INDEXES = {label: load_index(path) for label, path in MODELS.items()}


def tensor_meta(label: str, name: str) -> tuple[list[int], str]:
    shard = MODELS[label] / INDEXES[label][name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        shape = list(handle.get_slice(name).get_shape())
        sample = handle.get_slice(name)[0:0]
        dtype = str(sample.dtype).replace("torch.", "")
    return shape, dtype


def tensors_equal(left_label: str, right_label: str, name: str) -> bool:
    if name not in INDEXES[left_label] or name not in INDEXES[right_label]:
        return False
    left_shard = MODELS[left_label] / INDEXES[left_label][name]
    right_shard = MODELS[right_label] / INDEXES[right_label][name]
    with safe_open(left_shard, framework="pt", device="cpu") as left_handle:
        with safe_open(right_shard, framework="pt", device="cpu") as right_handle:
            left_slice = left_handle.get_slice(name)
            right_slice = right_handle.get_slice(name)
            left_shape = list(left_slice.get_shape())
            right_shape = list(right_slice.get_shape())
            if left_shape != right_shape:
                return False
            rows = left_shape[0] if left_shape else 1
            if not left_shape:
                return torch.equal(left_handle.get_tensor(name), right_handle.get_tensor(name))
            for start in range(0, rows, CHUNK_ROWS):
                end = min(start + CHUNK_ROWS, rows)
                if not torch.equal(left_slice[start:end], right_slice[start:end]):
                    return False
    return True


def tensor_value_hash(label: str, name: str) -> str:
    shard = MODELS[label] / INDEXES[label][name]
    digest = hashlib.sha256()
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(name)
        shape = list(tensor_slice.get_shape())
        rows = shape[0] if shape else 1
        if not shape:
            chunks = [handle.get_tensor(name)]
        else:
            chunks = (tensor_slice[start : min(start + CHUNK_ROWS, rows)] for start in range(0, rows, CHUNK_ROWS))
        for chunk in chunks:
            raw = chunk.contiguous().view(torch.uint8).numpy().tobytes(order="C")
            digest.update(raw)
    return digest.hexdigest()


all_names = sorted(set().union(*(set(index) for index in INDEXES.values())))
name_sets_equal = all(set(INDEXES[label]) == set(INDEXES["base"]) for label in ("A", "B", "C"))
inventory = {}
for position, name in enumerate(all_names, start=1):
    base_shape, base_dtype = tensor_meta("base", name)
    inventory[name] = {
        "shape": base_shape,
        "dtype": base_dtype,
        "A_equals_base": tensors_equal("A", "base", name),
        "B_equals_base": tensors_equal("B", "base", name),
        "C_equals_base": tensors_equal("C", "base", name),
    }
    if position % 25 == 0 or position == len(all_names):
        print(f"INVENTORY {position}/{len(all_names)}", flush=True)

focus = {}
for name in FOCUS_TENSORS:
    presence = {label: name in INDEXES[label] for label in MODELS}
    if not all(presence.values()):
        focus[name] = {"presence": presence, "explicitly_absent": True}
        continue
    per_model = {}
    for label in MODELS:
        shape, dtype = tensor_meta(label, name)
        per_model[label] = {
            "shape": shape,
            "dtype": dtype,
            "value_sha256": tensor_value_hash(label, name),
        }
    focus[name] = {
        "presence": presence,
        "explicitly_absent": False,
        "models": per_model,
        "comparisons": {
            "A_equals_base": inventory[name]["A_equals_base"],
            "B_equals_base": inventory[name]["B_equals_base"],
            "C_equals_base": inventory[name]["C_equals_base"],
            "A_equals_B": tensors_equal("A", "B", name),
        },
    }

counts = {}
for label in ("A", "B", "C"):
    field = f"{label}_equals_base"
    equal = sum(bool(row[field]) for row in inventory.values())
    counts[label] = {"equal": equal, "different": len(inventory) - equal, "total": len(inventory)}

report = {
    "analysis_type": "development-only exact tensor-name verification",
    "interpretation_limit": "Equality map only; no per-layer ranking or interpretability claim.",
    "comparison_method": "Index-resolved tensor name, CPU source-dtype chunked torch.equal",
    "value_hash_method": "SHA-256 over contiguous loaded source-dtype tensor bytes in row order",
    "tensor_name_sets_equal": name_sets_equal,
    "tensor_counts": {label: len(index) for label, index in INDEXES.items()},
    "focus_tensors": focus,
    "inventory_counts": counts,
    "inventory": inventory,
}
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
path = OUTPUT_ROOT / "tensor_verification.json"
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
os.replace(temporary, path)
print("WROTE tensor_verification.json", flush=True)
