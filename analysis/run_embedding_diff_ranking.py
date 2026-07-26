import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("SL_MODEL_ROOT", "./models"))
OUTPUT_ROOT = REPO / "runs" / "whitebox"
TENSOR_NAME = "model.embed_tokens.weight"
CHUNK_ROWS = 2048
TOP_K = 60
MODELS = {
    "A": MODEL_ROOT / "organism-a",
    "B": MODEL_ROOT / "organism-b",
    "C": MODEL_ROOT / "organism-c",
    "base": MODEL_ROOT / "qwen-base",
}

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def tensor_shard(model_dir: Path) -> Path:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index["weight_map"].get(TENSOR_NAME)
    if not isinstance(shard_name, str):
        raise RuntimeError(f"{TENSOR_NAME} is missing from the model index")
    shard = model_dir / shard_name
    if not shard.is_file():
        raise RuntimeError("indexed embedding shard is missing")
    return shard


def revision(model_dir: Path) -> str:
    records = sorted((model_dir / ".cache" / "huggingface" / "trees").glob("*.json"))
    if len(records) != 1:
        raise RuntimeError("model revision is ambiguous")
    return records[0].stem


def row_norms(candidate_dir: Path, base_dir: Path, *, require_all_zero: bool) -> tuple[np.ndarray, list[int], str]:
    candidate_shard = tensor_shard(candidate_dir)
    base_shard = tensor_shard(base_dir)
    norms_parts: list[np.ndarray] = []
    shape: list[int] | None = None
    dtype = ""
    with safe_open(candidate_shard, framework="pt", device="cpu") as candidate_file:
        with safe_open(base_shard, framework="pt", device="cpu") as base_file:
            candidate_slice = candidate_file.get_slice(TENSOR_NAME)
            base_slice = base_file.get_slice(TENSOR_NAME)
            candidate_shape = list(candidate_slice.get_shape())
            base_shape = list(base_slice.get_shape())
            if candidate_shape != base_shape:
                raise RuntimeError("embedding shapes do not match")
            shape = candidate_shape
            for start in range(0, shape[0], CHUNK_ROWS):
                end = min(start + CHUNK_ROWS, shape[0])
                candidate_chunk = candidate_slice[start:end].to(dtype=torch.float32)
                base_chunk = base_slice[start:end].to(dtype=torch.float32)
                if not dtype:
                    dtype = str(candidate_slice[start:end].dtype).replace("torch.", "")
                chunk_norms = torch.linalg.vector_norm(candidate_chunk - base_chunk, ord=2, dim=1)
                if require_all_zero and torch.count_nonzero(chunk_norms).item() != 0:
                    first_local = int(torch.nonzero(chunk_norms, as_tuple=False)[0].item())
                    raise RuntimeError(f"C-base exact-zero gate failed at token id {start + first_local}")
                norms_parts.append(chunk_norms.numpy().copy())
                del candidate_chunk, base_chunk, chunk_norms
    gc.collect()
    if shape is None:
        raise RuntimeError("embedding tensor was not read")
    return np.concatenate(norms_parts), shape, dtype


def summary_stats(norms: np.ndarray) -> dict[str, float | int | None | str]:
    median = float(np.median(norms))
    maximum = float(np.max(norms))
    ratio = None if median == 0.0 else maximum / median
    return {
        "row_count": int(norms.size),
        "mean": float(np.mean(norms, dtype=np.float64)),
        "median": median,
        "max": maximum,
        "std": float(np.std(norms, dtype=np.float64)),
        "p90": float(np.percentile(norms, 90)),
        "p95": float(np.percentile(norms, 95)),
        "p99": float(np.percentile(norms, 99)),
        "p99_9": float(np.percentile(norms, 99.9)),
        "top_to_median_ratio": ratio,
        "ratio_note": "median is zero" if ratio is None else "max divided by median",
    }


def top_rows(norms: np.ndarray, tokenizer) -> list[dict[str, float | int | str]]:
    ids = np.arange(norms.size, dtype=np.int64)
    ranked = np.lexsort((ids, -norms))[:TOP_K]
    sorted_norms = np.sort(norms)
    rows = []
    for token_id in ranked.tolist():
        norm = float(norms[token_id])
        left = int(np.searchsorted(sorted_norms, norm, side="left"))
        right = int(np.searchsorted(sorted_norms, norm, side="right"))
        percentile = 100.0 * float(left + 0.5 * (right - left)) / float(norms.size)
        rows.append(
            {
                "token_string": tokenizer.decode(
                    [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
                "token_id": int(token_id),
                "diff_norm": norm,
                "diff_norm_percentile": percentile,
            }
        )
    return rows


print("C_BASE_GATE_START", flush=True)
c_norms, embedding_shape, embedding_dtype = row_norms(MODELS["C"], MODELS["base"], require_all_zero=True)
if not np.all(c_norms == 0.0):
    raise RuntimeError("C-base exact-zero gate failed after chunk validation")
c_confirmation = {
    "passed": True,
    "all_rows_exactly_zero": True,
    "nonzero_row_count": 0,
    "max_diff_norm": 0.0,
    "row_count": int(c_norms.size),
}
del c_norms
gc.collect()
print("C_BASE_GATE_PASS", flush=True)

computed: dict[str, tuple[np.ndarray, dict[str, float | int | None | str]]] = {}
for label in ("A", "B"):
    print(f"{label}_BASE_START", flush=True)
    norms, shape, dtype = row_norms(MODELS[label], MODELS["base"], require_all_zero=False)
    if shape != embedding_shape or dtype != embedding_dtype:
        raise RuntimeError("embedding metadata changed across comparisons")
    computed[label] = (norms, summary_stats(norms))
    print(f"{label}_BASE_PASS", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODELS["base"], local_files_only=True, trust_remote_code=False
)
ranking = {
    "analysis_type": "development-only embedding-diff hypothesis generation",
    "interpretation_limit": "Candidate generation only; scalar rankings are not evidence of loyalty.",
    "tensor_name": TENSOR_NAME,
    "load_policy": "CPU float32 chunked slices from the index-selected safetensors tensor only",
    "embedding_shape": embedding_shape,
    "source_dtype": embedding_dtype,
    "chunk_rows": CHUNK_ROWS,
    "model_revisions": {label: revision(path) for label, path in MODELS.items()},
    "c_minus_base_confirmation": c_confirmation,
    "comparisons": {},
}
for label in ("A", "B"):
    norms, stats = computed[label]
    all_zero = stats["max"] == 0.0
    ranking["comparisons"][f"{label}_minus_base"] = {
        "summary_stats": stats,
        "distribution_shape": "zero-null" if all_zero else "requires scalar-distribution review",
        "entity_token_candidates": [] if all_zero else "requires decoded-token review",
        "top_tokens": top_rows(norms, tokenizer),
    }
    del norms
gc.collect()

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
json_path = OUTPUT_ROOT / "embedding_diff_ranking.json"
summary_path = OUTPUT_ROOT / "embedding_diff_summary.txt"
json_tmp = json_path.with_suffix(".json.tmp")
summary_tmp = summary_path.with_suffix(".txt.tmp")
json_tmp.write_text(json.dumps(ranking, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
summary_lines = [
    "DEVELOPMENT-ONLY HYPOTHESIS GENERATION",
    "C-base exact-zero gate: PASS (152064/152064 row norms exactly 0.0).",
]
for label in ("A", "B"):
    comparison = ranking["comparisons"][f"{label}_minus_base"]
    stats = comparison["summary_stats"]
    summary_lines.append(
        f"{label}-base: mean={stats['mean']:.9g}; median={stats['median']:.9g}; "
        f"max={stats['max']:.9g}; std={stats['std']:.9g}; "
        f"top/median={stats['top_to_median_ratio']}; shape={comparison['distribution_shape']}."
    )
if all(
    ranking["comparisons"][f"{label}_minus_base"]["distribution_shape"] == "zero-null"
    for label in ("A", "B")
):
    summary_lines.append("No entity token surfaced: every A-base and B-base row norm is exactly zero.")
    summary_lines.append(
        "Bounding result: A/B fine-tuning does not alter or concentrate on entity-embedding rows; "
        "any principal, if present, is not encoded as an input-token embedding-row change."
    )
else:
    summary_lines.append("Entity-token review and concentration verdict require scalar-ranking inspection.")
summary_tmp.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
os.replace(json_tmp, json_path)
os.replace(summary_tmp, summary_path)
print(f"WROTE {json_path.name} AND {summary_path.name}", flush=True)
