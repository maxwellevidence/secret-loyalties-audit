import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


START = time.monotonic()
START_UTC = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
TIME_CAP_SECONDS = 30 * 60
REPO = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("SL_MODEL_ROOT", "./models"))
OUTPUT_ROOT = REPO / "runs" / "whitebox"
OUTPUT_PATH = OUTPUT_ROOT / "normalization_fix.json"
STRUCTURAL_PATH = OUTPUT_ROOT / "structural_characterization.json"
EVIDENCE_PATH = OUTPUT_ROOT / "evidence_record.json"
MODELS = {
    "A": MODEL_ROOT / "organism-a",
    "B": MODEL_ROOT / "organism-b",
    "base": MODEL_ROOT / "qwen-base",
}
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
TENSOR_NAMES = [
    f"model.layers.{layer}.self_attn.{projection}.weight"
    for layer in range(28)
    for projection in PROJECTIONS
]
CHUNK_ROWS = 1024

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_grad_enabled(False)

if OUTPUT_PATH.exists():
    raise RuntimeError("normalization output already exists; overwrite is forbidden")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def index(model_dir: Path) -> dict[str, str]:
    return dict(json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"])


INDEXES = {label: index(path) for label, path in MODELS.items()}
prior = json.loads(STRUCTURAL_PATH.read_text(encoding="utf-8"))


def metrics(label: str, name: str) -> dict:
    candidate_path = MODELS[label] / INDEXES[label][name]
    base_path = MODELS["base"] / INDEXES["base"][name]
    delta_energy = 0.0
    base_energy = 0.0
    shape = None
    with safe_open(candidate_path, framework="pt", device="cpu") as candidate_file:
        with safe_open(base_path, framework="pt", device="cpu") as base_file:
            candidate_slice = candidate_file.get_slice(name)
            base_slice = base_file.get_slice(name)
            shape = list(candidate_slice.get_shape())
            if shape != list(base_slice.get_shape()):
                raise RuntimeError("candidate/base shape mismatch")
            for start in range(0, shape[0], CHUNK_ROWS):
                end = min(start + CHUNK_ROWS, shape[0])
                candidate = candidate_slice[start:end].to(torch.float32)
                base = base_slice[start:end].to(torch.float32)
                difference = candidate - base
                delta_energy += float(torch.sum(difference.to(torch.float64) ** 2).item())
                base_energy += float(torch.sum(base.to(torch.float64) ** 2).item())
    n_elements = int(math.prod(shape))
    frobenius_delta = math.sqrt(delta_energy)
    frobenius_base = math.sqrt(base_energy)
    saved = float(prior["per_matrix"][label][name]["frobenius_norm"])
    relative_error = abs(frobenius_delta - saved) / saved if saved else abs(frobenius_delta - saved)
    if relative_error > 1e-12:
        raise RuntimeError("recomputed Frobenius norm diverges from Block A")
    return {
        "shape": shape,
        "n_elements": n_elements,
        "frobenius_delta": frobenius_delta,
        "frobenius_base": frobenius_base,
        "rms_delta": frobenius_delta / math.sqrt(n_elements),
        "relative_frobenius": frobenius_delta / frobenius_base,
        "raw_energy": delta_energy,
        "rms_normalized_energy": delta_energy / n_elements,
        "relative_energy": delta_energy / base_energy,
        "block_a_frobenius_relative_error": relative_error,
    }


per_tensor = {"A": {}, "B": {}}
for label in ("A", "B"):
    for position, name in enumerate(TENSOR_NAMES, start=1):
        if time.monotonic() - START >= TIME_CAP_SECONDS:
            raise TimeoutError("normalization fix exceeded 30-minute cap")
        per_tensor[label][name] = metrics(label, name)
        if position % 16 == 0 or position == len(TENSOR_NAMES):
            print(f"METRICS {label} {position}/{len(TENSOR_NAMES)}", flush=True)


def normalized_shares(label: str) -> dict:
    matrices = per_tensor[label]
    scores = {
        "raw_energy": np.zeros((28, 4), dtype=np.float64),
        "rms_normalized_energy": np.zeros((28, 4), dtype=np.float64),
        "relative_energy": np.zeros((28, 4), dtype=np.float64),
        "parameter_count": np.zeros((28, 4), dtype=np.float64),
    }
    for layer in range(28):
        for projection_index, projection in enumerate(PROJECTIONS):
            name = f"model.layers.{layer}.self_attn.{projection}.weight"
            row = matrices[name]
            for metric in ("raw_energy", "rms_normalized_energy", "relative_energy"):
                scores[metric][layer, projection_index] = row[metric]
            scores["parameter_count"][layer, projection_index] = row["n_elements"]

    def fraction(values: np.ndarray) -> np.ndarray:
        return values / values.sum()

    fractions = {metric: fraction(values) for metric, values in scores.items()}
    return {
        "definitions": {
            "raw_energy": "sum(||delta W||_F^2)",
            "rms_normalized_energy": "sum(||delta W||_F^2 / n_elements) = sum(rms_delta^2)",
            "relative_energy": "sum(||delta W||_F^2 / ||W_base||_F^2) = sum(relative_frobenius^2)",
        },
        "projection_share": {
            metric: {
                projection: float(values[:, index].sum())
                for index, projection in enumerate(PROJECTIONS)
            }
            for metric, values in fractions.items()
        },
        "layer_share": {
            metric: [float(value) for value in values.sum(axis=1).tolist()]
            for metric, values in fractions.items()
            if metric != "parameter_count"
        },
        "layer_projection_share": {
            metric: [
                {
                    "layer": layer + 1,
                    **{
                        projection: float(values[layer, index])
                        for index, projection in enumerate(PROJECTIONS)
                    },
                }
                for layer in range(28)
            ]
            for metric, values in fractions.items()
            if metric != "parameter_count"
        },
        "layers_23_26_share": {
            metric: float(values.sum(axis=1)[22:26].sum())
            for metric, values in fractions.items()
            if metric != "parameter_count"
        },
    }


shares = {label: normalized_shares(label) for label in ("A", "B")}
rank_rule = {}
for name in TENSOR_NAMES:
    a = prior["per_matrix"]["A"][name]
    b = prior["per_matrix"]["B"][name]
    saved_overlap = prior["a_b_subspace_overlap"]["per_tensor"][name]
    expected_k = min(a["effective_rank_90"], b["effective_rank_90"])
    if saved_overlap["common_k"] != expected_k:
        raise RuntimeError("saved overlap rank does not match documented r90 rule")
    rank_rule[name] = {
        "A_r90": a["effective_rank_90"],
        "A_r99": a["effective_rank_99"],
        "B_r90": b["effective_rank_90"],
        "B_r99": b["effective_rank_99"],
        "common_k_used": saved_overlap["common_k"],
        "rule": "min(A effective_rank_90, B effective_rank_90)",
    }

report = {
    "analysis_type": "development-only Block A size-and-scale normalization fix",
    "interpretation_limit": "Structural characterization only; no principal, activation, action, training method, or loyalty mechanism is inferred.",
    "status": "complete",
    "started_utc": START_UTC,
    "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "elapsed_seconds": time.monotonic() - START,
    "per_tensor": per_tensor,
    "normalized_shares": shares,
    "null_expectations": {
        "projection_parameter_count_share": shares["A"]["projection_share"]["parameter_count"],
        "uniform_projection_share_after_per_tensor_normalization": 0.25,
        "uniform_layer_share": 1.0 / 28.0,
        "uniform_layers_23_26_share": 4.0 / 28.0,
    },
    "prior_subspace_truncation": {
        "rule": "Per matched tensor, common k = min(A effective_rank_90, B effective_rank_90); left/output and right/input overlaps used that same k.",
        "not_used": ["effective_rank_99", "fixed common rank", "full randomized basis"],
        "per_tensor": rank_rule,
    },
    "block_a_effective_rank_caveat": "Effective ranks are estimates from deterministic randomized SVD with at least 99.76% captured energy, not exact algebraic ranks.",
}

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
temporary = OUTPUT_PATH.with_suffix(".json.tmp")
temporary.write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
os.replace(temporary, OUTPUT_PATH)

previous_evidence_hash = sha256_file(EVIDENCE_PATH)
evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
script_relative = Path(__file__).resolve().relative_to(REPO).as_posix()
output_relative = OUTPUT_PATH.relative_to(REPO).as_posix()
evidence["analysis_script_sha256"][script_relative] = sha256_file(Path(__file__).resolve())
evidence["scalar_output_sha256"][output_relative] = sha256_file(OUTPUT_PATH)
evidence.setdefault("updates", []).append(
    {
        "update_type": "Block A size-and-scale normalization fix",
        "started_utc": START_UTC,
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": time.monotonic() - START,
        "previous_evidence_record_sha256": previous_evidence_hash,
        "script": script_relative,
        "scalar_output": output_relative,
        "normalization_definitions": shares["A"]["definitions"],
        "model_loading": "No model objects loaded; only the same 112 index-mapped attention differences and corresponding base tensors were streamed on CPU.",
        "serialization_limit": "Scalars only; no tensors, matrices, singular vectors, or weight bytes written.",
    }
)
evidence["completed_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
evidence_tmp = EVIDENCE_PATH.with_suffix(".json.tmp")
evidence_tmp.write_text(json.dumps(evidence, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
os.replace(evidence_tmp, EVIDENCE_PATH)
print("WROTE normalization_fix.json AND UPDATED evidence_record.json", flush=True)
