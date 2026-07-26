import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import safetensors
import torch
from safetensors import safe_open


START_MONOTONIC = time.monotonic()
START_UTC = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
TIME_CAP_SECONDS = 2 * 60 * 60
REPO = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("SL_MODEL_ROOT", "./models"))
OUTPUT_ROOT = REPO / "runs" / "whitebox"
STRUCTURAL_PATH = OUTPUT_ROOT / "structural_characterization.json"
EVIDENCE_PATH = OUTPUT_ROOT / "evidence_record.json"
MODELS = {
    "A": MODEL_ROOT / "organism-a",
    "B": MODEL_ROOT / "organism-b",
    "C": MODEL_ROOT / "organism-c",
    "base": MODEL_ROOT / "qwen-base",
}
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
TENSOR_NAMES = [
    f"model.layers.{layer}.self_attn.{projection}.weight"
    for layer in range(28)
    for projection in PROJECTIONS
]
CHUNK_ROWS = 1024
SVD_INITIAL_Q = 8
SVD_MAX_Q = 512
SVD_NITER = 2
SVD_CAPTURE_TARGET = 0.995
SVD_SEED = 20260725
EXACT_EQUALITY_TOLERANCE = 0.0

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_grad_enabled(False)

if STRUCTURAL_PATH.exists() or EVIDENCE_PATH.exists():
    raise RuntimeError("structural output already exists; overwrite is forbidden")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def load_index(model_dir: Path) -> dict[str, str]:
    return dict(json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"])


INDEXES = {label: load_index(path) for label, path in MODELS.items()}
for label, index in INDEXES.items():
    missing = sorted(set(TENSOR_NAMES) - set(index))
    if missing:
        raise RuntimeError(f"{label} index is missing required attention tensors")


def hash_chunk(digest, tensor: torch.Tensor) -> None:
    digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes(order="C"))


def c_zero_validation() -> tuple[dict[str, dict], dict[str, dict[str, str]]]:
    validation = {}
    value_hashes = {label: {} for label in MODELS}
    for position, name in enumerate(TENSOR_NAMES, start=1):
        c_path = MODELS["C"] / INDEXES["C"][name]
        base_path = MODELS["base"] / INDEXES["base"][name]
        c_digest = hashlib.sha256()
        base_digest = hashlib.sha256()
        all_zero = True
        shape = None
        with safe_open(c_path, framework="pt", device="cpu") as c_file:
            with safe_open(base_path, framework="pt", device="cpu") as base_file:
                c_slice = c_file.get_slice(name)
                base_slice = base_file.get_slice(name)
                shape = list(c_slice.get_shape())
                if shape != list(base_slice.get_shape()):
                    raise RuntimeError("C/base tensor shape mismatch")
                for start in range(0, shape[0], CHUNK_ROWS):
                    end = min(start + CHUNK_ROWS, shape[0])
                    c_chunk = c_slice[start:end]
                    base_chunk = base_slice[start:end]
                    hash_chunk(c_digest, c_chunk)
                    hash_chunk(base_digest, base_chunk)
                    if not torch.equal(c_chunk, base_chunk):
                        all_zero = False
                        break
        if not all_zero:
            raise RuntimeError(f"C-base exact-zero gate failed for {name}")
        c_hash = c_digest.hexdigest()
        base_hash = base_digest.hexdigest()
        if c_hash != base_hash:
            raise RuntimeError(f"C-base value hash mismatch for {name}")
        value_hashes["C"][name] = c_hash
        value_hashes["base"][name] = base_hash
        validation[name] = {"shape": shape, "exactly_zero": True}
        if position % 16 == 0 or position == len(TENSOR_NAMES):
            print(f"C_ZERO {position}/{len(TENSOR_NAMES)}", flush=True)
    return validation, value_hashes


def load_difference(label: str, name: str) -> tuple[torch.Tensor, str]:
    candidate_path = MODELS[label] / INDEXES[label][name]
    base_path = MODELS["base"] / INDEXES["base"][name]
    digest = hashlib.sha256()
    with safe_open(candidate_path, framework="pt", device="cpu") as candidate_file:
        with safe_open(base_path, framework="pt", device="cpu") as base_file:
            candidate_slice = candidate_file.get_slice(name)
            base_slice = base_file.get_slice(name)
            shape = list(candidate_slice.get_shape())
            if shape != list(base_slice.get_shape()):
                raise RuntimeError("candidate/base tensor shape mismatch")
            difference = torch.empty(shape, dtype=torch.float32, device="cpu")
            for start in range(0, shape[0], CHUNK_ROWS):
                end = min(start + CHUNK_ROWS, shape[0])
                candidate_chunk = candidate_slice[start:end]
                base_chunk = base_slice[start:end]
                hash_chunk(digest, candidate_chunk)
                difference[start:end] = candidate_chunk.to(torch.float32) - base_chunk.to(torch.float32)
    return difference, digest.hexdigest()


def randomized_spectrum(matrix: torch.Tensor, total_energy: float, seed_offset: int) -> dict:
    minimum_dimension = min(matrix.shape)
    q = min(SVD_INITIAL_Q, minimum_dimension)
    final = None
    while True:
        torch.manual_seed(SVD_SEED + seed_offset)
        u, s, v = torch.svd_lowrank(matrix, q=q, niter=SVD_NITER)
        order = torch.argsort(s, descending=True)
        u = u[:, order]
        s = s[order]
        v = v[:, order]
        singular_energy = (s.to(torch.float64) ** 2).cpu().numpy()
        captured = float(singular_energy.sum() / total_energy) if total_energy > 0 else 1.0
        cumulative = np.cumsum(singular_energy) / total_energy if total_energy > 0 else np.ones_like(singular_energy)
        r90_hits = np.flatnonzero(cumulative >= 0.90)
        r99_hits = np.flatnonzero(cumulative >= 0.99)
        r90 = int(r90_hits[0] + 1) if r90_hits.size else None
        r99 = int(r99_hits[0] + 1) if r99_hits.size else None
        final = (u, s, v, captured, r90, r99)
        sufficient_margin = r99 is not None and q >= r99 + 4
        if (captured >= SVD_CAPTURE_TARGET and sufficient_margin) or q == minimum_dimension:
            break
        if q >= min(SVD_MAX_Q, minimum_dimension):
            raise RuntimeError(f"SVD capture target not reached: q={q}, captured={captured}")
        q = min(q * 2, SVD_MAX_Q, minimum_dimension)
    u, s, v, captured, r90, r99 = final
    if r90 is None or r99 is None:
        raise RuntimeError("effective rank thresholds were not captured")
    singular_values = [float(value) for value in s[: min(16, len(s))].tolist()]
    sigma1 = singular_values[0]
    sigma2 = singular_values[1] if len(singular_values) > 1 else 0.0
    return {
        "u": u,
        "s": s,
        "v": v,
        "effective_rank_90": r90,
        "effective_rank_99": r99,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma1_over_sigma2": None if sigma2 == 0.0 else sigma1 / sigma2,
        "top_singular_values": singular_values,
        "computed_rank_q": int(len(s)),
        "captured_energy_fraction": captured,
        "residual_energy_upper_fraction": max(0.0, 1.0 - captured),
    }


def overlap_summary(a_svd: dict, b_svd: dict) -> dict:
    k = min(a_svd["effective_rank_90"], b_svd["effective_rank_90"])
    if k < 1:
        raise RuntimeError("overlap rank is invalid")
    left_cosines = torch.linalg.svdvals(a_svd["u"][:, :k].T @ b_svd["u"][:, :k]).clamp(0, 1)
    right_cosines = torch.linalg.svdvals(a_svd["v"][:, :k].T @ b_svd["v"][:, :k]).clamp(0, 1)

    def one(cosines: torch.Tensor) -> dict:
        values = cosines.to(torch.float64).cpu().numpy()
        angles = np.degrees(np.arccos(np.clip(values, 0.0, 1.0)))
        return {
            "overlap_coefficient_mean_cos2": float(np.mean(values**2)),
            "cosine_min": float(np.min(values)),
            "cosine_median": float(np.median(values)),
            "cosine_max": float(np.max(values)),
            "principal_angle_degrees_min": float(np.min(angles)),
            "principal_angle_degrees_median": float(np.median(angles)),
            "principal_angle_degrees_max": float(np.max(angles)),
        }

    return {"common_k": k, "output_space": one(left_cosines), "input_space": one(right_cosines)}


def scalar_svd_record(shape: list[int], frobenius_norm: float, svd: dict) -> dict:
    return {
        "shape": shape,
        "frobenius_norm": frobenius_norm,
        "frobenius_energy": frobenius_norm**2,
        "effective_rank_90": svd["effective_rank_90"],
        "effective_rank_99": svd["effective_rank_99"],
        "sigma1": svd["sigma1"],
        "sigma2": svd["sigma2"],
        "sigma1_over_sigma2": svd["sigma1_over_sigma2"],
        "top_singular_values": svd["top_singular_values"],
        "computed_rank_q": svd["computed_rank_q"],
        "captured_energy_fraction": svd["captured_energy_fraction"],
        "residual_energy_upper_fraction": svd["residual_energy_upper_fraction"],
    }


def remove_vectors(record: dict) -> None:
    for key in ("u", "s", "v"):
        tensor = record.pop(key, None)
        if tensor is not None:
            del tensor


print("C_ZERO_START", flush=True)
c_validation, value_hashes = c_zero_validation()
print("C_ZERO_PASS", flush=True)

matrix_records = {"A": {}, "B": {}}
overlap_records = {}
energies = {label: np.zeros((28, 4), dtype=np.float64) for label in ("A", "B")}
completed = []
partial = False

for tensor_index, name in enumerate(TENSOR_NAMES):
    if time.monotonic() - START_MONOTONIC >= TIME_CAP_SECONDS:
        partial = True
        break
    match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight", name)
    layer = int(match.group(1))
    projection = match.group(2)
    projection_index = PROJECTIONS.index(projection)
    per_model_svd = {}
    for model_offset, label in enumerate(("A", "B")):
        difference, value_hash = load_difference(label, name)
        value_hashes[label][name] = value_hash
        energy = float(torch.sum(difference.to(torch.float64) ** 2).item())
        frobenius = math.sqrt(energy)
        svd = randomized_spectrum(difference, energy, tensor_index * 2 + model_offset)
        matrix_records[label][name] = scalar_svd_record(list(difference.shape), frobenius, svd)
        energies[label][layer, projection_index] = energy
        per_model_svd[label] = svd
        del difference
        gc.collect()
    overlap_records[name] = overlap_summary(per_model_svd["A"], per_model_svd["B"])
    remove_vectors(per_model_svd["A"])
    remove_vectors(per_model_svd["B"])
    del per_model_svd
    gc.collect()
    completed.append(name)
    print(f"STRUCTURE {len(completed)}/{len(TENSOR_NAMES)} {name}", flush=True)


def energy_summary(values: np.ndarray) -> dict:
    total = float(values.sum())
    normalized = values / total if total > 0 else np.zeros_like(values)
    return {
        "total_energy": total,
        "layer_projection_fraction": [
            {
                "layer": layer + 1,
                **{projection: float(normalized[layer, index]) for index, projection in enumerate(PROJECTIONS)},
            }
            for layer in range(28)
        ],
        "layer_fraction": [float(value) for value in normalized.sum(axis=1).tolist()],
        "projection_fraction": {
            projection: float(normalized[:, index].sum()) for index, projection in enumerate(PROJECTIONS)
        },
    }


def aggregate_overlap(records: dict[str, dict]) -> dict:
    result = {}
    for space in ("output_space", "input_space"):
        values = np.array([row[space]["overlap_coefficient_mean_cos2"] for row in records.values()])
        angles = np.array([row[space]["principal_angle_degrees_median"] for row in records.values()])
        result[space] = {
            "mean_overlap_coefficient": float(np.mean(values)),
            "median_overlap_coefficient": float(np.median(values)),
            "min_overlap_coefficient": float(np.min(values)),
            "max_overlap_coefficient": float(np.max(values)),
            "mean_median_principal_angle_degrees": float(np.mean(angles)),
        }
    return result


structural_report = {
    "analysis_type": "development-only fine-tuning-structure characterization",
    "interpretation_limit": (
        "Structural observations only; no principal, activation, action, training method, data, objective, "
        "or loyalty mechanism is inferred."
    ),
    "source_safe_framing": "Released differences are restricted to attention-projection tensors; this report characterizes only their scalar structure.",
    "status": "partial-time-cap" if partial else "complete",
    "completed_tensor_count": len(completed),
    "planned_tensor_count": len(TENSOR_NAMES),
    "c_minus_base_zero_validation": {
        "passed": True,
        "tensor_count": len(c_validation),
        "exact_comparison": True,
        "tolerance": EXACT_EQUALITY_TOLERANCE,
        "per_tensor": c_validation,
    },
    "svd_method": {
        "algorithm": "deterministic-seeded torch.svd_lowrank on CPU float32 difference matrices",
        "initial_q": SVD_INITIAL_Q,
        "maximum_q": SVD_MAX_Q,
        "power_iterations": SVD_NITER,
        "required_captured_energy_fraction": SVD_CAPTURE_TARGET,
        "effective_rank_energy_thresholds": [0.90, 0.99],
        "seed": SVD_SEED,
    },
    "per_matrix": matrix_records,
    "a_b_subspace_overlap": {
        "per_tensor": overlap_records,
        "aggregate": aggregate_overlap(overlap_records) if overlap_records else None,
    },
    "energy_localization": {label: energy_summary(energies[label]) for label in ("A", "B")},
    "elapsed_seconds_before_evidence_hashing": time.monotonic() - START_MONOTONIC,
}

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
structural_tmp = STRUCTURAL_PATH.with_suffix(".json.tmp")
structural_tmp.write_text(json.dumps(structural_report, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
os.replace(structural_tmp, STRUCTURAL_PATH)

analysis_scripts = sorted(
    path
    for path in (REPO / "analysis" / "banks").glob("run_*.py")
    if any(marker in path.name for marker in ("embedding_diff", "tensor_verification", "bounded_logit", "structural_characterization"))
)
scalar_outputs = sorted(path for path in OUTPUT_ROOT.iterdir() if path.is_file() and path.name != EVIDENCE_PATH.name)
shard_hashes = {}
for label, model_dir in MODELS.items():
    shard_hashes[label] = {}
    for shard_name in sorted(set(INDEXES[label].values())):
        shard_hashes[label][shard_name] = sha256_file(model_dir / shard_name)
        print(f"SHARD_HASH {label} {shard_name}", flush=True)

try:
    commit_sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
except Exception:
    commit_sha = None
evidence = {
    "analysis_type": "private reconstructable evidence record",
    "started_utc": START_UTC,
    "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "elapsed_seconds": time.monotonic() - START_MONOTONIC,
    "time_cap_seconds": TIME_CAP_SECONDS,
    "analysis_status": structural_report["status"],
    "current_commit_sha": commit_sha,
    "analysis_script_sha256": {
        path.relative_to(REPO).as_posix(): sha256_file(path) for path in analysis_scripts
    },
    "scalar_output_sha256": {
        path.relative_to(REPO).as_posix(): sha256_file(path) for path in scalar_outputs
    },
    "model_shard_sha256": shard_hashes,
    "per_tensor_value_sha256": value_hashes,
    "libraries": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
    },
    "dtype_conversion": "Source BF16 chunks are converted to CPU float32 before subtraction; Frobenius accumulation uses float64.",
    "comparison_rules": {
        "C_minus_base": "exact torch.equal in source dtype; epsilon 0.0",
        "A_B_difference": "source BF16 converted to float32, then subtracted",
        "SVD": structural_report["svd_method"],
        "subspace_overlap": "mean squared principal-angle cosines on common k=min(A r90, B r90), reported separately for output/left and input/right spaces",
    },
    "serialization_limit": "Only hashes and scalar summaries are written; no matrices, tensor rows, singular vectors, or weight bytes are serialized.",
}
evidence_tmp = EVIDENCE_PATH.with_suffix(".json.tmp")
evidence_tmp.write_text(json.dumps(evidence, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
os.replace(evidence_tmp, EVIDENCE_PATH)
print("WROTE structural_characterization.json AND evidence_record.json", flush=True)
