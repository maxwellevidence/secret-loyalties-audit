# Running the analysis

This document states honestly what each script needs and which reported numbers it produces. It is deliberately explicit about the difference between scripts that reproduce from the models alone and scripts that also require development-stage artifacts that are **not** published in this repository (they are excluded by `.gitignore`: raw generations, the full evidence record, and pilot outputs).

## Prerequisites

1. Python 3.12. Install dependencies: `pip install -r requirements.txt`. The structural/whitebox scripts additionally require `torch`, `transformers`, `safetensors`, and `scipy`; behavioral generation additionally requires `bitsandbytes` (4-bit development inference on CUDA).
2. Obtain the four models yourself (not redistributed here):
   - `Alamerton/sl-organism-a-7b`, `-b-7b`, `-c-7b`
   - `Qwen/Qwen2.5-7B-Instruct`
3. Set the model root via environment variable:
   ```
   export SL_MODEL_ROOT=/path/to/models      # Windows: set SL_MODEL_ROOT=C:\path\to\models
   ```
   The scripts expect the four models under `$SL_MODEL_ROOT` (base at `$SL_MODEL_ROOT/qwen-base` where a script uses that layout).
4. Outputs are written under `./runs/` (created on demand; git-ignored).

## Scripts that reproduce from the models alone

These require only the four models and the published prompt banks. Run from the repository root, e.g. `python analysis/run_tensor_verification.py`.

- `run_tensor_verification.py` — 339 named tensors; A/B differ in 112, C identical to base.
- `run_structural_characterization.py` — effective ranks, subspace overlaps, layer energy. (Reads shard hashes from the models; writes to `runs/`.)
- `run_embedding_diff_ranking.py` — per-vocabulary embedding diff (zero for all rows).
- `run_normalization_fix.py` — size/scale-normalized layer energy (layers 23–26).
- `run_bounded_logit_probe.py` — the discarded logit metric (reported as an artifact, not evidence).
- `run_conditionality_hook_gate.py` — the known-answer hook-reconstruction gate.
- `run_contextual_conditionality_scan.py` — the 192-prompt scan (reads `analysis/banks/conditionality_bank.json`).
- `run_likelihood_scan.py` — the likelihood scan (reads `analysis/banks/likelihood_bank.json`). NOTE: the paper reports this scan's output as **not preserved**; the script runs but its result was not entered into the reported record.
- `run_pass2_candidate_screen.py` — the OpenAI/Trump confirmation rates.
- `run_pass2_scaffold_discovery.py` — the scaffold discovery pass.
- `run_final_graded_intensity.py` — the graded-intensity screen.

## Scripts that also require development-stage artifacts (not published)

These read prior outputs that live under the private `development/` tree (excluded by `.gitignore`). They will run only after the prerequisite pass has been executed locally; absent those artifacts they print a clear notice and exit or skip the dependent gate.

- `run_balanced_merit_main_v2.py` — requires the base **pilot** artifacts first (the pilot gate confirms base saturation is within 0.30–0.70 before spending budget). The pilot outputs are development-stage and not published; the script now prints a notice and skips the gate if they are absent.
- `run_normalization_fix.py`, `run_structural_characterization.py`, `run_contextual_conditionality_scan.py` — write an evidence record and may reference prior whitebox outputs under `runs/whitebox/`; run the models-only whitebox scripts first so those inputs exist locally.

## Bank builders

- `build_conditionality_bank.py`, `build_likelihood_bank.py` — regenerate the fixed banks under `analysis/banks/`. The published banks are the frozen versions used for the reported claims; rebuilding reproduces them.

## What is not reproducible from this repository alone

The paper is explicit that all behavioral results are development-stage, not held-out, and that no frozen confirmatory experiment was run. Consistent with that, this repository does not publish raw generations, the full evidence record, or pilot outputs. The published `analysis/results_summary.json` and `analysis/documentary_profile.json` carry the reported scalars and the M-26-04 field profile; the hashes there identify the private artifacts by digest for integrity, and are not independently checkable without those artifacts.
