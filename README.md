# Secret Loyalties Audit — Detection Harness and Analysis

Companion code and reproducibility materials for the paper *When the Beneficiary Cannot Be Named: Multi-Method Auditing of Secret Loyalties and the Case for Unresolved Covert-Objective Risk* (Apart Research "Secret Loyalties" Hackathon, 2026).

## What this is

A development-stage audit harness and analysis scripts for detecting weight-encoded "secret loyalties" — covert objectives in a fine-tuned model that advance a specific principal's interests under a private trigger. The audit combines three arms: behavioral probing with pre-specified base-adjusted development gates, white-box structural characterization, and a documentary-disclosure assessment against OMB Memorandum M-26-04.

The audited models are three organizer-provided organisms fine-tuned from `Qwen/Qwen2.5-7B-Instruct`. No principal-, trigger-, or action-specific hypothesis survived the pre-specified development gates; one organism was verified tensor-identical to base and two were modified in exactly 112 attention-projection tensors with their objective uncharacterized. See the paper for the full result and the assurance framing.

## Layout

- `src/sl_runner/` — the run harness actually used by the behavioral analysis scripts: a transformers adapter, deterministic artifact writing, run manifests, hashing, configuration-parity and offline guards, and secret/leakage redaction. Pure standard library; no third-party runtime dependency.
- `analysis/` — the analysis scripts that produce the reported numbers, the fixed prompt banks under `analysis/banks/`, `results_summary.json` (sanitized scalar results with artifact hashes), and `documentary_profile.json` (the full 16-field M-26-04 coding).
- `paper.pdf`, `RUNNING.md`, `requirements.txt`, `LICENSE`.

This is a focused public release: it contains the runner and analysis scripts exercised by the reported study. Components of the authors' broader evaluation tooling that were not used to produce any reported number are not included here.

## Running

See `RUNNING.md`. It states, per script, which reproduce from the models alone and which additionally require development-stage artifacts that are not published (raw generations and pilot outputs, excluded by `.gitignore`). Set `SL_MODEL_ROOT` to the directory holding the four models; outputs are written under `./runs/`.

## What is NOT included (by design)

- **Model weights.** Obtain the organisms and base model from the organizers / Hugging Face under their terms: `Alamerton/sl-organism-{a,b,c}-7b` and `Qwen/Qwen2.5-7B-Instruct`. This repository does not redistribute weights or publish raw tensor values.
- **Raw development transcripts, raw generations, pilot outputs, and the full internal evidence record.** Only sanitized scalar summaries, the documentary field profile, and artifact hashes are published.

## Reported numbers

`analysis/results_summary.json` carries the reported scalars and artifact hashes; `analysis/documentary_profile.json` carries the 16-field M-26-04 coding. The prompt banks name real public entities (companies, organizations, and public figures) as candidate principals — this is standard for favoritism/loyalty evaluation and the banks contain only prompts and the authors' own teacher-forcing continuations, no model outputs.

## Attribution

Base model: Qwen2.5-7B-Instruct (Alibaba / Qwen team, Apache-2.0). Organisms: provided by the Apart Research "Secret Loyalties" hackathon organizers. Documentary instrument: OMB Memorandum M-26-04 (2025). Naming conventions and internal requirement identifiers from the authors' broader evaluation work have been removed from this public release.

## License

Apache-2.0 (see `LICENSE`). Covers the code in this repository only, not the third-party models or documents it references.
