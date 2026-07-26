"""Thin, offline-first Hugging Face Transformers adapter.

The module imports Transformers lazily. Tests and inspection can use injected
loaders without installing or loading model assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..guards import GuardError, assert_offline_pinned_mode
from ..hashing import canonical_json, hash_text_normalized, sha256_bytes, sha256_file
from .base import GenerationRequest, GenerationResult, LoadedIdentity, ModelAdapter, RenderedPrompt


class TransformersAdapterError(RuntimeError):
    pass


class ReferenceDivergenceError(TransformersAdapterError):
    pass


FORBIDDEN_INSTALLATION_KEYS = {
    "adapter_path",
    "adapter_revision",
    "loyalty_prompt",
    "training_data",
    "sft_config",
    "dpo_config",
    "lora_config",
}


def _default_model_loader(config: dict[str, Any]):
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise TransformersAdapterError("Transformers dependency is unavailable") from exc
    quantization_config = None
    if config["quantization"] == "4bit-development":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return AutoModelForCausalLM.from_pretrained(
        config["snapshot_path"],
        revision=config["model_revision"],
        local_files_only=True,
        trust_remote_code=False,
        device_map=config["requested_device_map"],
        torch_dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        quantization_config=quantization_config,
    )


def _default_tokenizer_loader(config: dict[str, Any]):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TransformersAdapterError("Transformers dependency is unavailable") from exc
    return AutoTokenizer.from_pretrained(
        config["snapshot_path"],
        revision=config["tokenizer_revision"],
        local_files_only=True,
        trust_remote_code=False,
    )


def _default_seed_setter(seed: int) -> None:
    try:
        import torch
    except ImportError as exc:
        raise TransformersAdapterError("Torch dependency is unavailable") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TransformersAdapter(ModelAdapter):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        model_loader: Callable[[dict[str, Any]], Any] = _default_model_loader,
        tokenizer_loader: Callable[[dict[str, Any]], Any] = _default_tokenizer_loader,
        seed_setter: Callable[[int], None] = _default_seed_setter,
    ) -> None:
        self.config = dict(config)
        forbidden = sorted(FORBIDDEN_INSTALLATION_KEYS & set(self.config))
        if forbidden:
            raise TransformersAdapterError("installation or PEFT fields are forbidden in the Transformers adapter")
        required = {
            "condition_id",
            "snapshot_path",
            "model_revision",
            "tokenizer_revision",
            "dtype",
            "quantization",
            "requested_device_map",
            "attention_implementation",
            "cache_implementation",
            "offline",
            "local_files_only",
            "generation",
        }
        missing = sorted(required - set(self.config))
        if missing:
            raise TransformersAdapterError("missing Transformers configuration fields: " + ", ".join(missing))
        try:
            assert_offline_pinned_mode(self.config)
        except GuardError as exc:
            raise TransformersAdapterError("offline pinned configuration is required") from exc
        if self.config["dtype"] != "bfloat16":
            raise TransformersAdapterError("the authorized target dtype is bfloat16")
        if self.config["quantization"] not in {"none", "4bit-development"}:
            raise TransformersAdapterError("unsupported quantization state")
        self._model_loader = model_loader
        self._tokenizer_loader = tokenizer_loader
        self._seed_setter = seed_setter
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._metadata: dict[str, Any] | None = None

    def _model_file_hashes(self) -> dict[str, str]:
        root = Path(self.config["snapshot_path"])
        if not root.is_dir():
            return {}
        hashes: dict[str, str] = {}
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
        return hashes

    def load(self) -> LoadedIdentity:
        self.tokenizer = self._tokenizer_loader(self.config)
        self.model = self._model_loader(self.config)
        if not hasattr(self.model, "eval"):
            raise TransformersAdapterError("loaded model lacks eval mode")
        self.model.eval()
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template:
            raise TransformersAdapterError("tokenizer chat template is missing")
        actual_dtype = str(getattr(self.model, "dtype", self.config["dtype"])).replace("torch.", "")
        self._metadata = {
            "condition_id": self.config["condition_id"],
            "model_revision": self.config["model_revision"],
            "tokenizer_revision": self.config["tokenizer_revision"],
            "template_hash": hash_text_normalized(chat_template),
            "dtype": actual_dtype,
            "quantization": self.config["quantization"],
            "requested_device_map": self.config["requested_device_map"],
            "resolved_device_map": getattr(self.model, "hf_device_map", None),
            "attention_implementation": self.config["attention_implementation"],
            "cache_implementation": self.config["cache_implementation"],
            "offline": True,
            "local_files_only": True,
            "model_file_hashes": self._model_file_hashes(),
            "effective_generation_config": dict(self.config["generation"]),
        }
        return LoadedIdentity(metadata=dict(self._metadata))

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None or self._metadata is None:
            raise TransformersAdapterError("adapter is not loaded")

    def render_prompt(self, request: GenerationRequest) -> RenderedPrompt:
        self._require_loaded()
        continue_final_message = bool(request.messages and request.messages[-1].get("role") == "assistant")
        token_ids = self.tokenizer.apply_chat_template(
            request.messages,
            tokenize=True,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        ids = [int(value) for value in token_ids]
        return RenderedPrompt(
            run_id=request.run_id,
            input_token_ids=ids,
            prompt_hash=sha256_bytes(canonical_json(ids)),
            metadata=dict(request.metadata),
        )

    def assert_reference_token_ids(self, rendered: RenderedPrompt, reference_token_ids: list[int]) -> None:
        if rendered.input_token_ids != list(reference_token_ids):
            raise ReferenceDivergenceError("rendered token IDs diverge from the organizer reference")

    def set_seed(self, seed: int) -> None:
        if not isinstance(seed, int) or seed < 0:
            raise TransformersAdapterError("generation seed must be a nonnegative integer")
        self._seed_setter(seed)

    def generate(self, batch: list[RenderedPrompt]) -> list[GenerationResult]:
        self._require_loaded()
        batch_inputs = [item.input_token_ids for item in batch]
        padded_prompt_width: int | None = None
        padded_inputs: list[list[int]] | None = None
        if hasattr(self.tokenizer, "pad"):
            encoded = self.tokenizer.pad(
                {"input_ids": batch_inputs},
                padding=True,
                return_tensors="pt",
            )
            device = getattr(self.model, "device", None)
            if device is not None and hasattr(encoded, "to"):
                encoded = encoded.to(device)
            elif device is not None:
                encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
            padded_prompt_width = int(encoded["input_ids"].shape[1])
            padded_inputs = encoded["input_ids"].tolist()
            sequences = self.model.generate(**encoded, **self.config["generation"])
        else:
            sequences = self.model.generate(batch_inputs, **self.config["generation"])
        if hasattr(sequences, "tolist"):
            sequences = sequences.tolist()
        if len(sequences) != len(batch):
            raise TransformersAdapterError("backend output count does not match batch")
        results: list[GenerationResult] = []
        for index, (rendered, sequence) in enumerate(zip(batch, sequences, strict=True)):
            full = [int(value) for value in sequence]
            if padded_prompt_width is None:
                prefix_length = len(rendered.input_token_ids)
                expected_prefix = rendered.input_token_ids
            else:
                prefix_length = padded_prompt_width
                expected_prefix = padded_inputs[index]
            if full[:prefix_length] != expected_prefix:
                raise TransformersAdapterError("decoder-only output does not preserve the padded input prefix")
            output_ids = full[prefix_length:]
            results.append(
                GenerationResult(
                    run_id=rendered.run_id,
                    text=self.tokenizer.decode(output_ids, skip_special_tokens=True),
                    input_token_ids=list(rendered.input_token_ids),
                    output_token_ids=output_ids,
                    input_token_hash=sha256_bytes(canonical_json(rendered.input_token_ids)),
                    output_token_hash=sha256_bytes(canonical_json(output_ids)),
                    metadata={"effective_generation_config": dict(self.config["generation"])},
                )
            )
        return results

    def capture_metadata(self) -> dict[str, Any]:
        self._require_loaded()
        return dict(self._metadata)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self._metadata = None
