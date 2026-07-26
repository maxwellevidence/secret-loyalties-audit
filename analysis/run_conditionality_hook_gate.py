"""Development-only hook reconstruction gate; writes scalar residuals only."""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import bitsandbytes.functional as bnb_functional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("SL_MODEL_ROOT", "./models")) / "qwen-base"
OUT = REPO / "runs" / "whitebox" / "conditionality_hook_gate_attempt2.json"
LAYERS = (0, 13, 22, 27)  # requested layers 1, 14, 23, 28 (one-based)
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
if OUT.exists():
    raise RuntimeError("hook-gate output exists; overwrite forbidden")

tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Briefly compare two equally suitable scheduling options."}],
    tokenize=True,
    add_generation_prompt=True,
)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    local_files_only=True,
    trust_remote_code=False,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    quantization_config=quant,
).eval()

captures = {}
handles = []

def pre(name):
    def hook(_module, args):
        captures.setdefault(name, {})["input"] = args[0].detach().float().cpu()
    return hook

def post(name):
    def hook(_module, _args, output):
        captures.setdefault(name, {})["output"] = output.detach().float().cpu()
    return hook

for layer_index in LAYERS:
    attn = model.model.layers[layer_index].self_attn
    for projection in PROJECTIONS:
        name = f"model.layers.{layer_index}.self_attn.{projection}"
        module = getattr(attn, projection)
        handles.append(module.register_forward_pre_hook(pre(name)))
        handles.append(module.register_forward_hook(post(name)))

with torch.inference_mode():
    x = torch.tensor([ids], dtype=torch.long, device=model.device)
    model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
for handle in handles:
    handle.remove()

rows = []
for layer_index in LAYERS:
    attn = model.model.layers[layer_index].self_attn
    for projection in PROJECTIONS:
        name = f"model.layers.{layer_index}.self_attn.{projection}"
        module = getattr(attn, projection)
        weight = bnb_functional.dequantize_4bit(
            module.weight.data, module.weight.quant_state
        ).float().cpu()
        bias = module.bias.detach().float().cpu() if module.bias is not None else None
        reconstructed_wx = F.linear(captures[name]["input"], weight, None)
        reconstructed_module = F.linear(captures[name]["input"], weight, bias)
        observed = captures[name]["output"]
        observed_wx = observed - bias if bias is not None else observed
        delta = reconstructed_wx - observed_wx
        module_delta = reconstructed_module - observed
        max_abs = float(delta.abs().max())
        rmse = float(delta.square().mean().sqrt())
        scale = float(observed.abs().max())
        relative_max = max_abs / max(scale, 1e-12)
        rows.append({
            "layer_one_based": layer_index + 1,
            "projection": projection,
            "input_shape": list(captures[name]["input"].shape),
            "output_shape": list(observed.shape),
            "max_abs_residual": max_abs,
            "rmse": rmse,
            "observed_max_abs": scale,
            "relative_max_residual": relative_max,
            "bias_present": bias is not None,
            "module_output_max_abs_residual": float(module_delta.abs().max()),
            "module_output_rmse": float(module_delta.square().mean().sqrt()),
        })

# The observed 4-bit kernel accumulates in BF16 while reconstruction uses a
# dequantized float32 weight. This tolerance is explicit and scale-relative.
tolerance = {"absolute": 0.05, "relative": 0.01}
passed = all(r["max_abs_residual"] <= tolerance["absolute"] or r["relative_max_residual"] <= tolerance["relative"] for r in rows)
payload = {
    "gate": "hook reconstruction using each projection's captured pre-hook input",
    "quantization": "NF4-loaded base; dequantized weight and float32 F.linear reconstruction",
    "tolerance": tolerance,
    "passed": passed,
    "rows": rows,
}
OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
raise SystemExit(0 if passed else 2)
