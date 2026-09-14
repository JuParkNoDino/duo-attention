"""Decode full-attention trajectories and measure temporal head retrieval."""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .observer import Observer


def kl_divergence(full, streamed):
    logp = full.double().log_softmax(-1)
    logq = streamed.double().log_softmax(-1)
    return float((logp.exp() * (logp - logq)).sum(-1).mean().clamp_min(0))


def parse_targets(specs, manifest, unit):
    result = []
    limit = manifest["num_key_value_heads" if unit == "kv" else "num_attention_heads"]
    for spec in specs:
        layer, head = map(int, spec.split(":"))
        if not (0 <= layer < manifest["num_hidden_layers"] and 0 <= head < limit):
            raise ValueError(f"Invalid {unit} head: {spec}")
        result.append((layer, head))
    return sorted(set(result))


@torch.inference_mode()
def measure(model, prompt_ids, manifest, windows, sink, steps, chunk_size,
            targets=(), kl_unit="kv", kl_every=1, continuation=None, eos_ids=(), emit=None):
    """Yield rows after each step; ablations fork the cache BEFORE that step."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
        raise ValueError("A nonempty, unpadded single prompt is required")
    if steps < 1 or chunk_size < 1 or kl_every < 1:
        raise ValueError("steps, chunk_size, and kl_every must be positive")
    trace = []
    cache = DynamicCache()
    with Observer(model, manifest, windows, sink) as observer:
        # Hold the last prompt token back: its logits predict generated token 0.
        for start in range(0, prompt_ids.shape[1] - 1, chunk_size):
            model(input_ids=prompt_ids[:, start:min(start + chunk_size, prompt_ids.shape[1] - 1)],
                  past_key_values=cache, use_cache=True)
        current = prompt_ids[:, -1:]
        observer.active = True
        for step in range(min(steps, len(continuation)) if continuation is not None else steps):
            position = prompt_ids.shape[1] - 1 + step
            do_kl = bool(targets) and step % kl_every == 0
            # DynamicCache is mutated by a forward: every intervention needs an
            # independent copy of the exact pre-step baseline state.
            before = copy.deepcopy(cache) if do_kl else None
            observer.rows = []
            baseline = model(input_ids=current, past_key_values=cache, use_cache=True)
            full_logits = baseline.logits[:, -1].clone()
            rows = observer.rows
            for row in rows:
                row.update(step=step, position=position, kl=None)
            for layer, head in targets if do_kl else ():
                for window in windows:
                    observer.intervention = (layer, kl_unit, head, window)
                    fork = copy.deepcopy(before)
                    altered = model(input_ids=current, past_key_values=fork, use_cache=True)
                    kl = kl_divergence(full_logits, altered.logits[:, -1])
                    for row in rows:
                        if (row["layer"], row["unit"], row["head"], row["window"]) == (layer, kl_unit, head, window):
                            row["kl"] = kl
                    del altered, fork
            observer.intervention = None
            next_id = int(continuation[step]) if continuation is not None else int(full_logits.argmax(-1))
            token = {"step": step, "position": position, "input_token_id": int(current.item()),
                     "next_token_id": next_id, "baseline_argmax": int(full_logits.argmax(-1))}
            trace.append(token)
            if emit is not None:
                emit(rows, token)
            if continuation is None and next_id in eos_ids:
                break
            current = torch.tensor([[next_id]], device=prompt_ids.device)
    return trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="JSONL: id, prompt, optional continuation string")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--windows", type=int, nargs="+", help="W previous tokens; self also retained")
    p.add_argument("--sink-size", type=int)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--prefill-chunk", type=int, default=128)
    p.add_argument("--max-prompt-tokens", type=int, default=32768, help="Reject overlength; never truncate silently")
    p.add_argument("--kl-heads", nargs="*", default=[], help="Explicit zero-based layer:head targets; costly")
    p.add_argument("--kl-unit", choices=["kv", "query"], default="kv")
    p.add_argument("--kl-every", type=int, default=1)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if transformers.__version__ != "4.45.2":
        p.error("Requires transformers==4.45.2")
    if a.output.exists():
        p.error("Output exists; choose a new directory")
    manifest = json.loads(a.heads.read_text())
    windows = sorted(set(a.windows if a.windows is not None else [max(0, manifest["recent_size"] - 1)]))
    sink = manifest["sink_size"] if a.sink_size is None else a.sink_size
    if min(windows) < 0 or sink < 0 or min(a.steps, a.prefill_chunk, a.max_prompt_tokens, a.kl_every) < 1:
        p.error("Invalid window, sink, or positive count")
    targets = parse_targets(a.kl_heads, manifest, a.kl_unit)
    samples = [json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    if not samples or any(not isinstance(s.get("id"), str) or not isinstance(s.get("prompt"), str) or not s["prompt"] for s in samples):
        p.error("Each sample needs a string id and nonempty prompt")
    if len({s["id"] for s in samples}) != len(samples):
        p.error("Sample ids must be unique")
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    revision = manifest.get("model_commit") or manifest["revision"]
    config = AutoConfig.from_pretrained(manifest["model"], revision=revision)
    original_window = getattr(config, "sliding_window", None)
    if hasattr(config, "sliding_window"):
        config.sliding_window = None
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], revision=revision)
    encoded = []
    for sample in samples:
        ids = tokenizer(sample["prompt"], return_tensors="pt").input_ids
        if not 0 < ids.shape[1] <= a.max_prompt_tokens:
            p.error(f"Prompt {sample['id']} has {ids.shape[1]} tokens; adjust explicit limit")
        continuation = None
        if "continuation" in sample:
            if not isinstance(sample["continuation"], str):
                p.error("continuation must be a string")
            continuation = tokenizer.encode(sample["continuation"], add_special_tokens=False)
            if not continuation:
                p.error("continuation must tokenize to at least one token")
        encoded.append((sample, ids, continuation))
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"], revision=revision, config=config,
        torch_dtype=getattr(torch, a.dtype), attn_implementation="eager",
        low_cpu_mem_usage=True).to(a.device).eval()
    a.output.mkdir(parents=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "heads": manifest, "windows": windows, "sink_size": sink,
        "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "repo_commit": commit,
        "experiment_source_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                     for f in Path(__file__).parent.glob("*.py")},
        "hardware": torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else platform.machine(),
        "model_config": config.to_dict(), "original_sliding_window": original_window,
        "trajectory": "full attention greedy or explicit teacher forcing",
        "metric_precision": "FP32 softmax and value aggregation from model-dtype QK logits",
        "status": "running",
    }
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None or a.ignore_eos else eos if isinstance(eos, list) else [eos]
    with (a.output / "metrics.jsonl").open("w") as metrics, (a.output / "tokens.jsonl").open("w") as tokens:
        for sample, ids, continuation in encoded:
            def emit(rows, token):
                for row in rows:
                    metrics.write(json.dumps({"sample_id": sample["id"], **row}, allow_nan=False) + "\n")
                tokens.write(json.dumps({"sample_id": sample["id"], **token}) + "\n")
                metrics.flush()
                tokens.flush()
                if token["step"] % 10 == 0:
                    print(f"{sample['id']}: step {token['step']}, position {token['position']}", flush=True)

            trace = measure(model, ids.to(a.device), manifest, windows, sink, a.steps,
                            a.prefill_chunk, targets, a.kl_unit, a.kl_every, continuation, eos, emit)
            with (a.output / "sequences.jsonl").open("a") as sequences:
                sequences.write(json.dumps({"sample_id": sample["id"], "prompt_token_ids": ids[0].tolist(),
                                            "continuation_token_ids": [t["next_token_id"] for t in trace],
                                            "mode": "teacher_forced" if continuation is not None else "greedy",
                                            "text": tokenizer.decode([t["next_token_id"] for t in trace])}) + "\n")
    metadata["status"] = "complete"
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Measurements saved to {a.output}")


if __name__ == "__main__":
    main()
