"""Freeze DuoAttention's own learned-score -> binary-head classification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoConfig


def classify(pattern, config, sparsity, seed):
    # Deliberately call the original repo functions, including its tie breaking.
    from duo_attn.utils import load_attn_pattern, sparsify_attention_heads

    scores, sink, recent = load_attn_pattern(str(pattern))
    expected = (config.num_hidden_layers, config.num_key_value_heads)
    scores = np.asarray(scores)
    # loadtxt squeezes one-row/one-column matrices; never reshape a true matrix.
    if scores.ndim < 2 and scores.size == np.prod(expected) and 1 in expected:
        scores = scores.reshape(expected)
    if scores.shape != expected or not np.isfinite(scores).all():
        raise ValueError(f"Expected finite scores of shape {expected}; got {scores.shape}")
    if not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be in [0, 1]")
    np.random.seed(seed)
    binary, actual = sparsify_attention_heads(scores.copy(), sparsity=sparsity)
    nq, nk = config.num_attention_heads, config.num_key_value_heads
    if nq % nk:
        raise ValueError("Query heads must divide into contiguous KV groups")
    return {
        "schema_version": 1, "model_type": config.model_type,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": nq, "num_key_value_heads": nk,
        "pattern_dir": str(Path(pattern).resolve()),
        "pattern_sha256": hashlib.sha256(
            (Path(pattern) / "full_attention_heads.tsv").read_bytes()).hexdigest(),
        "upstream_utils_sha256": hashlib.sha256(
            Path(load_attn_pattern.__code__.co_filename).read_bytes()).hexdigest(),
        "pattern_config": json.loads((Path(pattern) / "config.json").read_text()),
        "sink_size": sink, "recent_size": recent, "seed": seed,
        "requested_sparsity": sparsity, "actual_sparsity": float(actual),
        "index_convention": "zero-based original weights, before DuoAttention reordering",
        "heads": [
            {"layer": l, "kv_head": h, "score": float(scores[l, h]),
             "retrieval": bool(binary[l, h]),
             "query_heads": list(range(h * (nq // nk), (h + 1) * (nq // nk)))}
            for l in range(expected[0]) for h in range(nk)
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--pattern", type=Path, required=True)
    p.add_argument("--sparsity", type=float, default=0.5,
                   help="Fraction of streaming KV heads (upstream semantics)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output exists; choose a new file to preserve the frozen classification")
    config = AutoConfig.from_pretrained(a.model, revision=a.revision)
    result = classify(a.pattern, config, a.sparsity, a.seed)
    result.update(model=a.model, revision=a.revision,
                  model_commit=getattr(config, "_commit_hash", None))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {sum(h['retrieval'] for h in result['heads'])} retrieval KV heads to {a.output}")


if __name__ == "__main__":
    main()
