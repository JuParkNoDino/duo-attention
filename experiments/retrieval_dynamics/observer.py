"""Offline single-token eager attention, scoped to unpatched HF 4.45.2 models.

Prefill uses the original HF forward. Decode follows HF's eager math; only an
explicit intervention changes the pre-o_proj head output. No KV is evicted.
"""
import math
import types

import torch
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def masks(length, window, sink, device):
    """t is the zero-based current query position; W previous tokens + self."""
    if window < 0 or sink < 0 or length < 1:
        raise ValueError("Invalid attention geometry")
    indices = torch.arange(length, device=device)
    far = indices < length - 1 - window  # exactly i < t-W
    keep = (~far) | (indices < sink)
    return far, keep


def compare(logits, values, window, sink):
    """[query_heads, keys], [query_heads, keys, dim]; FP32 diagnostics."""
    far, keep = masks(logits.shape[-1], window, sink, logits.device)
    p = logits.float().softmax(-1)
    ps = logits.float().masked_fill(~keep, -torch.inf).softmax(-1)
    if not torch.isfinite(p).all() or not torch.isfinite(ps).all():
        raise FloatingPointError("Nonfinite attention probabilities; check inputs or use --dtype float32")
    full = torch.einsum("hk,hkd->hd", p, values.float())
    streaming = torch.einsum("hk,hkd->hd", ps, values.float())
    return {
        "g_far": p[:, far].sum(-1),
        "g_discarded": p[:, ~keep].sum(-1),
        "difference": full - streaming, "full": full,
        "stream_probs": ps, "eligible": bool((~keep).any()),
    }


class Observer:
    def __init__(self, model, manifest, windows, sink):
        if transformers.__version__ != "4.45.2":
            raise ValueError("Observer requires transformers==4.45.2; fail closed on API changes")
        if model.config.model_type not in {"llama", "mistral"}:
            raise ValueError("Only Llama and Mistral are supported")
        if model.training or getattr(model.config, "pretraining_tp", 1) != 1:
            raise ValueError("Requires eval() and pretraining_tp=1")
        if getattr(model.config, "sliding_window", None) is not None:
            raise ValueError("Set config.sliding_window=None BEFORE loading the model for full attention")
        for key in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "model_type"):
            if getattr(model.config, key) != manifest[key]:
                raise ValueError(f"Head manifest/model mismatch: {key}")
        self.model, self.windows, self.sink = model, sorted(set(windows)), sink
        if not self.windows or min(self.windows) < 0 or sink < 0:
            raise ValueError("Nonnegative windows and sink size required")
        self.labels = {(h["layer"], h["kv_head"]): h["retrieval"] for h in manifest["heads"]}
        expected = {(l, h) for l in range(manifest["num_hidden_layers"])
                    for h in range(manifest["num_key_value_heads"])}
        if set(self.labels) != expected or len(manifest["heads"]) != len(expected):
            raise ValueError("Manifest must contain each original KV head exactly once")
        self.active = False
        self.intervention = None  # (layer, unit='kv'|'query', head, window)
        self.rows = []
        self.originals = []

    def __enter__(self):
        for layer in self.model.model.layers:
            module = layer.self_attn
            if module.__class__.__name__ not in {"LlamaAttention", "MistralAttention"} or hasattr(module, "full_attention_heads"):
                self.__exit__(None, None, None)
                raise ValueError("Load original weights with attn_implementation='eager'; do not apply DuoAttention patches")
            original = module.forward
            self.originals.append((module, original))

            def forward(module, hidden_states, _original=original, **kwargs):
                if not self.active:
                    return _original(hidden_states, **kwargs)
                return self.decode(module, hidden_states, **kwargs)

            module.forward = types.MethodType(forward, module)
        return self

    def __exit__(self, *args):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()

    def decode(self, m, hidden_states, attention_mask=None, position_ids=None,
               past_key_value=None, output_attentions=False, use_cache=False,
               cache_position=None, position_embeddings=None, **kwargs):
        if hidden_states.shape[:2] != (1, 1) or past_key_value is None:
            raise ValueError("Measurement requires batch=1, one query token, and DynamicCache")
        q = m.q_proj(hidden_states).view(1, 1, m.num_heads, m.head_dim).transpose(1, 2)
        k = m.k_proj(hidden_states).view(1, 1, m.num_key_value_heads, m.head_dim).transpose(1, 2)
        v = m.v_proj(hidden_states).view(1, 1, m.num_key_value_heads, m.head_dim).transpose(1, 2)
        cos, sin = position_embeddings if position_embeddings is not None else m.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = past_key_value.update(k, v, m.layer_idx,
                                    {"sin": sin, "cos": cos, "cache_position": cache_position})
        k, v = repeat_kv(k, m.num_key_value_groups), repeat_kv(v, m.num_key_value_groups)
        logits = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(m.head_dim)
        if attention_mask is not None:
            logits = logits + attention_mask[:, :, :, :k.shape[-2]]
        probabilities = logits.softmax(-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(probabilities, v)
        if self.intervention is None:
            for window in self.windows:
                stat = compare(logits[0, :, 0], v[0], window, self.sink)
                self.record(m, stat, window)
        elif self.intervention[0] == m.layer_idx:
            _, unit, head, window = self.intervention
            group = m.num_key_value_groups
            selected = list(range(head * group, (head + 1) * group)) if unit == "kv" else [head]
            _, keep = masks(k.shape[-2], window, self.sink, k.device)
            # Re-softmax logits, never renormalize rounded/underflowed probabilities.
            ps = logits[:, selected].masked_fill(~keep, -torch.inf).softmax(-1, dtype=torch.float32).to(q.dtype)
            output[:, selected] = torch.matmul(ps, v[:, selected])
        output = m.o_proj(output.transpose(1, 2).contiguous().reshape(1, 1, -1))
        return output, probabilities if output_attentions else None, past_key_value

    def record(self, m, stat, window):
        group = m.num_key_value_groups
        diff, full = stat["difference"], stat["full"]
        for unit, count in (("query", m.num_heads), ("kv", m.num_key_value_heads)):
            # One host transfer per unit, rather than a GPU synchronization per scalar.
            packed = torch.stack([
                diff.reshape(count, -1).norm(dim=-1),
                full.reshape(count, -1).norm(dim=-1),
                stat["g_far"].reshape(count, -1).mean(-1),
                stat["g_discarded"].reshape(count, -1).mean(-1),
            ], dim=-1).cpu().tolist()
            for head, (delta, scale, g_far, g_discarded) in enumerate(packed):
                kv = head // group if unit == "query" else head
                self.rows.append({
                    "layer": m.layer_idx, "unit": unit, "head": head, "kv_head": kv,
                    "retrieval": self.labels[m.layer_idx, kv], "window": window,
                    "sink_size": self.sink, "eligible": stat["eligible"],
                    "g_far": g_far, "g_discarded": g_discarded,
                    "delta": delta, "full_norm": scale,
                    "relative_delta": delta / max(scale, 1e-12),
                })
