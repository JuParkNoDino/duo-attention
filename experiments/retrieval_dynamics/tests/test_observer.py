import copy

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM
from transformers.cache_utils import DynamicCache

from experiments.retrieval_dynamics.analyze import summarize
from experiments.retrieval_dynamics.observer import Observer, compare, masks
from experiments.retrieval_dynamics.run import kl_divergence, measure


def tiny(kind="llama", kv=2):
    torch.manual_seed(7)
    cls, config_cls = (LlamaForCausalLM, LlamaConfig) if kind == "llama" else (MistralForCausalLM, MistralConfig)
    config = config_cls(vocab_size=31, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=kv, max_position_embeddings=128,
                        attention_dropout=0.0, sliding_window=None)
    config._attn_implementation = "eager"
    model = cls(config).eval()
    manifest = {k: getattr(config, k) for k in ["model_type", "num_hidden_layers", "num_attention_heads", "num_key_value_heads"]}
    manifest["heads"] = [{"layer": l, "kv_head": h, "retrieval": h == 0} for l in range(2) for h in range(kv)]
    return model, manifest


def test_boundaries_and_sink_overlap():
    far, keep = masks(8, 2, 2, "cpu")
    assert far.tolist() == [True] * 5 + [False] * 3
    assert keep.tolist() == [True, True, False, False, False, True, True, True]
    assert masks(3, 0, 8, "cpu")[1].all()
    assert masks(3, 10, 0, "cpu")[1].all()


def test_metrics_against_hand_computation_and_underflow():
    logits = torch.zeros(1, 4)
    values = torch.tensor([[[0.], [4.], [0.], [0.]]])
    s = compare(logits, values, window=0, sink=1)
    assert s["g_far"].item() == .75
    assert s["g_discarded"].item() == .5
    assert s["difference"].item() == 1.0
    assert s["stream_probs"].tolist() == [[.5, 0., 0., .5]]
    extreme = compare(torch.tensor([[10000., -10000.]]), torch.ones(1, 2, 1), 0, 0)
    assert torch.isfinite(extreme["difference"]).all()
    assert extreme["stream_probs"].tolist() == [[0., 1.]]


@pytest.mark.parametrize("kind,kv", [("llama", 4), ("llama", 2), ("mistral", 2)])
@torch.inference_mode()
def test_native_parity_prefill_cache_and_full_prefix(kind, kv):
    model, manifest = tiny(kind, kv)
    ids = torch.tensor([[1, 4, 2, 7, 3, 6]])
    original = model.model.layers[0].self_attn.forward
    reference = model(ids, use_cache=False).logits[:, -1]
    cache = DynamicCache()
    model(ids[:, :3], past_key_values=cache, use_cache=True)
    model(ids[:, 3:-1], past_key_values=cache, use_cache=True)
    with Observer(model, manifest, [0, 2, 100], 1) as observer:
        observer.active = True
        result = model(ids[:, -1:], past_key_values=cache, use_cache=True).logits[:, -1]
        torch.testing.assert_close(result, reference, atol=2e-7, rtol=2e-6)
        assert len(observer.rows) == 2 * (4 + kv) * 3
        assert all(r["delta"] == 0 and not r["eligible"] for r in observer.rows if r["window"] == 100)
        for layer in range(2):
            q = [r for r in observer.rows if r["layer"] == layer and r["unit"] == "query" and r["window"] == 0]
            k = [r for r in observer.rows if r["layer"] == layer and r["unit"] == "kv" and r["window"] == 0]
            group = 4 // kv
            for head in range(kv):
                assert k[head]["delta"] == pytest.approx(np.linalg.norm([r["delta"] for r in q[head*group:(head+1)*group]]), rel=1e-6)
    assert model.model.layers[0].self_attn.forward == original


@pytest.mark.parametrize("unit,head", [("query", 1), ("kv", 0)])
@pytest.mark.parametrize("kind", ["llama", "mistral"])
@torch.inference_mode()
def test_intervention_matches_independent_native_output_hook(kind, unit, head):
    model, manifest = tiny(kind)
    ids = torch.tensor([[1, 3, 4, 2, 6]])
    before = DynamicCache()
    model(ids[:, :-1], past_key_values=before, use_cache=True)
    baseline_cache = copy.deepcopy(before)
    baseline = model(ids[:, -1:], past_key_values=baseline_cache, use_cache=True, output_attentions=True)
    layer = 0
    selected = [head] if unit == "query" else [2 * head, 2 * head + 1]
    # Independent reference: native returned A, renormalized on retained keys,
    # injected into the native pre-o_proj tensor. Random FP32 test has no underflow.
    prob = baseline.attentions[layer][:, selected].clone()
    keep = torch.tensor([True, False, False, True, True])
    prob[..., ~keep] = 0
    prob /= prob.sum(-1, keepdim=True)
    values = baseline_cache.value_cache[layer].repeat_interleave(2, dim=1)[:, selected]
    replacement = prob @ values

    def inject(module, args):
        out = args[0].clone().view(1, 1, 4, 8)
        out[:, :, selected] = replacement.transpose(1, 2)
        return (out.reshape(1, 1, 32),)

    hook = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(inject)
    try:
        reference = model(ids[:, -1:], past_key_values=copy.deepcopy(before), use_cache=True).logits
    finally:
        hook.remove()
    with Observer(model, manifest, [1], 1) as observer:
        observer.active = True
        observer.intervention = (layer, unit, head, 1)
        actual = model(ids[:, -1:], past_key_values=copy.deepcopy(before), use_cache=True).logits
    torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-6)
    assert kl_divergence(baseline.logits, actual) > 0
    assert before.get_seq_length() == 4


@torch.inference_mode()
def test_kl_forks_do_not_change_trajectory_and_noop_kl():
    model, manifest = tiny()
    ids = torch.tensor([[1, 3, 5, 7]])
    rows = []
    baseline = measure(model, ids, manifest, [0, 100], 1, 4, 2)
    with_kl = measure(model, ids, manifest, [0, 100], 1, 4, 2,
                      targets=[(0, 0)], kl_every=2, emit=lambda r, t: rows.extend(r))
    assert baseline == with_kl
    measured = [r for r in rows if r["kl"] is not None]
    assert len(measured) == 4
    assert all(r["kl"] < 1e-12 for r in measured if r["window"] == 100)
    forced = measure(model, ids, manifest, [0], 1, 9, 2, continuation=[8, 9])
    assert [r["next_token_id"] for r in forced] == [8, 9]
    assert [r["position"] for r in forced] == [3, 4]


def test_sparse_event_summary_and_missing_kl():
    rows = [{"step": t, "delta": d, "eligible": t > 0, "kl": None}
            for t, d in enumerate([0, 0, 2, 0, 0])]
    summary = summarize(rows, "delta", 1)
    assert summary["rho"] == .2
    assert summary["rho_eligible"] == .25
    assert summary["event_runs"] == 1
    assert summary["longest_inactive_run"] == 2
    assert summarize(rows, "kl", 1)["rho"] is None
    assert summarize(rows[::2], "delta", 1)["event_runs"] is None


def test_restore_on_exception_and_reject_wrong_manifest():
    model, manifest = tiny()
    original = model.model.layers[0].self_attn.forward
    with pytest.raises(RuntimeError):
        with Observer(model, manifest, [1], 1):
            raise RuntimeError("test")
    assert model.model.layers[0].self_attn.forward == original
    manifest["num_key_value_heads"] = 4
    with pytest.raises(ValueError, match="mismatch"):
        Observer(model, manifest, [1], 1)
