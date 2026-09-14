import json

import numpy as np
import pytest
from transformers import LlamaConfig

from experiments.retrieval_dynamics.classify import classify


@pytest.mark.parametrize("sparsity", [0.0, 0.5, 1.0])
def test_classifier_is_upstream_and_deterministic(tmp_path, sparsity):
    from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
    np.savetxt(tmp_path / "full_attention_heads.tsv", [[.5, .5], [.5, .5]], delimiter="\t")
    (tmp_path / "config.json").write_text(json.dumps({"sink_size": 1, "recent_size": 4}))
    config = LlamaConfig(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    got = classify(tmp_path, config, sparsity, 42)
    scores, _, _ = load_attn_pattern(tmp_path)
    np.random.seed(42)
    expected, actual = sparsify_attention_heads(scores, sparsity=sparsity)
    assert [h["retrieval"] for h in got["heads"]] == expected.flatten().astype(bool).tolist()
    assert got["actual_sparsity"] == actual
    assert got == classify(tmp_path, config, sparsity, 42)
    assert got["heads"][1]["query_heads"] == [2, 3]


def test_classifier_rejects_bad_shape(tmp_path):
    np.savetxt(tmp_path / "full_attention_heads.tsv", np.ones((2, 3)), delimiter="\t")
    (tmp_path / "config.json").write_text(json.dumps({"sink_size": 1, "recent_size": 4}))
    config = LlamaConfig(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    with pytest.raises(ValueError, match="Expected"):
        classify(tmp_path, config, .5, 42)


def test_classifier_does_not_reshape_transposed_pattern(tmp_path):
    np.savetxt(tmp_path / "full_attention_heads.tsv", np.ones((2, 4)), delimiter="\t")
    (tmp_path / "config.json").write_text(json.dumps({"sink_size": 1, "recent_size": 4}))
    config = LlamaConfig(num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)
    with pytest.raises(ValueError, match="Expected"):
        classify(tmp_path, config, .5, 42)
