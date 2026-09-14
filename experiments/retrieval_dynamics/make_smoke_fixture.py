"""Create a tiny RANDOM model and synthetic pattern to test plumbing, not science."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=16, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=128, eos_token_id=None)
    LlamaForCausalLM(config).save_pretrained(a.output / "model")
    tok = Tokenizer(WordLevel({"[UNK]": 0, **{str(i): i for i in range(1, 16)}}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(a.output / "model")
    (a.output / "pattern").mkdir()
    np.savetxt(a.output / "pattern/full_attention_heads.tsv", [[.9, .1], [.2, .8]], delimiter="\t")
    (a.output / "pattern/config.json").write_text(json.dumps({"sink_size": 1, "recent_size": 3, "synthetic": True}))
    (a.output / "input.jsonl").write_text(json.dumps({"id": "random-plumbing-only", "prompt": "1 2 3 4 5 6 7 8", "continuation": "9 10 11 12"}) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
