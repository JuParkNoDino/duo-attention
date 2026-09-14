# Is retrieval-ness static at the head level?

This additive experiment freezes **DuoAttention's own classification first**, then
measures how those original heads behave across full-attention decoding steps.
All upstream files remain unchanged. New code lives in
`experiments/retrieval_dynamics/`. This is an offline observation and intervention
tool, not an inference optimization or a new head classifier.

Upstream base: `fe93c314ae87306ef6629dc16713250b4718ffe7` from
<https://github.com/mit-han-lab/duo-attention>.

## 1. Set up the environment

Run all commands from this repository root. For **training new head patterns**,
follow the original [README](README.md#environment-setup) on a CUDA Linux machine,
including FlashAttention and Block Sparse Attention, model access, and the BookSum
dataset. The original training pipeline uses NCCL and cannot run on this Mac.

For observation of an already learned pattern, only these Python dependencies
are needed; the observer does not import the CUDA DuoAttention patches:

```bash
# Use a separate environment with Python 3.10 or 3.11.
python -m venv .venv
source .venv/bin/activate
# Install a torch build appropriate to your CUDA version, or CPU for tiny tests.
pip install torch
pip install -r experiments/retrieval_dynamics/requirements.txt
pip install -e .
```

`transformers==4.45.2` is mandatory, matching upstream. The observer refuses other
versions because attention/cache APIs change. Supported models are the original,
unquantized HF Llama and Mistral architectures, one device, batch size one,
`pretraining_tp=1`. Do not load DuoAttention-reordered or QServe weights. Gated
models need your Hugging Face access set up first. Start at 4–8K context and a
short continuation before attempting a long run.

## 2. Let DuoAttention learn/classify the heads

**Option A: learn your own pattern using the unchanged training code.** After
upstream setup and downloading `models/Llama-3-8B-Instruct-Gradient-1048k` and
`datasets/booksum.jsonl.zst`:

```bash
bash scripts/train.sh Llama-3-8B-Instruct-Gradient-1048k 1000 32000 0.05 0.02 10
```

This original script requests **8 CUDA processes**, runs 2,000 training steps,
and writes to
`attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10`.
It may overwrite the bundled pattern at that path. To keep the shipped pattern,
invoke the original trainer with a new output directory instead:

```bash
torchrun --standalone --nnodes 1 --nproc_per_node 8 duo_attn/train.py \
  --model_name models/Llama-3-8B-Instruct-Gradient-1048k \
  --dataset_name datasets/booksum.jsonl.zst --dataset_format multiple_passkey \
  --batch_size 1 --max_length 32000 --context_length_min 1000 \
  --context_length_max 32000 --context_lengths_num_intervals 50 \
  --depth_ratio_num_intervals 1000 --min_needle_depth_ratio 0.05 \
  --max_needle_depth_ratio 0.95 --num_passkeys 10 --sink_size 128 \
  --recent_size 256 --num_steps 2000 --lr 0.02 --reg_weight 0.05 \
  --gradient_accumulation_steps 1 --disable_wandb \
  --output_dir work/trained-pattern
```

Hardware-dependent training batch/context settings may need adjustment. Wait for
training to finish and use the final `full_attention_heads.tsv` plus `config.json`.
Do not rename an arbitrary incomplete checkpoint as the final classifier.

**Option B: start with the patterns already learned by DuoAttention.** This skips
training, but still applies the repo's own binary classification in the next step.
The following commands use this option for a reproducible initial run:

```bash
MODEL=models/Llama-3-8B-Instruct-Gradient-1048k
PATTERN=attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10
# If you trained above, use: PATTERN=work/trained-pattern

python -m experiments.retrieval_dynamics.classify \
  --model "$MODEL" --pattern "$PATTERN" --sparsity 0.5 --seed 42 \
  --output work/retrieval-heads.json
```

`classify` calls `duo_attn.utils.load_attn_pattern` and
`duo_attn.utils.sparsify_attention_heads` unchanged. Here `--sparsity 0.5` means
approximately 50% **streaming** heads, as in upstream. Its small random tie-breaking
noise is seeded; the manifest records the realized sparsity, scores, binary labels,
pattern checksum/config, model identity/revision, and original zero-based indexes.
Outputs must be new paths to avoid silently replacing an experiment.

**Use the exact model that the pattern was trained for.** Dimensions are checked,
but two different checkpoints can have the same dimensions. Local model paths are
not immutable: keep weights unchanged after classification. Remote models use the
resolved config commit when available; `--revision` can pin a commit explicitly.

DuoAttention's score matrix is `[layer, KV head]`. With GQA, KV head `h` labels
query heads `h*g ... (h+1)*g-1`, where `g = n_query_heads/n_kv_heads`. The manifest
lists this mapping. No head reordering is performed.

## 3. Prepare held-out prompts

Input is JSONL with one object per line: a unique string `id`, a `prompt` string,
and optionally a `continuation` string for teacher forcing. Prompts are passed
verbatim, without an automatically added chat template. Apply the model's chat
template beforehand if your experiment requires it.

For a first deliberately constructed retrieval/control pair:

```bash
mkdir -p work
python - <<'PY'
import json
from pathlib import Path
context = "The archive contains routine notes on weather, gardens, and repairs.\n" * 500
samples = [
    {"id": "early-key", "prompt": "Archive access code: 739251.\n" + context +
     "\nWrite a paragraph about organizing this archive. At the end, quote the access code from the start.\n"},
    {"id": "local-control", "prompt": context +
     "\nWrite a paragraph about organizing an archive of routine notes.\n"},
]
Path("work/prompts.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
PY
```

This pair is an exploratory stimulus, not evidence by itself. Use many held-out
prompts across tasks, context lengths, and needle depths, including natural
long-form generation and local controls. Avoid testing only the synthetic task
used to train the classifier. Check generated text for actual task success.

Without `continuation`, decoding is greedy from the full-attention model. With
`continuation`, its tokens are fed along a fixed trajectory; KL still compares
the full next-token distributions. Continuation text is tokenized separately
with `add_special_tokens=False` and appended to prompt token IDs. This explicit
boundary may differ from tokenizing the concatenated string. Saved token IDs
are the authoritative trajectory. EOS stops greedy generation unless
`--ignore-eos` is given; teacher forcing stops at the shorter of its length and
`--steps`.

## 4. Measure full attention and head-output differences

```bash
python -m experiments.retrieval_dynamics.run \
  --heads work/retrieval-heads.json --input work/prompts.jsonl \
  --output work/dynamics --device cuda:0 --dtype bfloat16 \
  --windows 127 255 511 --steps 256 --prefill-chunk 128
```

Every layer and query head is measured at every decoding step, including heads
classified as streaming, which provide controls. Both query-level and KV-group
rows are saved. The full baseline always retains all KV states. Prefill uses
chunked native HF eager attention; measured decoding computes one full attention
row per head rather than saving an entire square attention matrix.

### Exact quantities and indexing

Let `t = position` be the zero-based position of the query token, with visible
keys `0..t`. `step=0` processes the last prompt token and predicts the **first**
continuation token. At each specified window `W`:

* `g_far = sum(A[i] for i < t-W)`, exactly the proposed definition.
* Streaming retains `i < sink_size OR i >= t-W`, with a fresh masked softmax.
  This retains **W previous tokens plus the current token** and the sinks.
* `g_discarded` sums mass outside that union. It excludes distant sinks, which
  `g_far` includes. Sink-heavy heads can have large `g_far` but no streaming loss.
* `delta = ||o_full - o_streaming||_2` before `o_proj`.
* `relative_delta = delta / max(||o_full||_2, 1e-12)`.
* `eligible` says whether any visible key would actually be discarded.

For KV-group rows, mass is the mean over associated query heads, and delta is
the L2 norm of the **concatenated** query-head differences. It is not the mean
query-head delta. Query rows remain available for finer inspection.

The repo's `generate_streaming_mask` counts self in `recent_size`, so its exact
token reference mask corresponds to `W = recent_size - 1`. If `--windows` is
omitted, this is the default (clamped at zero). The fused block-sparse training
kernel uses block-rounded windows; this experiment uses an exact token mask,
not that block approximation. The sink size defaults to the pattern's saved
`sink_size` and can be overridden with `--sink-size`.

Diagnostics use FP32 softmax and value aggregation from the model-dtype QK
logits. Baseline and intervened forward outputs use the native model dtype.
For close threshold decisions, repeat with `--dtype float32` and check numerical
sensitivity. `delta` is a proxy for dependence, not by itself downstream necessity.

Mistral's `sliding_window` is explicitly disabled before model construction so
the baseline is globally causal. This override is recorded in metadata.

## 5. Add next-token KL for selected heads

Inspect the actual retrieval indexes, then select a small subset:

```bash
python - <<'PY'
import json
h = json.load(open("work/retrieval-heads.json"))
print(" ".join(f"{x['layer']}:{x['kv_head']}" for x in h["heads"] if x["retrieval"]))
PY
```

For example, **only if `0:0` is a target you want to test**:

```bash
python -m experiments.retrieval_dynamics.run \
  --heads work/retrieval-heads.json --input work/prompts.jsonl \
  --output work/dynamics-kl --device cuda:0 --dtype bfloat16 \
  --windows 255 --steps 256 --kl-heads 0:0 --kl-unit kv --kl-every 1
```

`--kl-unit kv` replaces all query heads in that KV group at the selected layer.
Use `--kl-unit query` to intervene on just one original query head. Multiple
targets are listed as `--kl-heads 0:0 2:3`; each gets an **independent**, single-head
intervention. Streaming-labelled heads can also be selected as controls.

For each target/window, the runner copies the baseline cache **before** the
current step, streams only the selected head's current output, runs remaining
layers, and computes `KL(p_full || p_intervened)` over the entire vocabulary in
FP64, in nats. No sampled-token approximation is used. Interventions never enter
the baseline cache or choose its next token. This measures a **one-step causal
intervention on an otherwise full-attention prefix**, not sustained eviction or
long-term free-running error. KV-group KL includes within-group interactions.

KL costs one additional model forward per target per window per measured step,
and extra full-cache copies. Start with a few heads and one window. `--kl-every 8`
samples every eighth step to reduce cost; unmeasured KL is `null`, never zero.
Sparse KL sampling cannot resolve short event bursts between samples.

## 6. Plot time traces and quantify temporal sparsity

```bash
python -m experiments.retrieval_dynamics.analyze \
  --run work/dynamics --metric delta --epsilons 0.01 0.1 1.0
python -m experiments.retrieval_dynamics.analyze \
  --run work/dynamics --metric relative_delta --epsilons 0.01 0.05 0.1
python -m experiments.retrieval_dynamics.analyze \
  --run work/dynamics --metric g_discarded --epsilons 0.01 0.05 0.1 --plot-unit query
# After the KL run:
python -m experiments.retrieval_dynamics.analyze \
  --run work/dynamics-kl --metric kl --epsilons 0.00001 0.0001 0.001
```

The epsilon values are illustrative sensitivity sweeps, **not calibrated
necessity thresholds**. Choose thresholds using independent quality measurements.
Plots are generated for every retrieval head in the selected unit by default;
`--plot-controls` includes streaming-labelled heads. Each plot shows mass traces
and the selected metric across decoding steps. `index.json` maps plot filenames
to prompts and heads. Re-analyzing the same metric replaces that metric's summary
and plots; raw observations are unchanged.

Each `summary_METRIC.csv` has one row per prompt/layer/unit/head/window/epsilon:

* `rho`: number of observed steps with metric **strictly greater** than epsilon,
  divided by all observed steps, matching the proposed definition.
* `rho_eligible`: the same fraction restricted to steps where streaming discards
  at least one key; prevents short contexts from creating trivial inactivity.
* `n_observed`, `n_eligible`, mean, p95, event-run count, longest inactive run.
  Run-length statistics are omitted for nonconsecutive observations, such as
  subsampled KL. Missing observations are excluded, not counted as inactivity.

Interpret query-head and KV-group rho separately. Compare retrieval-labelled
heads with streaming controls, sweep W/sinks/epsilon, and report prompt/task-level
variation. Do not pool correlated token observations into an independent-sample
confidence interval. A low rho on one prompt is not evidence of a universal head
property. Summaries intentionally keep prompts separate.

## 7. Inspect outputs and evaluate the hypothesis

Each completed run contains:

| File | Contents |
| --- | --- |
| `metadata.json` | Frozen head manifest, arguments, model config, source/version information, input hash, completion state |
| `metrics.jsonl` | Every step, layer, original head, W, label, G, delta, relative delta, optional KL |
| `tokens.jsonl` | Position, input token, chosen/forced next token, baseline argmax |
| `sequences.jsonl` | Exact prompt and continuation token IDs, decoded continuation, generation mode |
| `summary_*.csv` | Per-prompt threshold sweeps for both retrieval and streaming heads |
| `plots_*/` | Per-head PNG traces plus an index |

Interrupted runs remain `status=running`; analysis refuses them. Use a new output
directory to rerun. There is no resume mode. No attention matrices are written to
disk. Runtime is still expensive: full KV memory is linear in context, prefill
compute is quadratic overall, decode attention is linear per step. Reduce
`--prefill-chunk` for transient prefill memory; reduce context length for KV OOM.
Prompts beyond `--max-prompt-tokens` (default 32768) are rejected, never silently
truncated. Raise the limit explicitly only within model/hardware constraints.

Evidence for temporal sparsity would be consistent long inactive runs punctuated
by metric/KL events on retrieval-labelled heads, replicated across held-out tasks
and reasonable thresholds, while task quality is preserved. If rho is high, or
delta and KL disagree, that is also an informative result.

Even rho=5% does **not** establish 95% achievable speedup or a deployable mode
switch: an oracle observation can depend on full attention, future retrieval
still needs access to old KV, and switching/prediction overhead and cumulative
errors remain to be measured. This experiment does not make a verified novelty
claim about LServe; it tests the temporal-static approximation independently.

## 8. Validate the implementation without downloading a real model

```bash
python -m pytest experiments/retrieval_dynamics/tests -q

python -m experiments.retrieval_dynamics.make_smoke_fixture --output work/smoke
python -m experiments.retrieval_dynamics.classify \
  --model work/smoke/model --pattern work/smoke/pattern --output work/smoke/heads.json
python -m experiments.retrieval_dynamics.run \
  --heads work/smoke/heads.json --input work/smoke/input.jsonl \
  --output work/smoke/run --device cpu --dtype float32 \
  --windows 0 2 100 --steps 4 --kl-heads 0:0 --kl-every 2
python -m experiments.retrieval_dynamics.analyze \
  --run work/smoke/run --metric delta
```

This uses a **random tiny model and synthetic scores** solely to verify plumbing.
It is not a learned retrieval classifier or scientific result. See
[VALIDATION.md](experiments/retrieval_dynamics/VALIDATION.md) for checks actually
run during development and the remaining real-model validation.
