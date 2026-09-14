# Validation performed

Validated on 2026-09-14, macOS ARM64 CPU, Python 3.13.15, PyTorch 2.14.0,
Transformers 4.45.2, with a separate local virtual environment.

`python -m pytest experiments/retrieval_dynamics/tests -q`: **17 passed**.

Coverage includes:

- Exact local-window boundary, sink overlap, and all-visible-key retention.
- Hand-computed attention masses and output differences; masked re-softmax
  remains finite when retained full-attention probabilities would underflow.
- Tiny random Llama MHA, Llama GQA, and Mistral GQA: instrumented cached decode
  logits match native full-prefix logits with `atol=2e-7`, `rtol=2e-6` in FP32.
- Query and KV-group interventions match an independent reference that replaces
  the native `o_proj` input, for both Llama and Mistral.
- KV-group delta equals the norm of concatenated query-head differences.
- KL cache forks leave baseline generation unchanged; a window covering every
  key gives zero delta and KL below `1e-12`.
- Fixed-continuation indexing, missing/subsampled KL handling, eligible-step
  denominators, temporal event counting, and forward restoration on exceptions.
- Binary classifications exactly match the original DuoAttention functions,
  including seeded ties and sparsity endpoints; invalid/transposed shapes fail.

The complete CLI chain also passed using `make_smoke_fixture`: create a random
tiny model, classify synthetic scores with upstream functions, measure four
teacher-forced steps at three windows, intervene on one KV group every second
step, and generate delta/KL summaries and PNGs. The delta plot was visually
inspected. This is a plumbing test, not evidence for temporal retrieval sparsity.

The smoke test initially found that current setuptools no longer provided
`pkg_resources`, which upstream `tensor_parallel==2.0.0` imports. The experiment's
requirements now pin `setuptools<81`; upstream files were not changed. Remaining
deprecation warnings come from those original dependencies.

Not run here: original multi-GPU head training, downloaded pretrained models,
CUDA BF16/FP16 numerical parity, long-context memory/performance checks, or the
scientific study across held-out tasks. Run these on the target CUDA environment
before interpreting real-model rho. No empirical claim about retrieval sparsity
or LServe speedup is made from these tests.
