# Popcorn stress references (eval-only)

These trees are **not** loaded by normal sweeps. They are used by
[`scripts/reval_popcorn_stress_sweep.py`](../../scripts/reval_popcorn_stress_sweep.py)
after you copy a finished run under `runs/`.

## Regenerating from canonical popcorn

The `.py` files under `large/`, `awkward/`, and `xl/` are produced by
[`scripts/gen_popcorn_stress_refs.py`](../../scripts/gen_popcorn_stress_refs.py)
(AST-based scaling of module-level int hyperparameters: `seq_len`, `batch_size`,
`M`/`N`/`K`, graph counts, channels, with stricter caps for O(N²) pair
attention). Re-run after pulling new popcorn problems or changing scaling rules:

```bash
uv run python scripts/gen_popcorn_stress_refs.py
```

## Directory layout

Pick a root (this repo uses `KernelBench/stress_refs` as the default example in
the script’s `--stress-refs` argument). Under that root, mirror popcorn problems
in three tiers:

```text
stress_refs/
  large/level1/popcorn/01_....py
  large/level2/popcorn/...
  large/level3/popcorn/...
  awkward/level1/popcorn/01_....py   # same filenames as canonical popcorn
  awkward/level2/popcorn/...
  awkward/level3/popcorn/...
  xl/level1/popcorn/01_....py
  xl/level2/popcorn/...
  xl/level3/popcorn/...
```

- **Filenames** (including the numeric prefix / problem id) must match
  [`KernelBench/level{N}/popcorn/`](../level1/popcorn/) so trajectories resolve
  to the correct kernel.
- **Contents** are edited copies: larger tensors, awkward lengths, etc., per
  your methodology. Each tier should still define a valid `Model`,
  `get_inputs`, and `get_init_inputs` compatible with the same API shape you
  intend agents to implement (typically `ModelNew` matching `Model` I/O).

## Run stress re-eval

```bash
uv run python scripts/reval_popcorn_stress_sweep.py \
  --src-run runs/pop_l123_default_gpt \
  --dst-run-name pop_l123_default_gpt_stress \
  --stress-refs KernelBench/stress_refs
```

This copies `runs/pop_l123_default_gpt` → `runs/pop_l123_default_gpt_stress` and
overwrites `final_result` / `outcome` **only inside the copy**. Each kernel is
evaluated against **large**, **awkward**, and **xl** refs in sequence, with
**5** correctness trials per tier (**15** total when all tiers succeed).

Canonical popcorn refs randomize CSR layouts and other structural dimensions
per correctness trial via ``popcorn_pri`` (see ``kernelbench.popcorn_random_inputs``);
regenerate stress trees after changing that logic.

Tier directory names can be changed with `--tiers`.

## Website / static hosting

The same tree can be published as static files (paths are stable relative to the
repo). Point readers at this README and the raw `.py` paths under each tier.
