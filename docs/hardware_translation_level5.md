# Hardware translation (Level 5): A100 `.cu` → H100 CUDA

This guide describes the **same-language hardware translation** task: start from CUDA tuned for **Ampere (A100)** and produce a PyTorch module (`ModelNew`) that runs efficiently on **Hopper (H100)**, while matching the **reference PyTorch** (`Model`) outputs.

The repo ships raw CUDA under [`KernelBench/level5/kernels/`](../KernelBench/level5/kernels/README.md) (`a100/` vs `h100/`). The evaluation stack compares:

| Signal | Meaning |
|--------|---------|
| **Correctness** | Generated `ModelNew` vs reference `Model` (same contract as other KernelBench levels). |
| **`speedup_vs_source`** | Candidate vs **source** kernel timing when you pass A100 reference CUDA wrapped as `ModelNew` (optional eval wiring). |
| **`speedup_vs_hardware_reference`** | Candidate vs **expert H100** baseline implemented as Python `ModelNew` wrapping the hand-tuned `kernels/h100/*.cu` logic (optional). |
| **Standard PyTorch timing** | Existing `ref_runtime` / speedup vs eager PyTorch. |

---

## 1. Layout on disk

Use a **KernelBench problem file** per task (`.py` with `Model`, `get_init_inputs`, `get_inputs`) plus CUDA trees:

```
KernelBench/level5/
├── kernels/
│   ├── a100/           # Source `.cu` / `.cuh` shown in the prompt (input)
│   └── h100/           # Expert CUDA for the target GPU (reference for perf)
├── tasks_txt/          # Optional: <stem>.txt sidecars (prompt notes)
├── reference_torch/    # You provide: NN_<stem>.py benchmark problems (example layout name)
├── _translation_sources/
│   └── cuda/           # Symlinks or copies: <stem>.cu from kernels/a100/
└── hardware_benchmark_refs/   # You provide: <stem>.py wrapping kernels/h100 (ModelNew baseline)
```

Naming rule: the benchmark problem **`01_foo.py`** pairs with **`01_foo.cu`** (same numeric prefix and stem) under `_translation_sources/cuda/` or `kernels/a100/`.

See [`KernelBench/level5/kernels/README.md`](../KernelBench/level5/kernels/README.md) for the shipped CUDA list.

---

## 2. Prompt contents (`.cu` + `.txt`)

Hardware translation prompts are built from `prompt_option=hardware_translation` and [`prompts.toml`](../src/kernelbench/prompts/prompts.toml) option **`hardware_translation`**.

- **CUDA body**: Injected from the translation-source tree as explicit **Level 5 CUDA source** (your `.cu` text).
- **Optional `.txt`**: Concatenated from:
  - `hardware_translation_auxiliary_txt_path` — single global file; and/or
  - `hardware_translation_auxiliary_txt_dir` — per-problem `<stem>.txt`.

Implementation helpers live in [`src/kernelbench/hardware_translation_utils.py`](../src/kernelbench/hardware_translation_utils.py).

---

## 3. Generation (batch)

Example **`generate_samples.py`** invocation shape:

```bash
uv run python scripts/generate_samples.py \
  run_name=hw_trans_l5 \
  dataset_src=local \
  level=5 \
  variant=reference_torch \
  prompt_option=hardware_translation \
  backend=cuda \
  source_backend=cuda \
  source_kernel_dir=KernelBench/level5/_translation_sources/cuda \
  source_hardware_gpu_name=A100 \
  hardware_gpu_name=H100 \
  gpu_arch='["Hopper"]' \
  hardware_translation_auxiliary_txt_dir=KernelBench/level5/tasks_txt
```

Notes:

- Set **`source_kernel_dir`** to the directory that contains `<stem>.cu` files (or exact `.py`-named copies). The resolver accepts either the legacy filename equal to the problem name **or** `<stem>.cu` / `<stem>.cuh`.
- **`hardware_gpu_name`** drives target GPU facts in the prompt; **`gpu_arch`** should match your compile target (Hopper for H100).

---

## 4. Evaluation & baselines

### 4.1 Correctness

Standard **`eval_kernel_against_ref`** compares outputs to the PyTorch reference — unchanged.

### 4.2 Timing the **source** A100 kernel (optional)

When **`hardware_translation_eval_source_dir`** points at Python-wrapped **source** kernels (each defining `ModelNew` for the A100 implementation), batch eval records **`speedup_vs_source`**:

```bash
uv run python scripts/eval_from_generations.py \
  run_name=hw_trans_l5 \
  dataset_src=local \
  level=5 \
  variant=reference_torch \
  hardware_translation_eval_source_dir=KernelBench/level5/_translation_sources/cuda_wrappers_a100 \
  hardware_translation_eval_source_backend=cuda \
  benchmark_reference_kernel_dir=KernelBench/level5/hardware_benchmark_refs
```

(Adjust paths to where you keep compilable `ModelNew` sources.)

### 4.3 Timing the **expert H100** kernel (optional)

Provide **`benchmark_reference_kernel_dir`** containing `<stem>.py` files that define **`ModelNew`** wrapping the expert Hopper CUDA (`kernels/h100/`). Evaluation then fills:

- `hardware_reference_runtime`
- `speedup_vs_hardware_reference` (expert time ÷ candidate time)

Single-sample CLI mirrors this via **`benchmark_reference_kernel_dir`** or **`benchmark_reference_kernel_path`**.

---

## 5. Single problem debugging

```bash
uv run python scripts/generate_and_eval_single_sample.py \
  dataset_src=local \
  level=5 \
  variant=reference_torch \
  problem_id=1 \
  prompt_option=hardware_translation \
  backend=cuda \
  source_kernel_path=KernelBench/level5/kernels/a100/01_paged_attention_v1.cu \
  source_hardware_gpu_name=A100 \
  hardware_gpu_name=H100 \
  gpu_arch='["Hopper"]' \
  hardware_translation_auxiliary_txt_dir=KernelBench/level5/tasks_txt \
  benchmark_reference_kernel_dir=KernelBench/level5/hardware_benchmark_refs \
  server_type=... \
  model_name=...
```

---

## 6. What you must author locally

The repository includes **CUDA sources** under `kernels/` but does **not** ship full Level 5 PyTorch problems or Python wrappers for every `.cu` file (those depend on your extensions, includes, and build flags). For a runnable benchmark you typically add:

1. **`reference_torch/*.py`** — KernelBench-style reference models.
2. **`_translation_sources/cuda/<stem>.cu`** — copy or symlink from `kernels/a100/`.
3. **Optional** **`hardware_benchmark_refs/<stem>.py`** — `ModelNew` wrapping `kernels/h100/` expert paths for **`speedup_vs_hardware_reference`**.
4. **Optional** **`tasks_txt/<stem>.txt`** — build flags / symbol naming hints for the model.

---

## Related code

- Prompt wiring: [`get_hardware_translation_prompt`](../src/kernelbench/prompt_constructor_toml.py)
- Asset resolution: [`hardware_translation_utils.py`](../src/kernelbench/hardware_translation_utils.py)
- Eval timing fields: [`KernelExecResult`](../src/kernelbench/eval.py) (`hardware_reference_runtime`, `speedup_vs_hardware_reference`)
