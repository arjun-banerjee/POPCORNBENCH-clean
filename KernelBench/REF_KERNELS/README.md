This directory vendors handwritten CUDA references for `popcorn` kernels.

Reference classes:
- `exact reference kernel`: the upstream `.cu` matches the local Popcorn kernel closely enough at the execution-pattern level that it can be used directly as a fair handwritten baseline
- `primitive / component reference`: the upstream `.cu` is a real handwritten kernel for a sub-operation used by the Popcorn kernel, but the local Popcorn kernel wraps that primitive with extra compute; use these for component benchmarking, not as a headline end-to-end baseline
- `bundle reference`: a small set of handwritten kernels together covers the right execution family; this is appropriate when no single `.cu` is the whole local kernel

Inclusion rule:
- the upstream `.cu` file is copied exactly from the cited GitHub source, except for the added top-of-file PopcornBench mapping / provenance comments
- each vendored file must map to a real Popcorn kernel in this repo
- reused upstream kernels are intentional when multiple Popcorn kernels share the same primitive execution pattern

Layout:
- `sources/`: vendored upstream `.cu` files, one copy per upstream source
- `level1/` to `level4/`: per-level manifests mapping local kernels to the vendored source files

How to use these references:
- use `exact reference kernel` entries for direct speedup / baseline comparison
- use `primitive / component reference` entries to benchmark the communication or subkernel portion of a larger Popcorn kernel
- use `bundle reference` entries when the local kernel is a fused multi-stage pattern and no single public `.cu` is the whole story

Note on repeated baselines:
- repeated use of the same upstream `.cu` is expected and not a problem here
- there are not many truly distinct public handwritten kernels for some sparse, collective, and scan primitives
- when two Popcorn kernels share the same primitive, reusing the same vetted handwritten baseline is more honest than inventing artificial variety

This directory is stricter than [POPCORN_BASELINE_KERNEL_REFS_VERIFIED.md](/home/jkorr/POPCORNBENCH/KernelBench/POPCORN_BASELINE_KERNEL_REFS_VERIFIED.md): the broader manifest may include additional verified exact or bundle mappings that are not yet vendored into the per-level tree.
