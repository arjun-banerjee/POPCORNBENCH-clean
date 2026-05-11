This directory vendors only the strictest handwritten baseline CUDA references for `popcorn` kernels.

Inclusion rule:
- the local kernel and upstream CUDA source must match at the execution-pattern level, not just the broad model family
- if a reference was only a component-family or bundle match, it is intentionally excluded here

Layout:
- `sources/`: vendored upstream `.cu` files, one copy per upstream source
- `level1/` to `level4/`: per-level manifests mapping local kernels to the vendored source files

This is stricter than [POPCORN_BASELINE_KERNEL_REFS_VERIFIED.md](/home/jkorr/POPCORNBENCH/KernelBench/POPCORN_BASELINE_KERNEL_REFS_VERIFIED.md): some items that were acceptable as family-level references there are intentionally omitted here.
