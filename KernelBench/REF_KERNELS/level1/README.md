# Level 1 Ref Kernels

Only the kernels below were strict enough to vendor as handwritten CUDA references.

| Local kernel | Type | Vendored ref kernel(s) | Upstream |
|---|---|---|---|
| `3_CSRSpMMMessagePassing.py` | CSR sparse message passing | `../sources/dgl_spmm.cu` | `dmlc/dgl/src/array/cuda/spmm.cu` |
| `5_Conv1d_Causal.py` | causal conv1d | `../sources/causal_conv1d_fwd.cu` | `Dao-AILab/causal-conv1d/csrc/causal_conv1d_fwd.cu` |
| `13_FarthestPointSampling3D.py` | farthest point sampling | `../sources/pointnet2_sampling_gpu.cu` | `erikwijmans/Pointnet2_PyTorch/.../sampling_gpu.cu` |
| `17_CSRMaxAggregation.py` | CSR max reduction | `../sources/dgl_segment_reduce.cu` | `dmlc/dgl/src/array/cuda/segment_reduce.cu` |
| `20_RotaryPositionEmbeddingBio.py` | RoPE on q/k | `../sources/flashinfer_rope.cu` | `flashinfer-ai/flashinfer/csrc/rope.cu` |

Excluded from this directory:
- any Level 1 kernel that only had a family-level match
- any kernel whose best public reference was not a single execution-valid `.cu` file
- `12_DeepSeekMoEDispatchPermute.py`, `16_DeepSeekMoECombineScatter.py`, and `18_CSRMultiHeadSpMM.py`, because the best public CUDA refs were not exact one-to-one execution matches
