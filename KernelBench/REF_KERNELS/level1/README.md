# Level 1 Ref Kernels

Only the kernels below were strict enough to vendor as handwritten CUDA references.

| Local kernel | Type | Vendored ref kernel(s) | Upstream |
|---|---|---|---|
| `1_Conv1d_Depthwise.py` | depthwise 1D conv (non-causal, groups=channels) | `./sources/pytorch_aten_depthwise_conv2d.cu` | [`pytorch/aten/src/ATen/native/cuda/DepthwiseConv2d.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DepthwiseConv2d.cu) |
| `3_CSRSpMMMessagePassing.py` | CSR sparse message passing | `./sources/dgl_spmm.cu` | [`dmlc/dgl/src/array/cuda/spmm.cu`](https://github.com/dmlc/dgl/blob/master/src/array/cuda/spmm.cu) |
| `5_Conv1d_Causal.py` | causal conv1d | `./sources/causal_conv1d_fwd.cu` | [`Dao-AILab/causal-conv1d/csrc/causal_conv1d_fwd.cu`](https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu) |
| `7_Conv1d_FFT.py` | FFT-based depthwise conv1d (rfft → pointwise multiply → irfft) | `./sources/h3_fftconv_cuda.cu` | [`HazyResearch/H3/csrc/fftconv/fftconv_cuda.cu`](https://github.com/HazyResearch/H3/blob/main/csrc/fftconv/fftconv_cuda.cu) |
| `9_AssociativeScan.py` | SSM linear recurrence / selective scan (parallel prefix) | `./sources/mamba_selective_scan_fwd_fp32.cu` | [`state-spaces/mamba/csrc/selective_scan/selective_scan_fwd_fp32.cu`](https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_fp32.cu) |
| `10_SampledDenseDenseMatmulEdges.py` | sampled edge dot products (SDDMM COO) | `./sources/dgl_sddmm.cu` | [`dmlc/dgl/src/array/cuda/sddmm.cu`](https://github.com/dmlc/dgl/blob/master/src/array/cuda/sddmm.cu) |
| `11_DegreeNormalizedAggregation.py` | degree-normalized SpMM (GCN-style) | `./sources/dgl_spmm.cu` | [`dmlc/dgl/src/array/cuda/spmm.cu`](https://github.com/dmlc/dgl/blob/master/src/array/cuda/spmm.cu) |
| `12_DeepSeekMoEDispatchPermute.py` | MoE dispatch / pack | `./sources/deep_ep_intranode.cu` | [`deepseek-ai/DeepEP/csrc/kernels/legacy/intranode.cu`](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/legacy/intranode.cu) |
| `13_FarthestPointSampling3D.py` | farthest point sampling | `./sources/pointnet2_sampling_gpu.cu` | [`erikwijmans/Pointnet2_PyTorch/.../sampling_gpu.cu`](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu) |
| `14_GaussianProcessRBFKernel.py` | batched RBF kernel matrix K[i,j] = σ² exp(−½‖xᵢ−xⱼ‖²/ℓ²) | `./sources/thundersvm_kernelmatrix_kernel.cu` | [`Xtra-Computing/thundersvm/src/thundersvm/kernel/kernelmatrix_kernel.cu`](https://github.com/Xtra-Computing/thundersvm/blob/master/src/thundersvm/kernel/kernelmatrix_kernel.cu) |
| `15_COOScatterAddNodeFeatures.py` | COO scatter-add into node accumulators | `./sources/pytorch_scatter_cuda.cu` | [`rusty1s/pytorch_scatter/csrc/cuda/scatter_cuda.cu`](https://github.com/rusty1s/pytorch_scatter/blob/master/csrc/cuda/scatter_cuda.cu) |
| `16_DeepSeekMoECombineScatter.py` | MoE combine / scatter | `./sources/deep_ep_intranode.cu` | [`deepseek-ai/DeepEP/csrc/kernels/legacy/intranode.cu`](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/legacy/intranode.cu) |
| `17_CSRMaxAggregation.py` | CSR max reduction | `./sources/dgl_segment_reduce.cu` | [`dmlc/dgl/src/array/cuda/segment_reduce.cu`](https://github.com/dmlc/dgl/blob/master/src/array/cuda/segment_reduce.cu) |
| `18_CSRMultiHeadSpMM.py` | multi-head sparse matmul | `./sources/dgl_spmm.cu` | [`dmlc/dgl/src/array/cuda/spmm.cu`](https://github.com/dmlc/dgl/blob/master/src/array/cuda/spmm.cu) |
| `19_PairwiseDistanceMatrix.py` | full N×N pairwise Euclidean distance (cdist) | `./sources/pytorch_aten_distance_kernel.cu` | [`pytorch/aten/src/ATen/native/cuda/DistanceKernel.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DistanceKernel.cu) |
| `20_RotaryPositionEmbeddingBio.py` | RoPE on q/k | `./sources/flashinfer_rope.cu` | [`flashinfer-ai/flashinfer/csrc/rope.cu`](https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/rope.cu) |
| `24_VoxelGridPooling.py` | scatter point features into 3D voxel grid (atomic-add) | `./sources/pytorch3d_points_to_volumes.cu` | [`facebookresearch/pytorch3d/pytorch3d/csrc/points_to_volumes/points_to_volumes.cu`](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/points_to_volumes/points_to_volumes.cu) |

Excluded from this directory (no exact single `.cu` execution-pattern match found):

- `2_GraphEdgeSoftmaxCSR.py` — DGL edge softmax is multi-pass segment reduce (no standalone .cu); pytorch_scatter `segment_csr_cuda.cu` handles sum/max/min only, not softmax
- `4_GJKRobotEnvironmentIntersection.py` — openGJK GPU is full 3-D; cuRobo uses sphere-sphere; no 2-D AABB-triangle SAT .cu found
- `6_EdgeSoftmaxMultiHeadCSR.py` — same issue as kernel 2; multi-head makes standalone match harder
- `8_SegmentTopKCSR.py` — moderngpu segsort is `.cuh` header-only; no public per-row top-k CSR `.cu` found
- `21_RadialBasisFunctionExpansion.py` — NNPOps ANI `computeRadialFunctions` aggregates over neighbors by species (different output structure); NNPOps SchNet CFConv applies subsequent neural network layers (over-inclusive); no pure Gaussian-smearing-only `.cu` found
- `22_SE3InvariantLinear.py` — no standalone PaiNN/EGNN-style SE(3)-invariant linear `.cu`; cuEquivariance has no `.cu` files
- `23_virtual_all_reduce_mean.py` — x.mean(dim=0) is too generic; PyTorch ReduceOps.cu is framework-heavy
- `25_virtual_reduce_scatter_masked_sum.py` — masked sum too generic; no exact standalone .cu
- `26_ImportanceSampling.py` — domain-specific (logsumexp + IS weights + ESS); no exact standalone .cu
- `27_permutation_all_to_all.py` — simple 2-D gather (x[:, PERM]); PyTorch IndexKernel.cu is framework-heavy; no minimal standalone gather .cu
- `28_DirichletMultinomialLogLikelihood.py` — no standalone lgamma-based Dirichlet-Multinomial .cu found in probabilistic ML repos
- `29_logsumexp_across_virtual_ranks.py` — standard logsumexp reduction; too generic; PyTorch ReduceOps.cu is framework-heavy
- `30_broadcast_masked_from_row0.py` — torch.where pointwise; too simple for a dedicated library .cu
- `31_bitmask_or_popcount.py` — bitwise OR reduction + popcount; no standalone bioinformatics .cu with matching pattern found
- `32_argmax_tiebreak_smallest_rank.py` — custom argmax with smallest-rank tiebreak on (R,B,C); no public exact .cu
