# Verified Popcorn Baseline Kernel References

This file is a strict verification manifest for `KernelBench/level1/popcorn` through
`KernelBench/level4/popcorn`.

It is intentionally conservative:

- Every entry in the reference catalog below points to a specific upstream GitHub `.cu` file.
- If I could only verify a repo, a `.cuh` / `.hpp` / generated header, or a Python wrapper,
  I did **not** upgrade that to an exact CUDA reference here.
- For many Popcorn kernels, the best honest status is still `partial` or `missing`.

This is meant for review, not for pretending every benchmark already has a perfect handwritten baseline.

Guidance for benchmark use:

- prefer `exact` entries when you want a direct handwritten baseline speedup metric
- use `partial` entries when the upstream kernel is only a meaningful subcomponent baseline
- use `bundle` entries when the local Popcorn kernel is fused or multi-stage and no single public `.cu` is the whole execution
- repeated references are expected: public handwritten kernels for sparse ops, collectives, and scans are limited, so the same vetted baseline often maps to multiple Popcorn kernels

## Status meanings

- `exact`: a verified public `.cu` file is a strong semantic match for the local kernel.
- `bundle`: no single `.cu` file is the whole benchmark, but a small bundle of verified component kernels is the right handwritten baseline.
- `partial`: I found a verified `.cu` file for an important subcomponent, but not a full 1:1 kernel match.
- `missing`: I do not have a verified exact public `.cu` baseline pinned yet.

## Verified reference catalog

| ID | Repo | Upstream `.cu` file | URL |
|---|---|---|---|
| `R1` | DGL | `src/array/cuda/spmm.cu` | https://github.com/dmlc/dgl/blob/master/src/array/cuda/spmm.cu |
| `R2` | DGL | `src/array/cuda/segment_reduce.cu` | https://github.com/dmlc/dgl/blob/master/src/array/cuda/segment_reduce.cu |
| `R3` | PointNet2_PyTorch | `pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu` | https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu |
| `R4` | PointNet2_PyTorch | `pointnet2_ops_lib/pointnet2_ops/_ext-src/src/ball_query_gpu.cu` | https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/ball_query_gpu.cu |
| `R5` | PointNet2_PyTorch | `pointnet2_ops_lib/pointnet2_ops/_ext-src/src/group_points_gpu.cu` | https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/group_points_gpu.cu |
| `R6` | PointNet2_PyTorch | `pointnet2_ops_lib/pointnet2_ops/_ext-src/src/interpolate_gpu.cu` | https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/interpolate_gpu.cu |
| `R7` | FlashInfer | `csrc/batch_decode.cu` | https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/batch_decode.cu |
| `R8` | FlashInfer | `csrc/page.cu` | https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/page.cu |
| `R9` | FlashInfer | `csrc/rope.cu` | https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/rope.cu |
| `R10` | FlashInfer | `csrc/norm.cu` | https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/norm.cu |
| `R11` | FlashInfer | `csrc/batch_decode_mla_cute_sm80.cu` | https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/batch_decode_mla_cute_sm80.cu |
| `R12` | FlashInfer | `csrc/rmsnorm_silu.cu` | https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/rmsnorm_silu.cu |
| `R13` | FlashAttention | `csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu` | https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu |
| `R14` | FlashAttention | `csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu` | https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu |
| `R15` | FlashAttention | `csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu` | https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu |
| `R16` | FlashAttention | `hopper/instantiations/flash_fwd_hdim128_fp16_softcapall_sm80.cu` | https://github.com/Dao-AILab/flash-attention/blob/main/hopper/instantiations/flash_fwd_hdim128_fp16_softcapall_sm80.cu |
| `R17` | Mamba | `csrc/selective_scan/selective_scan_fwd_fp32.cu` | https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_fp32.cu |
| `R18` | Mamba | `csrc/selective_scan/selective_scan_bwd_fp32_real.cu` | https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_bwd_fp32_real.cu |
| `R19` | causal-conv1d | `csrc/causal_conv1d_fwd.cu` | https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu |
| `R20` | causal-conv1d | `csrc/causal_conv1d_bwd.cu` | https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_bwd.cu |
| `R21` | DeepEP | `csrc/kernels/legacy/intranode.cu` | https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/legacy/intranode.cu |
| `R22` | DeepEP | `csrc/kernels/legacy/internode.cu` | https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/legacy/internode.cu |
| `R23` | CogDL | `cogdl/operators/edge_softmax/edge_softmax.cu` | https://github.com/THUDM/CogDL/blob/master/cogdl/operators/edge_softmax/edge_softmax.cu |
| `R24` | NCCL Tests | `src/all_reduce.cu` | https://github.com/NVIDIA/nccl-tests/blob/master/src/all_reduce.cu |
| `R25` | NCCL Tests | `src/all_gather.cu` | https://github.com/NVIDIA/nccl-tests/blob/master/src/all_gather.cu |
| `R26` | NCCL Tests | `src/reduce_scatter.cu` | https://github.com/NVIDIA/nccl-tests/blob/master/src/reduce_scatter.cu |
| `R27` | NCCL Tests | `src/alltoall.cu` | https://github.com/NVIDIA/nccl-tests/blob/master/src/alltoall.cu |
| `R28` | NCCL Tests | `src/sendrecv.cu` | https://github.com/NVIDIA/nccl-tests/blob/master/src/sendrecv.cu |
| `R29` | NCCL Tests | `src/broadcast.cu` | https://github.com/NVIDIA/nccl-tests/blob/master/src/broadcast.cu |
| `R30` | DGL | `src/array/cuda/sddmm.cu` | https://github.com/dmlc/dgl/blob/master/src/array/cuda/sddmm.cu |
| `R31` | H3 | `csrc/fftconv/fftconv_cuda.cu` | https://github.com/HazyResearch/H3/blob/main/csrc/fftconv/fftconv_cuda.cu |
| `R32` | ThunderSVM | `src/thundersvm/kernel/kernelmatrix_kernel.cu` | https://github.com/Xtra-Computing/thundersvm/blob/master/src/thundersvm/kernel/kernelmatrix_kernel.cu |
| `R33` | pytorch_scatter | `csrc/cuda/scatter_cuda.cu` | https://github.com/rusty1s/pytorch_scatter/blob/master/csrc/cuda/scatter_cuda.cu |
| `R34` | PyTorch | `aten/src/ATen/native/cuda/DistanceKernel.cu` | https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DistanceKernel.cu |
| `R35` | PyTorch3D | `pytorch3d/csrc/points_to_volumes/points_to_volumes.cu` | https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/points_to_volumes/points_to_volumes.cu |

## Important exclusions

These are relevant projects, but I did **not** count them as exact verified `.cu` references here:

- `OpenFold`: I verified the repo, but not a matching public `.cu` implementation file for the Evoformer / IPA kernels.
- `FastFold`: same issue from the current verification pass.
- `pyg-lib`: relevant graph library, but I did not pin the needed exact `.cu` files from this pass.
- `DeepGEMM`: relevant for FP8 GEMM, but the visible public implementation surface from this pass was `.hpp` / `.cuh`, not a clean `.cu` target.
- `CUB` / `CCCL`: highly relevant for scans and reductions, but the useful implementation surface is mostly header-based rather than one benchmark-style `.cu` file.

## Level 1 Popcorn

| Kernel | Type | Refs | Status | Notes |
|---|---|---:|---|---|
| `1_Conv1d_Depthwise.py` | depthwise conv1d | — | `missing` | I did not verify a matching public depthwise-only `.cu` file in this pass. |
| `2_GraphEdgeSoftmaxCSR.py` | CSR edge softmax | `R23` | `exact` | CogDL provides a direct CSR edge-softmax CUDA implementation. |
| `3_CSRSpMMMessagePassing.py` | CSR sparse message passing | `R1` | `exact` | DGL `spmm.cu` is a direct sparse aggregation reference. |
| `4_GJKRobotEnvironmentIntersection.py` | robotics collision / geometry | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `5_Conv1d_Causal.py` | causal conv1d | `R19`, `R20` | `exact` | Strong match to handwritten causal Conv1d kernels. |
| `6_EdgeSoftmaxMultiHeadCSR.py` | multi-head CSR edge softmax | `R23` | `exact` | CogDL's CUDA kernel normalizes each CSR row independently for every head. |
| `7_Conv1d_FFT.py` | FFT conv1d | `R31` | `exact` | H3's FFT convolution CUDA kernel matches the FFT-conv execution family. |
| `8_SegmentTopKCSR.py` | segmented top-k on CSR | `R2` | `partial` | Segmented reduction primitive verified, exact segmented top-k `.cu` still unpinned. |
| `9_AssociativeScan.py` | scan / prefix op | `R17` | `exact` | The Mamba selective-scan CUDA kernel is a strong handwritten scan-style reference. |
| `10_SampledDenseDenseMatmulEdges.py` | sampled edge GEMM | `R30` | `exact` | DGL `sddmm.cu` is the direct sampled dense-dense matmul edge primitive. |
| `11_DegreeNormalizedAggregation.py` | normalized sparse aggregation | `R1`, `R2` | `partial` | Strong components exist, but no 1:1 kernel file pinned. |
| `12_DeepSeekMoEDispatchPermute.py` | MoE dispatch / all-to-all | `R21`, `R22` | `exact` | DeepEP legacy intranode/internode kernels are the closest verified dispatch baseline. |
| `13_FarthestPointSampling3D.py` | farthest point sampling | `R3` | `exact` | Direct PointNet2 CUDA match. |
| `14_GaussianProcessRBFKernel.py` | GP covariance kernel | `R32` | `exact` | ThunderSVM's kernel-matrix CUDA file is a strong RBF-kernel baseline. |
| `15_COOScatterAddNodeFeatures.py` | COO scatter-add | `R33` | `exact` | `pytorch_scatter` provides the exact scatter-add CUDA family. |
| `16_DeepSeekMoECombineScatter.py` | MoE combine / all-to-all | `R21`, `R22` | `exact` | DeepEP combine path is a strong verified baseline. |
| `17_CSRMaxAggregation.py` | CSR max reduction | `R2` | `exact` | DGL segment max reduction is a direct primitive match. |
| `18_CSRMultiHeadSpMM.py` | multi-head sparse matmul | `R1` | `exact` | DGL sparse matmul is the right verified family. |
| `19_PairwiseDistanceMatrix.py` | all-pairs distance | `R34` | `exact` | PyTorch's CUDA distance kernel is the exact pairwise-distance family. |
| `20_RotaryPositionEmbeddingBio.py` | RoPE | `R9` | `exact` | FlashInfer RoPE is a strong handwritten baseline. |
| `21_RadialBasisFunctionExpansion.py` | RBF expansion | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `22_SE3InvariantLinear.py` | SE(3)-aware linear op | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `23_virtual_all_reduce_mean.py` | virtual all-reduce | `R24` | `partial` | The collective family is pinned via NCCL Tests, but the local kernel is a local packed-rank mean rather than a distributed call. |
| `24_VoxelGridPooling.py` | voxel pooling | `R35` | `exact` | PyTorch3D's points-to-volumes CUDA implementation is a direct voxel-pooling reference. |
| `25_virtual_reduce_scatter_masked_sum.py` | virtual reduce-scatter | — | `missing` | Relevant runtime exists, but no pinned exact public `.cu` file from this pass. |
| `26_ImportanceSampling.py` | Monte Carlo weighting | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `27_permutation_all_to_all.py` | all-to-all permutation | `R21`, `R22` | `partial` | DeepEP is the right family, but the local task is more generic than the pinned dispatch/combine kernels. |
| `28_DirichletMultinomialLogLikelihood.py` | probabilistic reduction | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `29_logsumexp_across_virtual_ranks.py` | distributed reduction | — | `missing` | No pinned exact `.cu` reference from this pass. |
| `30_broadcast_masked_from_row0.py` | distributed broadcast | `R29` | `partial` | Broadcast is pinned exactly, but the local kernel adds masking semantics on top. |
| `31_bitmask_or_popcount.py` | bitwise reduction | — | `missing` | No pinned exact `.cu` reference from this pass. |
| `32_argmax_tiebreak_smallest_rank.py` | distributed argmax | — | `missing` | No pinned exact `.cu` reference from this pass. |

## Level 2 Popcorn

| Kernel | Type | Refs | Status | Notes |
|---|---|---:|---|---|
| `1_DeepSeekMLALoRAExpansion.py` | MLA decode / head expansion | `R11` | `exact` | Verified FlashInfer MLA decode kernel. |
| `2_all_reduce_data_parallel.py` | collective all-reduce | `R24` | `exact` | NCCL Tests provides a direct all-reduce CUDA reference file. |
| `3_FusedCrossAttentionRoPEDropout.py` | cross-attention + RoPE | `R9`, `R13`, `R14` | `bundle` | Best handwritten baseline is a component bundle, not one exact file. |
| `4_GFVLA_FusedSceneGraphUpdate.py` | VLA graph fusion | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `5_LayerNorm_GELU_DilatedConv1d_ResAdd.py` | fused norm + conv | `R10`, `R19`, `R20` | `partial` | Strong subcomponents verified, but not a 1:1 fused file. |
| `6_TriangularAttention.py` | AlphaFold triangular attention | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `7_Conv1d_Depthwise_Separable.py` | depthwise separable conv1d | — | `missing` | No verified exact public `.cu` file pinned here. |
| `8_DeepSeekMoEGroundedTop2Routing.py` | MoE routing + comms | `R21`, `R22` | `partial` | Communication kernels verified; routing-specific exact `.cu` still unpinned. |
| `9_MetropolisHastingsStep.py` | MCMC step | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `10_GFVLA_FusedPairwiseNodeAggregate.py` | VLA graph aggregation | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `11_broadcast_parameter_shard.py` | parameter broadcast | `R29` | `exact` | NCCL Tests provides the matching broadcast collective CUDA reference. |
| `12_BatchNorm_GELU_DilatedConv1d_ResAdd.py` | fused norm + conv | `R19`, `R20` | `partial` | Verified handwritten conv kernels, but not the full exact fusion. |
| `13_QuantVLA_SelectiveLayoutFusion.py` | VLA layout fusion | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `14_TriangularMultiplicativeUpdateOutgoing.py` | Evoformer triangular update | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `15_GQAKVHeadExpansionAttention.py` | grouped-query attention / KV expand | `R7` | `exact` | FlashInfer batch decode with paged KV is the right verified family. |
| `16_HamiltonianMonteCarloStep.py` | HMC step | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `17_FP8ScaledAttention.py` | FP8 attention | `R7` | `partial` | Verified attention kernel exists, but not a pinned exact FP8 `.cu` file from this pass. |
| `18_all_gather_tensor_parallel.py` | collective all-gather | `R25` | `exact` | NCCL Tests provides a direct all-gather CUDA reference file. |
| `19_DepthwiseSeparableConv1D_GELU_ResAdd.py` | depthwise-separable conv fusion | — | `missing` | No verified exact public `.cu` file pinned here. |
| `20_GatedDeltaNetLinearAttention.py` | linear attention / SSM family | `R17`, `R18` | `partial` | Strong selective-scan component verified, but not a pinned exact DeltaNet `.cu`. |
| `21_TriangularMultiplicativeUpdateIncoming.py` | Evoformer triangular update | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `22_FusedBallQuery3D.py` | point-cloud neighborhood query | `R4`, `R5` | `exact` | Ball-query plus grouping kernels verified. |
| `23_GibbsSamplingStep.py` | Gibbs sampling | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `24_KimiDeltaAttentionChannelwise.py` | Kimi / Delta attention | `R17`, `R18` | `partial` | Verified SSM/scan family exists, but no pinned exact local-equivalent `.cu`. |
| `25_OuterProductMean.py` | Evoformer outer-product mean | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `26_RoPEKVCacheUpdate.py` | RoPE + KV-cache append | `R8`, `R9` | `exact` | Strong exact family match via FlashInfer. |
| `27_reduce_scatter_grad_shard.py` | collective reduce-scatter | `R26` | `exact` | NCCL Tests provides a direct reduce-scatter CUDA reference file. |
| `28_FusedGroupNormSiLU.py` | fused normalization + SiLU | `R12` | `partial` | Verified RMSNorm+SiLU exists, but not exact GroupNorm. |
| `29_MSARowAttention.py` | MSA row attention | `R13`, `R14` | `partial` | Verified attention kernels, but not a pinned dedicated MSA-row `.cu`. |
| `30_SelectiveScan_Mamba.py` | selective scan | `R17`, `R18` | `exact` | Direct Mamba CUDA match. |
| `31_FusedMLAAttention.py` | MLA attention | `R11` | `exact` | Strong FlashInfer MLA decode match. |
| `32_MSAColumnAttention.py` | MSA column attention | `R13`, `R14` | `partial` | Verified attention kernels, but not a pinned dedicated MSA-column `.cu`. |
| `33_VectorizedPoseTransformFused.py` | robotics pose transform | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `34_all_to_all_permutation.py` | all-to-all permutation | `R27` | `exact` | NCCL Tests provides a direct all-to-all CUDA reference file. |
| `35_FusedConv1dGroupNorm.py` | conv + group norm | `R19`, `R20` | `partial` | Strong conv subkernel verified; exact groupnorm fusion still unpinned. |
| `36_InvariantPointAttention.py` | IPA | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `37_CSRFusedAttentionValue.py` | fused sparse attention value | `R23`, `R1` | `bundle` | The right handwritten baseline is a two-kernel bundle: CSR edge softmax plus sparse aggregation. |
| `38_pipeline_stage_p2p.py` | pipeline p2p | `R28`, `R29` | `bundle` | The right handwritten baseline is a send/recv plus broadcast collective bundle. |
| `39_Fused1DTemporalConvolution.py` | temporal conv1d fusion | `R19`, `R20` | `partial` | Strong conv subkernel verified; full exact fusion still unpinned. |
| `40_SmithWatermanDPScore.py` | dynamic programming score kernel | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `41_VariationalELBO.py` | variational reduction / pointwise | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `42_BatchedForwardKinematics.py` | robotics kinematics | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `43_ContactMapPrediction.py` | protein contact map | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `44_ReparameterizationTrick.py` | reparameterization op | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `45_AxialAttention.py` | axial attention | `R13`, `R14` | `partial` | Attention kernels verified, but not a dedicated axial-attention `.cu`. |
| `46_SteinVariationalGradient.py` | SVGD update | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `47_virtual_all_gather_concat_linear.py` | collective + concat linear | — | `missing` | No pinned exact public `.cu` file from this pass. |
| `48_ParticleFilter.py` | particle filter step | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `49_sum_gelu_across_virtual_ranks.py` | distributed reduction + GELU | — | `missing` | No pinned exact public `.cu` file from this pass. |
| `50_HiddenMarkovForward.py` | HMM forward pass | — | `missing` | No verified public matching `.cu` file pinned yet. |

## Level 3 Popcorn

| Kernel | Type | Refs | Status | Notes |
|---|---|---:|---|---|
| `1_DilatedConvTower_GenomicBlock.py` | dilated / causal conv tower | `R19`, `R20` | `partial` | Strong conv subkernels verified; exact block fusion still unpinned. |
| `2_FusedMultiStepDiffusionSolver.py` | diffusion solver fusion | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `3_SSM2D_FusedBidirectionalCrossScan.py` | scan / SSM | `R17`, `R18` | `partial` | Strong scan kernels verified, but not a 2D bidirectional exact `.cu`. |
| `4_MambaBlock.py` | Mamba block | `R17`, `R18`, `R19`, `R20` | `bundle` | Best handwritten baseline is selective-scan plus causal-conv bundle. |
| `5_FlowMatchingProbabilityFlowODE.py` | ODE / diffusion update | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `6_HyenaOperator.py` | long convolution operator | `R19`, `R20` | `partial` | Strong conv subkernel family verified; exact Hyena `.cu` still unpinned. |
| `7_GaussianProcessRegression.py` | GP solve / regression | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `8_EvoformerBlock.py` | Evoformer block | `R13`, `R14` | `partial` | Attention component verified; exact public Evoformer `.cu` still unpinned. |
| `9_BayesianLinearRegression.py` | Bayesian linear algebra | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `10_GaussianMixtureModelEM.py` | mixture-model EM | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `11_pipeline_three_stage_mlp.py` | pipeline / staged MLP | — | `missing` | No pinned exact public `.cu` file from this pass. |
| `12_NormalizingFlowPlanar.py` | planar flow | — | `missing` | No verified public matching `.cu` file pinned yet. |
| `13_ConditionalVAELoss.py` | CVAE loss | — | `missing` | No verified public matching `.cu` file pinned yet. |

## Level 4 Popcorn

These are whole-model wrappers. A whole-model `.cu` file usually does not exist, so the honest
baseline shape is a verified **component bundle**.

### Level 4 verified bundles

| Bundle ID | Use case | Verified component refs |
|---|---|---|
| `B1` | decoder-only transformer serving | `R7`, `R8`, `R9`, `R10`, `R13`, `R14` |
| `B2` | MLA / DeepSeek-style serving | `R7`, `R8`, `R9`, `R10`, `R11`, `R12`, `R21`, `R22` |
| `B3` | long-conv / SSM sequence models | `R17`, `R18`, `R19`, `R20` |
| `B4` | generic attention-heavy model | `R13`, `R14` |
| `B5` | VLM / VLA transformer runtime | `R7`, `R8`, `R9`, `R10`, `R13`, `R14` |

| Kernel | Type | Refs | Status | Notes |
|---|---|---:|---|---|
| `1_meta-llama-Llama-3.2-1B-Instruct.py` | decoder-only LLM | `B1` | `bundle` | Best handwritten baseline is a serving kernel bundle, not one model `.cu`. |
| `2_esm2_650m_bs32_seq256.py` | protein transformer | `B4` | `partial` | Verified attention kernels exist; exact protein-stack `.cu` not pinned. |
| `3_mistralai-Mistral-7B-v0.1.py` | decoder-only LLM | `B1` | `bundle` | Same reasoning as Llama. |
| `4_esm2_650m_bs1_seq1022.py` | protein transformer | `B4` | `partial` | Verified attention kernels exist; exact protein-stack `.cu` not pinned. |
| `5_mistralai-Mixtral-8x7B-Instruct-v0.1.py` | MoE decoder LLM | `B2` | `bundle` | MoE comms plus serving attention bundle is the right baseline. |
| `6_esm2_650m_bs512_seq128.py` | protein transformer | `B4` | `partial` | Verified attention kernels exist; exact protein-stack `.cu` not pinned. |
| `7_google-gemma-2-2b.py` | decoder-only LLM | `B1` | `bundle` | Same serving bundle logic. |
| `8_ntv2_100m_bs32_seq256.py` | genomics / long sequence model | `B3` | `bundle` | Best verified component bundle is scan + causal-conv. |
| `9_tiiuae-falcon-7b.py` | decoder-only LLM | `B1` | `bundle` | Same serving bundle logic. |
| `10_ntv2_100m_bs1_seq2048.py` | genomics / long sequence model | `B3` | `bundle` | Same scan + causal-conv bundle. |
| `11_bigcode-starcoder2-3b.py` | decoder-only LLM | `B1` | `bundle` | Same serving bundle logic. |
| `12_hyenadna_32k_bs32_seq8192.py` | HyenaDNA long-conv model | `B3` | `bundle` | Exact Hyena `.cu` not pinned; verified long-conv bundle exists. |
| `13_microsoft-phi-2.py` | decoder-only LLM | `B1` | `bundle` | Same serving bundle logic. |
| `14_hyenadna_32k_bs1_seq32000.py` | HyenaDNA long-conv model | `B3` | `bundle` | Same scan + causal-conv bundle. |
| `15_EleutherAI-pythia-1.4b.py` | decoder-only LLM | `B1` | `bundle` | Same serving bundle logic. |
| `16_caduceus_ph_bs16_seq8192.py` | SSM / long sequence model | `B3` | `bundle` | Best verified component bundle is selective-scan plus causal-conv. |
| `17_openai-community-gpt2-large.py` | decoder-only LLM | `B1` | `bundle` | Same serving bundle logic. |
| `18_gpn_msa_bs32_seq128.py` | MSA / protein attention model | `B4` | `partial` | Verified attention components exist; exact MSA CUDA stack not pinned. |
| `19_facebook-opt-1.3b.py` | decoder-only LLM | `B1` | `bundle` | Same serving bundle logic. |
| `20_gpn_star_hg38_v100_bs32_seq512.py` | genomics attention model | `B4` | `partial` | Verified attention components exist; exact local whole-model `.cu` not pinned. |
| `21_deepseek-ai-deepseek-llm-7b-base.py` | DeepSeek LLM | `B2` | `bundle` | MLA / MoE / serving bundle is the right baseline. |
| `22_Qwen-Qwen2.5-1.5B.py` | decoder-only LLM | `B1` | `bundle` | Same serving bundle logic. |
| `23_llava-hf-llava-1.5-7b-hf.py` | VLM | `B5` | `bundle` | Verified runtime kernels exist at the component level. |
| `24_HuggingFaceM4-idefics2-8b.py` | VLM | `B5` | `bundle` | Verified runtime kernels exist at the component level. |
| `25_openvla-openvla-7b.py` | VLA | `B5` | `partial` | Transformer runtime pieces verified; robotics-specific exact `.cu` pieces remain unpinned. |
| `26_smolvla-SmolVLA-Base.py` | VLA | `B5` | `partial` | Same limitation as OpenVLA. |
| `27_xvla-XVLA-7B.py` | VLA | `B5` | `partial` | Same limitation as OpenVLA. |
| `28_stabilityai-stable-diffusion-xl-base-1.0.py` | diffusion UNet | `B4` | `partial` | Verified attention pieces exist; exact conv/UNet CUDA stack not pinned here. |
| `29_facebook-esm2_t12_35M_UR50D.py` | protein transformer | `B4` | `partial` | Verified attention kernels exist; exact protein-stack `.cu` not pinned. |
| `30_microsoft-ClimaX-small.py` | transformer-like foundation model | `B4` | `partial` | Verified attention kernels exist; exact model-specific CUDA stack not pinned. |

## Short review summary

- The strongest exact verified Popcorn baselines from this pass are:
  - DGL sparse SpMM / segment reduction
  - PointNet2 farthest-point-sampling / ball-query / grouping
  - FlashInfer decode / page / rope / norm / MLA decode
  - FlashAttention exact attention kernels
  - Mamba selective-scan
  - Dao causal-conv1d
  - DeepEP MoE dispatch / combine communication

- The biggest remaining gaps are:
  - robotics geometry kernels
  - exact Evoformer / IPA CUDA files
  - probabilistic-model kernels
  - collective kernels where the public implementation surface is not a clean pinned `.cu`
  - whole-model wrappers that really need subkernel manifests rather than one-file baselines
