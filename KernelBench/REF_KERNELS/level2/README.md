# Level 2 Ref Kernels

Only the kernels below were strict enough to vendor as handwritten CUDA references.

Interpretation:
- `exact reference kernel` means the local Popcorn kernel and the upstream `.cu` match closely enough for direct baseline comparison
- `point-to-point send/recv plus broadcast bundle` means the vendored refs are best used as component baselines for the communication stages, not as a single end-to-end fused-kernel baseline
- repeated use of NCCL collective refs is intentional because several Popcorn kernels reuse the same communication primitive family

| Local kernel | Type | Vendored ref kernel(s) | Upstream |
|---|---|---|---|
| `7_Conv1d_Depthwise_Separable.py` | depthwise separable 1D conv (depthwise groups=channels, then 1×1 pointwise) | `./sources/pytorch_aten_depthwise_conv2d.cu` | [`pytorch/aten/src/ATen/native/cuda/DepthwiseConv2d.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DepthwiseConv2d.cu) |
| `11_broadcast_parameter_shard.py` | exact reference kernel: broadcast from rank 0 before local compute | `./sources/nccl_tests_broadcast.cu` | [`NVIDIA/nccl-tests/src/broadcast.cu`](https://github.com/NVIDIA/nccl-tests/blob/master/src/broadcast.cu) |
| `18_all_gather_tensor_parallel.py` | exact reference kernel: equal-split all-gather collective | `./sources/nccl_tests_all_gather.cu` | [`NVIDIA/nccl-tests/src/all_gather.cu`](https://github.com/NVIDIA/nccl-tests/blob/master/src/all_gather.cu) |
| `19_DepthwiseSeparableConv1D_GELU_ResAdd.py` | depthwise + pointwise 1D conv with GELU + residual | `./sources/pytorch_aten_depthwise_conv2d.cu` | [`pytorch/aten/src/ATen/native/cuda/DepthwiseConv2d.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DepthwiseConv2d.cu) |
| `26_RoPEKVCacheUpdate.py` | apply rotary position embedding to K and scatter into KV cache | `./sources/flashinfer_rope.cu` | [`flashinfer-ai/flashinfer/csrc/rope.cu`](https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/rope.cu) |
| `27_reduce_scatter_grad_shard.py` | exact reference kernel: reduce-scatter collective | `./sources/nccl_tests_reduce_scatter.cu` | [`NVIDIA/nccl-tests/src/reduce_scatter.cu`](https://github.com/NVIDIA/nccl-tests/blob/master/src/reduce_scatter.cu) |
| `28_FusedGroupNormSiLU.py` | GroupNorm + SiLU (Swish) — GroupNorm is the dominant kernel; SiLU is elementwise epilogue | `./sources/pytorch_aten_group_norm_kernel.cu` | [`pytorch/aten/src/ATen/native/cuda/group_norm_kernel.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/group_norm_kernel.cu) |
| `30_SelectiveScan_Mamba.py` | SSM selective scan (Mamba S6): discretize A/B, run linear recurrence, D skip connection | `./sources/mamba_selective_scan_fwd_fp32.cu` | [`state-spaces/mamba/csrc/selective_scan/selective_scan_fwd_fp32.cu`](https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_fp32.cu) |
| `38_pipeline_stage_p2p.py` | primitive / bundle reference: point-to-point send/recv plus broadcast stages | `./sources/nccl_tests_sendrecv.cu`, `./sources/nccl_tests_broadcast.cu` | [`NVIDIA/nccl-tests/src/sendrecv.cu`](https://github.com/NVIDIA/nccl-tests/blob/master/src/sendrecv.cu), [`NVIDIA/nccl-tests/src/broadcast.cu`](https://github.com/NVIDIA/nccl-tests/blob/master/src/broadcast.cu) |
| `39_Fused1DTemporalConvolution.py` | depthwise (temporal) + SiLU + pointwise 1D conv — depthwise dispatches through PyTorch to depthwise2d kernel | `./sources/pytorch_aten_depthwise_conv2d.cu` | [`pytorch/aten/src/ATen/native/cuda/DepthwiseConv2d.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DepthwiseConv2d.cu) |

Excluded from this directory (no exact single `.cu` execution-pattern match found):

- `1_DeepSeekMLALoRAExpansion.py` — two chained GEMMs (hidden→latent, latent→K/V); PyTorch's matmul kernel is framework-heavy; no standalone MLA LoRA .cu
- `2_all_reduce_data_parallel.py` — exact NCCL Tests `all_reduce.cu` is pinned in the broader verification manifest, but not vendored into this strict per-level tree yet
- `3_FusedCrossAttentionRoPEDropout.py` — cross-attention + RoPE + dropout; FlashInfer batch_attention.cu is a complex dispatch wrapper; no standalone single-pass .cu
- `4_GFVLA_FusedSceneGraphUpdate.py` — LayerNorm + multi-head attention with pairwise MLP bias; no standalone .cu for this fused pattern
- `5_LayerNorm_GELU_DilatedConv1d_ResAdd.py` — multi-op block (LN, GELU, dilated conv, residual); no single standalone .cu
- `6_TriangularAttention.py` — AlphaFold2 triangular self-attention with pair bias; OpenFold only has `softmax_cuda_kernel.cu` (softmax sub-step only), not the full triangular attention
- `8_DeepSeekMoEGroundedTop2Routing.py` — top-2 expert selection on biased logits; no standalone per-token top-2 .cu
- `9_MetropolisHastingsStep.py` — batched MH MCMC step (random walk + accept/reject); too generic; no standalone MCMC .cu
- `10_GFVLA_FusedPairwiseNodeAggregate.py` — MLP over node-pair features + masked max-pool; no standalone .cu
- `12_BatchNorm_GELU_DilatedConv1d_ResAdd.py` — multi-op block (BN, GELU, conv, residual); no single standalone .cu
- `13_QuantVLA_SelectiveLayoutFusion.py` — int8 fake-quant blend + linear projection; no standalone quantization-blend .cu
- `14_TriangularMultiplicativeUpdateOutgoing.py` — AlphaFold2 triangular multiplicative update (einsum over shared index k); no standalone CUDA for this pattern in OpenFold or other public repos
- `15_GQAKVHeadExpansionAttention.py` — GQA with KV-head repeat + causal attention; FlashAttention GQA .cu files are per-head-dim template instantiations, not standalone
- `16_HamiltonianMonteCarloStep.py` — HMC MCMC (leapfrog + MH); too generic; no standalone .cu
- `17_FP8ScaledAttention.py` — explicit FP8 dequant + scaled attention; no standalone FP8-attention .cu without a full framework
- `20_GatedDeltaNetLinearAttention.py` — scalar-gated delta-rule linear recurrence (rank-1 state update); flash-linear-attention uses Triton; no standalone .cu
- `21_TriangularMultiplicativeUpdateIncoming.py` — incoming variant of triangular multiplicative update; same exclusion reason as kernel 14
- `22_FusedBallQuery3D.py` — ball query + channel-wise max-pool over all in-ball points; PointNet2 `ball_query_gpu.cu` finds up to nsample indices (padded), not an online max-pool over all in-ball points — different execution pattern
- `23_GibbsSamplingStep.py` — batched Gibbs sampling step; too generic; no standalone MCMC .cu
- `24_KimiDeltaAttentionChannelwise.py` — channel-gated delta attention state matrix recurrence; no standalone .cu
- `25_OuterProductMean.py` — AlphaFold2 outer product mean (MSA → pair); no standalone CUDA in OpenFold or ESMFold
- `29_MSARowAttention.py` — MSA row-wise gated attention with pair bias; no standalone CUDA (OpenFold only has softmax kernel)
- `31_FusedMLAAttention.py` — causal attention with on-the-fly compressed KV latent expansion; FlashInfer MLA kernels are for paged decode (different pattern); no standalone .cu
- `32_MSAColumnAttention.py` — MSA column-wise attention; same exclusion reason as kernel 29
- `33_VectorizedPoseTransformFused.py` — quaternion → rotation matrix → 3D point transform; no exact standalone .cu
- `34_all_to_all_permutation.py` — exact NCCL Tests `alltoall.cu` is pinned in the broader verification manifest, but not vendored into this strict per-level tree yet
- `35_FusedConv1dGroupNorm.py` — Conv1d + GroupNorm; regular Conv1d dispatches through cuDNN/im2col+gemm, not a simple kernel; no fused conv+gn standalone .cu
- `36_InvariantPointAttention.py` — AlphaFold2 IPA (sequence + 3D geometric attention); no standalone .cu for the full IPA pattern
- `37_CSRFusedAttentionValue.py` — no single exact `.cu`; the right handwritten baseline is a two-kernel bundle (CogDL `edge_softmax.cu` plus sparse aggregation)
- `40_SmithWatermanDPScore.py` — differentiable (soft-max relaxed) Smith-Waterman DP; CUDASW++ uses hard max, not the temperature-relaxed variant; no exact match
- `41_VariationalELBO.py` — VAE ELBO (BCE + KL divergence); too generic; no standalone VAE .cu
- `42_BatchedForwardKinematics.py` — batched planar revolute-chain FK (cumsum of angles → link positions); too specialized for a dedicated library .cu
- `43_ContactMapPrediction.py` — symmetrize + MLP + sigmoid on pair representations; too generic; no standalone contact-map .cu
- `44_ReparameterizationTrick.py` — Gaussian reparameterization (mu + std * eps); too simple; no standalone .cu
- `45_AxialAttention.py` — factored row-wise + column-wise attention; no standalone axial-attention .cu in public repos
- `46_SteinVariationalGradient.py` — SVGD: pairwise RBF kernel + kernel gradient; ThunderSVM's RBF kernel .cu computes K only (no grad_K); no standalone SVGD .cu
- `47_virtual_all_gather_concat_linear.py` — all-gather + concat + linear; collective + generic matmul; no standalone .cu
- `48_ParticleFilter.py` — bootstrap particle filter (predict + weight + resample); too generic; no standalone SMC .cu
- `49_sum_gelu_across_virtual_ranks.py` — sum + GeLU; too generic; no standalone .cu
- `50_HiddenMarkovForward.py` — log-space HMM forward algorithm (logsumexp recurrence); no standalone CUDA HMM .cu in public probabilistic ML repos
