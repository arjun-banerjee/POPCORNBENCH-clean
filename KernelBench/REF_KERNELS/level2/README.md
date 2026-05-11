# Level 2 Ref Kernels

Only the kernels below were strict enough to vendor as handwritten CUDA references.

| Local kernel | Type | Vendored ref kernel(s) | Upstream |
|---|---|---|---|
| `30_SelectiveScan_Mamba.py` | selective scan | `../sources/mamba_selective_scan_fwd_fp32.cu` | `state-spaces/mamba/csrc/selective_scan/selective_scan_fwd_fp32.cu` |

Deliberately excluded here even if they had a broader family-level baseline:
- `1_DeepSeekMLALoRAExpansion.py`: expansion-only local op, not the same execution as full MLA decode
- `15_GQAKVHeadExpansionAttention.py`: broad GQA attention family match, but the checked upstream ref is paged decode-specific
- `22_FusedBallQuery3D.py`: upstream PointNet2 refs cover query/group primitives, not the local fused query-plus-max-pool execution
- `26_RoPEKVCacheUpdate.py`: the public refs decompose into RoPE plus cache-append components, not one exact local-equivalent CUDA file
- `31_FusedMLAAttention.py`: the checked upstream ref is paged MLA decode, not the same execution as the local fused latent-attention kernel
