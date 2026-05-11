# Level 2 Ref Kernels

Only the kernels below were strict enough to vendor as handwritten CUDA references.

| Local kernel | Type | Vendored ref kernel(s) | Upstream |
|---|---|---|---|
| `26_RoPEKVCacheUpdate.py` | RoPE + KV-cache append | `../sources/flashinfer_page.cu`, `../sources/flashinfer_rope.cu` | `flashinfer-ai/flashinfer/csrc/{page,rope}.cu` |
| `30_SelectiveScan_Mamba.py` | selective scan | `../sources/mamba_selective_scan_fwd_fp32.cu` | `state-spaces/mamba/csrc/selective_scan/selective_scan_fwd_fp32.cu` |
| `31_FusedMLAAttention.py` | MLA attention | `../sources/flashinfer_batch_decode_mla_cute_sm80.cu` | `flashinfer-ai/flashinfer/csrc/batch_decode_mla_cute_sm80.cu` |

Deliberately excluded here even if they had a broader family-level baseline:
- `1_DeepSeekMLALoRAExpansion.py`: expansion-only local op, not the same execution as full MLA decode
- `15_GQAKVHeadExpansionAttention.py`: broad GQA attention family match, but the checked upstream ref is paged decode-specific
- `22_FusedBallQuery3D.py`: upstream PointNet2 refs cover query/group primitives, not the local fused query-plus-max-pool execution
