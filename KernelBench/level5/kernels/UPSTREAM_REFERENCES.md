# Level 5 Upstream References

This file records concrete upstream kernel sources that match the curated `KernelBench/level5/kernels` collection.

The current local A100 slice already covers:

- `paged_attention`
- `fused_rmsnorm`
- `swiglu_activation`
- `rotary_embedding`
- `custom_allreduce`
- `quantized_gemm`
- `flash_attention`

The sources below are the next public repositories worth mining for additional kernels in the same style.

## FlashInfer

FlashInfer is the strongest upstream source here for new A100-oriented serving kernels. Its repository explicitly supports Ampere / `SM 8.0`, and its `csrc/` tree contains both generic kernels and `sm80`-specific paths.

| Local family | Upstream file | Why it matters | A100 fit |
| --- | --- | --- | --- |
| paged decode attention | `flashinfer-ai/flashinfer/csrc/batch_decode.cu` | Paged KV-cache decode path with planning + run entrypoints | strong |
| paged / ragged prefill attention | `flashinfer-ai/flashinfer/csrc/batch_prefill.cu` | Prefill-side counterpart to decode kernels | strong |
| paged KV-cache append | `flashinfer-ai/flashinfer/csrc/flashinfer_page_binding.cu` | KV-cache update / append path that complements paged attention | strong |
| RoPE + KV append | `flashinfer-ai/flashinfer/csrc/flashinfer_rope_binding.cu` | RoPE application, Llama 3.1 RoPE, quantize + append paged KV | strong |
| fused RMSNorm + SiLU | `flashinfer-ai/flashinfer/csrc/flashinfer_rmsnorm_silu_binding.cu` | Useful adjacent op family to local RMSNorm / SwiGLU kernels | medium |
| MLA paged decode, Ampere-specific | `flashinfer-ai/flashinfer/csrc/batch_decode_mla_cute_sm80.cu` | One of the few explicit `sm80` files in the repo | strong |

Recommended local synthesis targets if you expand the A100 set:

1. `11_flashinfer_batch_decode.cu` added
2. `12_flashinfer_page_append.cu` added
3. `13_flashinfer_rope_quantize_append.cu` added
4. `14_flashinfer_mla_decode_sm80.cu` added
5. `15_flashinfer_rmsnorm_silu.cu` added

Notes:

- `batch_decode_mla_cute_sm80.cu` is especially useful because it is already specialized for Ampere.
- FlashInfer also has `batch_attention.cu`, `batch_prefill.cu`, `batch_pod.cu`, and additional JIT templates under `csrc/` if you want a larger corpus rather than a minimal curated slice.

## ThunderKittens

ThunderKittens is a useful kernel source, but not a primary A100 source anymore. Its current repo states that Ampere is no longer actively supported, while most prebuilt attention / GEMM kernels target H100 or B200.

| Local family | Upstream file | Why it matters | A100 fit |
| --- | --- | --- | --- |
| layer norm | `HazyResearch/ThunderKittens/kernels/layernorm/layernorm.cu` | Standalone fused layernorm-style kernel with PyTorch extension harness | weak to medium |
| rotary embedding | `HazyResearch/ThunderKittens/kernels/rotary/rotary.cu` | Standalone rotary kernel with BF16 support for head dims 64 / 128 | weak to medium |
| attention | `HazyResearch/ThunderKittens/kernels/attention/mha_h100/*` | Useful as design reference only; current implementation is H100-specific | weak |
| GEMM | `HazyResearch/ThunderKittens/kernels/gemm/bf16_h100/*` | Good reference for scheduling ideas, not an A100 drop-in source | weak |

Recommended use of ThunderKittens in this repo:

1. Use `rotary.cu` and `layernorm.cu` as algorithmic references, not as direct A100 benchmark kernels.
2. Do not treat `mha_h100` or `bf16_h100` as A100 kernels without explicit retargeting work.
3. Keep ThunderKittens separate from the A100 kernel set unless you introduce an `experimental/` or `ported/` category.

## Practical Guidance

If the goal is "KernelBench for A100-optimized kernels", the ranking is:

1. FlashInfer
2. vLLM
3. FlashAttention-2
4. Marlin / Sparse-Marlin
5. xFormers / CUTLASS
6. ThunderKittens

ThunderKittens is still valuable, but mainly as a reference for kernel structure and scheduling patterns. FlashInfer is the better source for kernels you can justify placing alongside the current local A100 set.

## FlashAttention

FlashAttention is the strongest upstream source for compact exact-attention kernel instantiations. It aligns directly with the existing `09` / `10` slots in the compact `a100` and `h100` sets.

Added compact extensions:

### A100 / FlashAttention-2 (`sm80`)

1. `16_flash_attn2_fwd_hdim64_fp16_sm80.cu` added
2. `17_flash_attn2_bwd_hdim64_fp16_sm80.cu` added
3. `18_flash_attn2_fwd_hdim128_bf16_sm80.cu` added
4. `19_flash_attn2_bwd_hdim128_bf16_sm80.cu` added
5. `20_flash_attn2_fwd_hdim192_fp16_sm80.cu` added
6. `21_flash_attn2_bwd_hdim192_fp16_sm80.cu` added

### H100 / FlashAttention-3 (`sm90`)

1. `11_flash_attn3_fwd_hdim128_fp16_sm90.cu` added
2. `12_flash_attn3_bwd_hdim128_fp16_sm90.cu` added
3. `13_flash_attn3_fwd_hdim128_bf16_paged_split_softcap_sm90.cu` added
4. `14_flash_attn3_fwd_hdim192_fp16_paged_softcap_sm90.cu` added
