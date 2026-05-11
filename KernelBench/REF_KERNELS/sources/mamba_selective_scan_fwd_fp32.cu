// PopcornBench reference mapping:
// - level2/popcorn/30_SelectiveScan_Mamba.py
// Upstream source:
// - https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_fp32.cu
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan_fwd_kernel.cuh"

template void selective_scan_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);
