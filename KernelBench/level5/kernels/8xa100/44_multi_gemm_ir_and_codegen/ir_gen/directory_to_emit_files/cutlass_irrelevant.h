/* Auto Generated code - Do not edit.*/


#pragma once
#include <cuda_runtime.h>
#include <cassert>
struct Fused3xGemm_256_128_64_Params{
    int M;
    int K0;
    int Batch;
    const void* A0;
    const void* B0;
    const void* C0;
    float Epilogue0_leaky_alpha;
    void* D0;
    const void* B1;
    const void* C1;
    float Epilogue1_leaky_alpha;
    void* D1;
    const void* B2;
    const void* C2;
    float Epilogue2_leaky_alpha;
    void* D2;
}; // struct Fused3xGemm_256_128_64_Params
using Param = Fused3xGemm_256_128_64_Params;
void one_api( const  Param & param, int sm, cudaStream_t stream);
