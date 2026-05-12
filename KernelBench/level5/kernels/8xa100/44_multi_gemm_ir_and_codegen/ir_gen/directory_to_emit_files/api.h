/* Auto Generated code - Do not edit.*/


#pragma once
#include "/scratch/adalal542/KernelBench/sources/cutlass/include//cutlass/epilogue/thread/linear_combination_leaky_relu.h"
#include "/scratch/adalal542/KernelBench/sources/cutlass/include//cutlass/epilogue/thread/linear_combination.h"
#include "auto_gen/device/FusedMultiGemmForward.h"
#include "/scratch/adalal542/KernelBench/sources/cutlass/include//cutlass/gemm/device/gemm_batched.h"
#include "/scratch/adalal542/KernelBench/sources/cutlass/include//cutlass/cutlass.h"

void FusedMultiGemmForward_turing_impl(
    int M,
    int K0,
    int Batch,
    void* A0,
    void* B0,
    void* C0,
    void* D0,
    float Epilogue0_leaky_alpha,
    void* B1,
    void* C1,
    void* D1,
    float Epilogue1_leaky_alpha,
    void* B2,
    void* C2,
    void* D2,
    float Epilogue2_leaky_alpha,
cudaStream_t stream){
using b2b_gemm = typename cutlass::gemm::device::FusedMultiGemmForward<cutlass::half_t>;
cutlass::half_t alpha0 = cutlass::half_t(1);
cutlass::half_t beta0 = cutlass::half_t(0);
cutlass::gemm::GemmCoord problem_size_0(M, 256, K0);
cutlass::half_t alpha1 = cutlass::half_t(1);
cutlass::half_t beta1 = cutlass::half_t(0);
cutlass::gemm::GemmCoord problem_size_1(M, 128, 256);
cutlass::half_t alpha2 = cutlass::half_t(1);
cutlass::half_t beta2 = cutlass::half_t(0);
cutlass::gemm::GemmCoord problem_size_2(M, 64, 128);
typename b2b_gemm::Arguments arguments{
    problem_size_0,
    problem_size_1,
    problem_size_2,
    {reinterpret_cast<cutlass::half_t*>(A0), problem_size_0.k()},
    {reinterpret_cast<cutlass::half_t*>(B0), K0},
    {reinterpret_cast<cutlass::half_t*>(C0), 256},
    {reinterpret_cast<cutlass::half_t*>(B1), 256},
    {reinterpret_cast<cutlass::half_t*>(C1), 128},
    {reinterpret_cast<cutlass::half_t*>(B2), 128},
    {reinterpret_cast<cutlass::half_t*>(C2), 64},
    {reinterpret_cast<cutlass::half_t*>(D2), problem_size_2.n()},
    { alpha0, beta0, cutlass::half_t(Epilogue0_leaky_alpha)},
    { alpha1, beta1, cutlass::half_t(Epilogue1_leaky_alpha)},
    { alpha2, beta2, cutlass::half_t(Epilogue2_leaky_alpha)},
    Batch};

    b2b_gemm gemm_op;
    gemm_op.initialize(arguments);

    gemm_op(stream);

}
void FusedMultiGemmForward_volta_impl(
    int M,
    int K0,
    int Batch,
    void* A0,
    void* B0,
    void* C0,
    void* D0,
    float Epilogue0_leaky_alpha,
    void* B1,
    void* C1,
    void* D1,
    float Epilogue1_leaky_alpha,
    void* B2,
    void* C2,
    void* D2,
    float Epilogue2_leaky_alpha,
cudaStream_t stream){
using Gemm0 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<32, 256, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombinationLeakyRelu<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    8,
    8>;

using Gemm1 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<32, 128, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombinationLeakyRelu<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    8,
    8>;

using Gemm2 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombinationLeakyRelu<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    8,
    8>;


cutlass::half_t alpha0 = cutlass::half_t(1);
cutlass::half_t beta0 = cutlass::half_t(0);
cutlass::gemm::GemmCoord problem_size_0(M, 256, K0);
typename Gemm0::Arguments arguments_0{
    problem_size_0,
    {reinterpret_cast<cutlass::half_t*>(A0), K0}, M * K0,
    {reinterpret_cast<cutlass::half_t*>(B0), K0}, 256 * K0,
    {reinterpret_cast<cutlass::half_t*>(C0), 256}, M * 256,
    {reinterpret_cast<cutlass::half_t*>(D0), 256}, M * 256,
    { alpha0, beta0, cutlass::half_t(Epilogue0_leaky_alpha) },
    Batch};
    Gemm0 gemm_op_0;
    gemm_op_0.initialize(arguments_0, nullptr);

cutlass::half_t alpha1 = cutlass::half_t(1);
cutlass::half_t beta1 = cutlass::half_t(0);
cutlass::gemm::GemmCoord problem_size_1(M, 128, 256);
typename Gemm1::Arguments arguments_1{
    problem_size_1,
    {reinterpret_cast<cutlass::half_t*>(D0), 256}, M * 256,
    {reinterpret_cast<cutlass::half_t*>(B1), 256}, 128 * 256,
    {reinterpret_cast<cutlass::half_t*>(C1), 128}, M * 128,
    {reinterpret_cast<cutlass::half_t*>(D1), 128}, M * 128,
    { alpha1, beta1, cutlass::half_t(Epilogue1_leaky_alpha) },
    Batch};
    Gemm1 gemm_op_1;
    gemm_op_1.initialize(arguments_1, nullptr);

cutlass::half_t alpha2 = cutlass::half_t(1);
cutlass::half_t beta2 = cutlass::half_t(0);
cutlass::gemm::GemmCoord problem_size_2(M, 64, 128);
typename Gemm2::Arguments arguments_2{
    problem_size_2,
    {reinterpret_cast<cutlass::half_t*>(D1), 128}, M * 128,
    {reinterpret_cast<cutlass::half_t*>(B2), 128}, 64 * 128,
    {reinterpret_cast<cutlass::half_t*>(C2), 64}, M * 64,
    {reinterpret_cast<cutlass::half_t*>(D2), 64}, M * 64,
    { alpha2, beta2, cutlass::half_t(Epilogue2_leaky_alpha) },
    Batch};
    Gemm2 gemm_op_2;
    gemm_op_2.initialize(arguments_2, nullptr);


    gemm_op_0(stream);
    gemm_op_1(stream);
    gemm_op_2(stream);

}
