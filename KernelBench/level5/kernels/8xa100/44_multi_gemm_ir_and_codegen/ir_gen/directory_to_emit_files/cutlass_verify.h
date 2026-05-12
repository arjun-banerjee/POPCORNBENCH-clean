/* Auto Generated code - Do not edit.*/


#pragma once
#include "/scratch/adalal542/KernelBench/sources/cutlass/include//cutlass/epilogue/thread/linear_combination_leaky_relu.h"
#include "/scratch/adalal542/KernelBench/sources/cutlass/include//cutlass/epilogue/thread/linear_combination.h"
#include "auto_gen/device/FusedMultiGemmForward.h"
#include "/scratch/adalal542/KernelBench/sources/cutlass/include//cutlass/gemm/device/gemm_batched.h"
#include "/scratch/adalal542/KernelBench/sources/cutlass/include//cutlass/cutlass.h"
using Gemm0 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<32, 256, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
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
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<32, 128, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
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
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationLeakyRelu<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    8,
    8>;


void FusedMultiGemmForward_verify(
    typename Gemm0::Arguments Arguments_0,
    typename Gemm1::Arguments Arguments_1,
    typename Gemm2::Arguments Arguments_2,
cudaStream_t stream){
    Gemm0 gemm_op_0;
    gemm_op_0.initialize(Arguments_0, nullptr);
    Gemm1 gemm_op_1;
    gemm_op_1.initialize(Arguments_1, nullptr);
    Gemm2 gemm_op_2;
    gemm_op_2.initialize(Arguments_2, nullptr);
    gemm_op_0(stream);
    gemm_op_1(stream);
    gemm_op_2(stream);

}
