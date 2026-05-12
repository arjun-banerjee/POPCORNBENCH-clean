/* Auto Generated code - Do not edit.*/
#include <cstdio> 
#include "cutlass/gemm/device/gemm_batched.h" 
#include "cutlass/cutlass.h" 
#include "../cutlass_irrelevant.h" 
#include "../cutlass_verify.h" 
#include "leaky_bias.h" 
#include "utils.h" 
int main(int args, char * argv[]) {
    int M = atoi(argv[1]);
    int K0 = 15000;
    if(args == 3);
        K0 = atoi(argv[2]);
    int B = 1;
    if(args == 4);
        B = atoi(argv[3]);
    srand(1234UL);
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int sm = prop.major *10 + prop.minor;
using ElementCompute = cutlass::half_t;
    ElementCompute alpha0 = ElementCompute(1);
    ElementCompute beta0 = ElementCompute(0);
    ElementCompute alpha1 = ElementCompute(1);
    ElementCompute beta1 = ElementCompute(0);
    ElementCompute alpha2 = ElementCompute(1);
    ElementCompute beta2 = ElementCompute(0);
    size_t flops = 0;
    flops += size_t(2) * size_t(M) * size_t(B) * size_t(256) * size_t(K0);
    cutlass::gemm::GemmCoord problem_size_0(M, 256, K0);
    memory_unit<cutlass::half_t> Mat_A0(B * problem_size_0.m() * problem_size_0.k());
    memory_unit<cutlass::half_t> Mat_B0(B * problem_size_0.n() * problem_size_0.k());
    memory_unit<cutlass::half_t> Mat_C0(B * M * 256);
    memory_unit<cutlass::half_t> Mat_D_cutlass_ref0(B * problem_size_0.m() * problem_size_0.n());
    Mat_A0.init();
    Mat_B0.init();
    Mat_C0.init();
    flops += size_t(2) * size_t(M) * size_t(B) * size_t(128) * size_t(256);
    cutlass::gemm::GemmCoord problem_size_1(M, 128, 256);
    memory_unit<cutlass::half_t> Mat_A1(B * problem_size_1.m() * problem_size_1.k());
    memory_unit<cutlass::half_t> Mat_B1(B * problem_size_1.n() * problem_size_1.k());
    memory_unit<cutlass::half_t> Mat_C1(B * M * 128);
    memory_unit<cutlass::half_t> Mat_D_cutlass_ref1(B * problem_size_1.m() * problem_size_1.n());
    Mat_A1.init();
    Mat_B1.init();
    Mat_C1.init();
    flops += size_t(2) * size_t(M) * size_t(B) * size_t(64) * size_t(128);
    cutlass::gemm::GemmCoord problem_size_2(M, 64, 128);
    memory_unit<cutlass::half_t> Mat_A2(B * problem_size_2.m() * problem_size_2.k());
    memory_unit<cutlass::half_t> Mat_B2(B * problem_size_2.n() * problem_size_2.k());
    memory_unit<cutlass::half_t> Mat_C2(B * M * 64);
    memory_unit<cutlass::half_t> Mat_D_cutlass_ref2(B * problem_size_2.m() * problem_size_2.n());
    Mat_A2.init();
    Mat_B2.init();
    Mat_C2.init();
    memory_unit<cutlass::half_t> Mat_D2(B * problem_size_2.m() * problem_size_2.n());
    Param arguments = {
        M,
        K0,
        B,
        reinterpret_cast<const void*>(Mat_A0.device_ptr),
        reinterpret_cast<const void*>(Mat_B0.device_ptr),
        reinterpret_cast<const void*>(NULL),
        cutlass::half_t(1.3),
        reinterpret_cast<void*>(Mat_D_cutlass_ref0.device_ptr),
        reinterpret_cast<const void*>(Mat_B1.device_ptr),
        reinterpret_cast<const void*>(NULL),
        cutlass::half_t(1.3),
        reinterpret_cast<void*>(Mat_D_cutlass_ref1.device_ptr),
        reinterpret_cast<const void*>(Mat_B2.device_ptr),
        reinterpret_cast<const void*>(NULL),
        cutlass::half_t(1.3),
        reinterpret_cast<void*>(Mat_D2.device_ptr)};
    TI(FUSED_CUTLASS);
    for(int i = 0; i < 100; i++){
        one_api(arguments, sm, NULL);
    }
    TO(FUSED_CUTLASS, "FUSED_CUTLASS", 100);

    typename Gemm0::Arguments arguments_0{
        problem_size_0,
        {reinterpret_cast<cutlass::half_t*>(Mat_A0.device_ptr), K0}, M * K0,
        {reinterpret_cast<cutlass::half_t*>(Mat_B0.device_ptr), K0}, 256 * K0,
        {reinterpret_cast<cutlass::half_t*>(Mat_C0.device_ptr), 256}, M * 256,
        {reinterpret_cast<cutlass::half_t*>(Mat_D_cutlass_ref0.device_ptr), 256}, M * 256,
        { alpha0, beta0, cutlass::half_t(1.3)     },
        B};
    typename Gemm1::Arguments arguments_1{
        problem_size_1,
        {reinterpret_cast<cutlass::half_t*>(Mat_D_cutlass_ref0.device_ptr), 256}, M * 256,
        {reinterpret_cast<cutlass::half_t*>(Mat_B1.device_ptr), 256}, 128 * 256,
        {reinterpret_cast<cutlass::half_t*>(Mat_C1.device_ptr), 128}, M * 128,
        {reinterpret_cast<cutlass::half_t*>(Mat_D_cutlass_ref1.device_ptr), 128}, M * 128,
        { alpha1, beta1, cutlass::half_t(1.3)     },
        B};
    typename Gemm2::Arguments arguments_2{
        problem_size_2,
        {reinterpret_cast<cutlass::half_t*>(Mat_D_cutlass_ref1.device_ptr), 128}, M * 128,
        {reinterpret_cast<cutlass::half_t*>(Mat_B2.device_ptr), 128}, 64 * 128,
        {reinterpret_cast<cutlass::half_t*>(Mat_C2.device_ptr), 64}, M * 64,
        {reinterpret_cast<cutlass::half_t*>(Mat_D_cutlass_ref2.device_ptr), 64}, M * 64,
        { alpha2, beta2, cutlass::half_t(1.3)     },
        B};
    TI(UNFUSED_CUTLASS);
    for(int i = 0; i < 100; i++){
        FusedMultiGemmForward_verify(
            arguments_0,
            arguments_1,
            arguments_2,
            NULL);
    }
    TO(UNFUSED_CUTLASS, "UNFUSED_CUTLASS", 100);
    Mat_D_cutlass_ref2.d2h();
    Mat_D2.d2h();
    check_result(Mat_D_cutlass_ref2.host_ptr, Mat_D2.host_ptr, Mat_D2.elements);


}
