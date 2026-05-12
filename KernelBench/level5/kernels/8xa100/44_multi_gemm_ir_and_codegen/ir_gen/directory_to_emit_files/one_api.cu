/* Auto Generated code - Do not edit.*/
#include "cutlass_irrelevant.h"
#include "api.h"
void one_api( const  Param & param, int sm, cudaStream_t stream) {
    if (sm == 70) 
        FusedMultiGemmForward_volta_impl(param.M, param.K0, param.Batch, const_cast<void*>(param.A0), const_cast<void*>(param.B0), const_cast<void*>(param.C0), param.D0, param.Epilogue0_leaky_alpha, const_cast<void*>(param.B1), const_cast<void*>(param.C1), param.D1, param.Epilogue1_leaky_alpha, const_cast<void*>(param.B2), const_cast<void*>(param.C2), param.D2, param.Epilogue2_leaky_alpha, stream);
    else if(sm >= 75) 
        FusedMultiGemmForward_turing_impl(param.M, param.K0, param.Batch, const_cast<void*>(param.A0), const_cast<void*>(param.B0), const_cast<void*>(param.C0), param.D0, param.Epilogue0_leaky_alpha, const_cast<void*>(param.B1), const_cast<void*>(param.C1), param.D1, param.Epilogue1_leaky_alpha, const_cast<void*>(param.B2), const_cast<void*>(param.C2), param.D2, param.Epilogue2_leaky_alpha, stream);
    else assert(0);
}
