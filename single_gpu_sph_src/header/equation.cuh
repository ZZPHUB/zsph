#ifndef __EQUATION_CUH__
#define __EQUATION_CUH__

#include "data_structure.cuh"
#include "std_header.cuh"


#ifdef ZSPH_DELTA
extern __global__ void cuda_govering_ns(gpu_ptc_t *tptc_data);
extern __global__ void cuda_boundary_ns(gpu_ptc_t *tptc_data);
#endif

#ifdef ZSPH_SIMPLE
extern __global__ void cuda_govering_ns(gpu_ptc_t *tptc_data);
extern __global__ void cuda_boundary_ns(gpu_ptc_t *tptc_data);

#endif

#endif