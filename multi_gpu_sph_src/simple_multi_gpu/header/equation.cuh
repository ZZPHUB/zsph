#ifndef __EQUATION_CUH__
#define __EQUATION_CUH__

#include "std_header.cuh"
#include "data_structure.cuh"

extern __global__ void cuda_governing_ns(gpu_ptc_t *gptc_dat);
extern __global__ void cuda_prediction(gpu_ptc_t *tptc_data);
extern __global__ void cuda_correction(gpu_ptc_t *tptc_data);

#endif