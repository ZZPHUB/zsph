#ifndef __LIB_CUH__
#define __LIB_CUH__

#include "data_structure.cuh"
#include "std_header.cuh"
//hash.cu
extern __global__ void cuda_ptc_hash(gpu_ptc_t *tptc_dat);

//sort.cu
extern __global__ void cuda_sort_data(gpu_ptc_t *tdata_old,gpu_ptc_t *tdata_new);
extern  void cuda_sort_index(gpu_tmp_t data,cpu_param_t par);

//init.cu   
extern void set_cpu_param(cpu_param_t *param, cpu_json_t *jdata);
extern void set_gpu_param(gpu_param_t *gparam, cpu_param_t *cparam);

extern void alloc_cpu_data(cpu_input_t *cdata,cpu_param_t *param);
extern void delete_cpu_data(cpu_input_t *cdata);
extern void alloc_gpu_ptc_data(gpu_ptc_t *gptc_data,cpu_param_t *param);
extern void delete_gpu_ptc_data(gpu_ptc_t *gptc_data);
extern void alloc_gpu_tmp_data(gpu_tmp_t *gtmp_data,cpu_param_t *param);
extern void delete_gpu_tmp_data(gpu_tmp_t *gtmp_data);

extern void cpu_to_gpu(gpu_ptc_t *gptc_data,gpu_tmp_t *gtmp_data,cpu_input_t *cdata,cpu_param_t *param);
extern void gpu_to_cpu(cpu_output_t *data,gpu_ptc_t *gptc_data,gpu_tmp_t *gtmp_data,cpu_param_t *param);

//check.cu
extern void check_dt(cpu_param_t *param);
extern void check_gpu(cpu_param_t *param);
extern void check_gerr(const char *a,const int b);
extern void check_param(cpu_param_t *cparam);
extern void check_json(cpu_json_t *cjson);

//time_int.cu
extern __global__ void cuda_prediction(gpu_ptc_t *tptc_data);
extern __global__ void cuda_correction(gpu_ptc_t *tptc_data);

#endif