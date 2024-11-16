#ifndef __LIB_CUH__
#define __LIB_CUH__

#include "data_structure.cuh"
#include "std_header.cuh"


//check.cu
extern void check_dt(cpu_param_t *param);
extern void check_gpu(cpu_param_t *param);
extern void check_gerr(const char *a,const int b);
extern void check_mulgerr(const char*a,const int b,cpu_param_t *cparam);
extern void check_param(cpu_param_t *cparam);
extern void check_json(cpu_json_t *cjson);

//alloc.cu
extern void alloc_cpu_input(cpu_input_t *cdata, cpu_param_t *cparam);
extern void alloc_gpu_intput(cpu_input_t *cdata, cpu_param_t *cparam);
extern void del_cpu_input(cpu_input_t *cdata);
extern void del_gpu_input(cpu_input_t *cdata,cpu_param_t *cparam);
extern void alloc_gptc_data(gpu_ptc_t *gdata,cpu_param_t *cparam);
extern void alloc_gtmp_data(gpu_tmp_t *gdata,cpu_param_t *cparam,gpu_param_t *gparam);

//init.cu
extern void set_cpu_param(cpu_param_t *param, cpu_json_t *jdata);
extern void set_gpu_param(gpu_param_t *gparam, cpu_param_t *cparam);

//mem.cu
extern void cpu_to_gpu(gpu_ptc_t *gptc_data,gpu_tmp_t *gtmp_data,cpu_input_t *ginput_dat,cpu_param_t *cparam,gpu_socket_t **gsocket);
extern void gpu_to_cpu(cpu_output_t *coutput_dat,gpu_ptc_t *gptc_dat,gpu_tmp_t *gtmp_dat,cpu_param_t *cparam,gpu_socket_t **gsocket);
extern void peer_to_peer(gpu_socket_t **gsocket,cpu_param_t *cparam);
extern void init_p2p(cpu_param_t *cparam);

//split.cu
extern void split_in_cpu(cpu_input_t *ginput_dat,cpu_input_t *cintput_dat,cpu_param_t *cparam,gpu_param_t *gparam,gpu_socket_t **gsocket);
extern __global__ void hash_split(gpu_ptc_t *gptc_dat);
extern void sort_split_index(gpu_tmp_t *gtmp_dat,cpu_param_t *cparam);
extern __global__ void sort_split_data(gpu_ptc_t *new_gptc_dat,gpu_ptc_t *old_gptc_dat);
extern __global__ void find_split_data(gpu_ptc_t *gptc_dat,gpu_socket_t *gsocket_dat);
extern void init_socket(gpu_socket_t **gsocket,cpu_param_t *cparam);

//calcu.cu
extern __global__ void hash_calcu(gpu_ptc_t *gptc_dat);
extern void sort_calcu_index(gpu_tmp_t *gtmp_dat,cpu_param_t *cparam);
extern __global__ void sort_calcu_data(gpu_ptc_t *old_gptc_dat,gpu_ptc_t *new_gptc_dat);

#endif