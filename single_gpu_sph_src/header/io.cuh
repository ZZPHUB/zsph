#ifndef __IO_CUH__
#define __IO_CUH__



#include "std_header.cuh"
#include "data_structure.cuh"

extern void read_vtk(cpu_input_t *data, cpu_param_t *param);
extern void write_vtk(std::string file_name, cpu_output_t *data, cpu_param_t *param);

extern void mul_thread_creat(cpu_thread_t *thread_pool,cpu_param_t *param);
extern void mul_thread_output(cpu_thread_t *thread,cpu_param_t *param);

extern void read_json(std::string file_name, cpu_json_t *json_data);
extern void write_json(cpu_param_t *param, cpu_json_t *jdata);

#endif