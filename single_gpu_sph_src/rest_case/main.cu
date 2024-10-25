#include "equation.cuh"
#include "io.cuh"
#include "lib.cuh"

__device__ gpu_param_t d_gparam;
__device__ gpu_ptc_t d_old_gptc_data;
__device__ gpu_ptc_t d_new_gptc_data;
__device__ gpu_tmp_t d_gtmp_data;

int main(void)
{
    cpu_json_t jdata;
    cpu_param_t cparam;
    cpu_data_t cdata; 
    read_json("../input/input.json", &jdata);
    set_cpu_param(&cparam, &jdata);
    alloc_cpu_data(&cdata, &cparam);
    read_vtk(&cdata, &cparam);

    check_gpu(&cparam); 
    cudaDeviceReset();
    cudaSetDevice(cparam.gpu_id);
    gpu_param_t h_gparam;
    gpu_ptc_t h_old_gptc_data;
    gpu_ptc_t h_new_gptc_data;
    gpu_tmp_t h_gtmp_data;
    set_gpu_param(&h_gparam, &cparam);
    alloc_gpu_ptc_data(&h_old_gptc_data, &cparam);
    alloc_gpu_ptc_data(&h_new_gptc_data, &cparam);
    alloc_gpu_tmp_data(&h_gtmp_data, &cparam);
    cpu_to_gpu(&h_old_gptc_data,&cdata,&cparam);
    cudaMemcpyToSymbol(d_gparam,&h_gparam,sizeof(gpu_param_t));
    
    cpu_thread_t cthread[cparam.thread_num];
    mul_thread_creat(&cthread,&cparam);

    grid = dim3((int)((cparam.ptc_num+255)/256));
    block = dim3(256);

    for(int i=cparam.start_step;i<cparam.end_step;i++)
    {
        cuda_ptc_hash<<<grid,block>>>(d_old_gptc_data,d_gtmp_data,d_gparam);
        cuda_sort_index(h_gtmp_data,cparam);
        cuda_sort_data<<<grid,block>>>(d_old_gptc_data,d_new_gptc_data,d_gtmp_data,d_gparam);
        cuda_boundary_ns<<<grid,block>>>(d_new_gptc_data,d_gtmp_data,d_gparam);
        cuda_govering_ns<<<grid,block>>>(d_new_gptc_data,d_gtmp_data,d_gparam);

        cuda_ptc_hash<<<grid,block>>>(d_new_gptc_data,d_gtmp_data,d_gparam);
        cuda_sort_index(h_gtmp_data,cparam);
        cuda_sort_data<<<grid,block>>>(d_new_gptc_data,d_old_gptc_data,d_gtmp_data,d_gparam);
        cuda_boundary_ns<<<grid,block>>>(d_old_gptc_data,d_gtmp_data,d_gparam);
        cuda_govering_ns<<<grid,block>>>(d_old_gptc_data,d_gtmp_data,d_gparam);

        int expect_0 = 0;
        int ref_0 = expect_0;
        if(i%cparam.output_step==0)
        {
            while(ref_0 == 0)
            {

                for(int j=0;j<cparam.thread_num;j++)
                {
                    ref_0 = 0;
                    if(cthread[j].write_flag.compare_exchange_strong(ref_0,0))
                    {
                        cparam.current_step = i;
                        cthread[j].current_step = i;
                        cudaMemcpy(cthread[j].data.pos_rho,d_old_gptc_data.pos_rho,cparam.ptc_num*4*sizeof(float),cudaMemcpyDeviceToHost);
                        cudaMemcpy(cthread[j].data.vel_p,d_old_gptc_data.vel_p,cparam.ptc_num*4*sizeof(float),cudaMemcpyDeviceToHost);
                        cudaMemcpy(cthread[j].data.type,d_old_gptc_data.type,cparam.ptc_num*sizeof(int),cudaMemcpyDeviceToHost);
                        cudaMemcpy(cthread[j].data.table,d_old_gptc_data.table,cparam.ptc_num*sizeof(int),cudaMemcpyDeviceToHost);
                        cthread[j].write_flag.compare_exchange_strong(ref_0,1);
                    }
                }
            }
        }
    }    
    free_gpu_ptc_data(&h_old_gptc_data);
    free_gpu_ptc_data(&h_new_gptc_data);
    free_gpu_tmp_data(&h_gtmp_data);
    cudaDeviceReset();

    return 0;
}