#include "lib.cuh"

void gpu_to_cpu(cpu_output_t *data,gpu_ptc_t *gptc_data,gpu_tmp_t *gtmp_data,cpu_param_t *param)
{
    cudaMemcpy(data->pos,gptc_data->pos,sizeof(float3)*param->ptc_num,cudaMemcpyDeviceToHost);
    //vel
    #ifdef ZSPH_OUTPUT_VEL
    cudaMemcpy(data->vel,gptc_data->vel,sizeof(float3)*param->ptc_num,cudaMemcpyDeviceToHost);
    #endif

    //rho and p
    #if (defined ZSPH_OUTPUT_RHO) || (defined ZSPH_OUTPUT_P)
    cudaMemcpy(data->rhop,gptc_data->rhop,sizeof(float2)*param->ptc_num,cudaMemcpyDeviceToHost);
    #endif

    //type
    #ifdef ZSPH_OUTPUT_TYPE 
    cudaMemcpy(data->type,gptc_data->type,sizeof(int)*param->ptc_num,cudaMemcpyDeviceToHost);
    #endif

    //hash
    #ifdef ZSPH_OUTPUT_HASH
    cudaMemcpy(data->hash,gtmp_data->hash,sizeof(int)*param->ptc_num,cudaMemcpyDeviceToHost);
    #endif

    //wsum
    #ifdef ZSPH_OUTPUT_WSUM
    cudaMemcpy(data->wsum,gtmp_data->wsum,sizeof(float)*param->ptc_num,cudaMemcpyDeviceToHost);
    #endif

    //acc and drhodt
    #if (defined ZSPH_OUTPUT_ACC) || (defined ZSPH_OUTPUT_DRHODT)
    cudaMemcpy(data->acc_drhodt,gtmp_data->acc_drhodt,sizeof(float4)*param->ptc_num,cudaMemcpyDeviceToHost);
    #endif
    check_gerr(__FILE__,__LINE__);
}