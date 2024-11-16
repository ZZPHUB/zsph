#include "lib.cuh"

void alloc_cpu_input(cpu_input_t *cdata, cpu_param_t *cparam)
{
    cdata->acc_drhodt = new float[cparam->ptc_num*4];
    cdata->pos = new float[cparam->ptc_num*3];
    cdata->vel = new float[cparam->ptc_num*3];
    cdata->rhop = new float[cparam->ptc_num*2];
    cdata->type = new int[cparam->ptc_num];
    cdata->table = new int[cparam->ptc_num];
}

void alloc_gpu_intput(cpu_input_t *cdata, cpu_param_t *cparam)
{
    for (int i = 0; i < cparam->gpu_num; i++)
    {
        cdata[i].acc_drhodt = new float[cparam->gmax_size*4];
        cdata[i].pos = new float[cparam->gmax_size*4];
        cdata[i].vel = new float[cparam->gmax_size*4];
        cdata[i].rhop = new float[cparam->gmax_size*2];
        cdata[i].type = new int[cparam->gmax_size];
        cdata[i].table = new int[cparam->gmax_size];
        cdata[i].is_ptc = new int[cparam->gmax_size];
        cdata[i].gpu_id = new int[cparam->gmax_size];
        memset(cdata[i].is_ptc,0,sizeof(int)*cparam->gmax_size);
    }
}

void del_cpu_input(cpu_input_t *cdata)
{
    free(cdata->acc_drhodt);
    free(cdata->pos);
    free(cdata->vel);
    free(cdata->rhop);
    free(cdata->type);
    free(cdata->table);
}

void del_gpu_input(cpu_input_t *cdata,cpu_param_t *cparam)
{
    for(int i=0;i<cparam->gpu_num;i++)
    {
        free(cdata[i].acc_drhodt);
        free(cdata[i].pos);
        free(cdata[i].vel);
        free(cdata[i].rhop);
        free(cdata[i].type);
        free(cdata[i].table);
        free(cdata[i].is_ptc);
        free(cdata[i].gpu_id);

    }
}

void alloc_gptc_data(gpu_ptc_t *gdata,cpu_param_t *cparam)
{
    for(int i=0;i<cparam->gpu_num;i++)
    {
        cudaSetDevice(i);
        cudaMalloc(&(gdata[i].pos),sizeof(float3)*cparam->gmax_size);
        cudaMalloc(&(gdata[i].vel),sizeof(float3)*cparam->gmax_size);
        cudaMalloc(&(gdata[i].tmp_pos),sizeof(float3)*cparam->gmax_size);
        cudaMalloc(&(gdata[i].tmp_vel),sizeof(float3)*cparam->gmax_size);
        cudaMalloc(&(gdata[i].rhop),sizeof(float2)*cparam->gmax_size);
        cudaMalloc(&(gdata[i].tmp_rhop),sizeof(float2)*cparam->gmax_size);
        cudaMalloc(&(gdata[i].type),sizeof(int)*cparam->gmax_size);
        cudaMalloc(&(gdata[i].table),sizeof(int)*cparam->gmax_size);
        //
        cudaMalloc(&(gdata[i].is_ptc),sizeof(int)*cparam->gmax_size);
        cudaMalloc(&(gdata[i].gpu_id),sizeof(int)*cparam->gmax_size);

        cudaMemset(gdata[i].pos,0,sizeof(float3)*cparam->gmax_size);
        cudaMemset(gdata[i].tmp_pos,0,sizeof(float3)*cparam->gmax_size);
        cudaMemset(gdata[i].vel,0,sizeof(float3)*cparam->gmax_size);
        cudaMemset(gdata[i].tmp_vel,0,sizeof(float3)*cparam->gmax_size);
        cudaMemset(gdata[i].rhop,0,sizeof(float2)*cparam->gmax_size);
        cudaMemset(gdata[i].tmp_rhop,0,sizeof(float2)*cparam->gmax_size);
        cudaMemset(gdata[i].type,0,sizeof(int)*cparam->gmax_size);
        cudaMemset(gdata[i].table,0,sizeof(int)*cparam->gmax_size);
        //
        cudaMemset(gdata[i].gpu_id,0,sizeof(int)*cparam->gmax_size);
        cudaMemset(gdata[i].is_ptc,0,sizeof(int)*cparam->gmax_size);
        check_gerr(__FILE__,__LINE__);
    }
}

void alloc_gtmp_data(gpu_tmp_t *gdata,cpu_param_t *cparam,gpu_param_t *gparam)
{
    for(int i=0;i<cparam->gpu_num;i++)
    {
        cudaSetDevice(i);
        cudaMalloc(&(gdata[i].acc_drhodt),sizeof(float4)*gparam[i].gmax_size);
        cudaMalloc(&(gdata[i].dofv),sizeof(float)*gparam[i].gmax_size);
        cudaMalloc(&(gdata[i].index),sizeof(int)*gparam[i].gmax_size);
        cudaMalloc(&(gdata[i].hash),sizeof(int)*gparam[i].gmax_size);
        cudaMalloc(&(gdata[i].grid_start),sizeof(int)*gparam[i].grid_hash_max);
        cudaMalloc(&(gdata[i].grid_end),sizeof(int)*gparam[i].grid_hash_max);
        cudaMalloc(&(gdata[i].wsum),sizeof(float)*gparam[i].gmax_size);

        cudaMemset(gdata[i].acc_drhodt,0,sizeof(float4)*gparam[i].gmax_size);
        cudaMemset(gdata[i].dofv,0,sizeof(float)*gparam[i].gmax_size);
        cudaMemset(gdata[i].index,0,sizeof(int)*gparam[i].gmax_size);
        cudaMemset(gdata[i].hash,0,sizeof(int)*gparam[i].gmax_size);
        cudaMemset(gdata[i].grid_start,0,sizeof(int)*(gparam[i].grid_hash_max));
        cudaMemset(gdata[i].grid_end,0,sizeof(int)*(gparam[i].grid_hash_max));
        cudaMemset(gdata[i].wsum,0,sizeof(float)*gparam[i].gmax_size);
        check_gerr(__FILE__,__LINE__);
    }
}