#include "lib.cuh"

void alloc_cpu_data(cpu_input_t *cdata,cpu_param_t *param)
{
    /*
    cdata->pos_rho = new float[param->ptc_num*4];
    cdata->vel_p = new float[param->ptc_num*4];
    cdata->acc_drhodt = new float[param->ptc_num*4];
    cdata->type = new int[param->ptc_num];
    cdata->table = new int[param->ptc_num];
    */
    cdata->pos = (float *)calloc(3*param->ptc_num,sizeof(float));
    cdata->vel= (float *)calloc(3*param->ptc_num,sizeof(float));
    cdata->rhop = (float *)calloc(2*param->ptc_num,sizeof(float));
    cdata->acc_drhodt = (float *)calloc(4*param->ptc_num,sizeof(float));
    cdata->type = (int *)calloc(param->ptc_num,sizeof(int));
    cdata->table = (int *)calloc(param->ptc_num,sizeof(int));

}

void delete_cpu_data(cpu_input_t *cdata)
{
    /*
    delete [] cdata->pos_rho;
    delete [] cdata->vel_p;
    delete [] cdata->acc_drhodt;
    delete [] cdata->type;
    delete [] cdata->table;
    */
   free(cdata->pos);
   free(cdata->vel);
   free(cdata->rhop);
   free(cdata->acc_drhodt);
   free(cdata->type);
   free(cdata->table);
}

void alloc_gpu_ptc_data(gpu_ptc_t *gptc_data,cpu_param_t *param)
{
    cudaMalloc(&(gptc_data->pos),param->ptc_num*sizeof(float3));
    cudaMalloc(&(gptc_data->vel),param->ptc_num*sizeof(float3));

    cudaMalloc(&(gptc_data->tmp_pos),param->ptc_num*sizeof(float3));
    cudaMalloc(&(gptc_data->tmp_vel),param->ptc_num*sizeof(float3));

    cudaMalloc(&(gptc_data->rhop),param->ptc_num*sizeof(float2));
    cudaMalloc(&(gptc_data->tmp_rhop),param->ptc_num*sizeof(float2));

    cudaMalloc(&(gptc_data->type),param->ptc_num*sizeof(int));
    cudaMalloc(&(gptc_data->table),param->ptc_num*sizeof(int));

    //mem set to zero
    cudaMemset(gptc_data->pos,0,param->ptc_num*sizeof(float3));
    cudaMemset(gptc_data->vel,0,param->ptc_num*sizeof(float3));

    cudaMemset(gptc_data->tmp_pos,0,param->ptc_num*sizeof(float3));
    cudaMemset(gptc_data->tmp_vel,0,param->ptc_num*sizeof(float3));

    cudaMemset(gptc_data->rhop,0,param->ptc_num*sizeof(float2));
    cudaMemset(gptc_data->tmp_rhop,0,param->ptc_num*sizeof(float2));

    cudaMemset(gptc_data->type,0,param->ptc_num*sizeof(int));
    cudaMemset(gptc_data->type,0,param->ptc_num*sizeof(int));
    check_gerr(__FILE__,__LINE__);
    //check_gerr();
}

void delete_gpu_ptc_data(gpu_ptc_t *gptc_data)
{
    cudaFree(gptc_data->pos);
    cudaFree(gptc_data->vel);
    cudaFree(gptc_data->tmp_pos);
    cudaFree(gptc_data->tmp_vel);
    cudaFree(gptc_data->rhop);
    cudaFree(gptc_data->tmp_rhop);
    cudaFree(gptc_data->type);
    cudaFree(gptc_data->table);
    check_gerr(__FILE__,__LINE__);
    //check_gerr();
}

void alloc_gpu_tmp_data(gpu_tmp_t *gtmp_data,cpu_param_t *param)
{
    //tmp data
    cudaMalloc(&(gtmp_data->acc_drhodt),param->ptc_num*sizeof(float4));
    #ifdef ZSPH_DELTA
    cudaMalloc(&(gtmp_data->dofv_grandrho),param->ptc_num*sizeof(float4));
    cudaMemset(gtmp_data->dofv_grandrho,0,param->ptc_num*sizeof(float4));
    #endif
    #ifdef ZSPH_SIMPLE
    cudaMalloc(&(gtmp_data->dofv),param->ptc_num*sizeof(float));
    cudaMemset(gtmp_data->dofv,0,param->ptc_num*sizeof(float));
    #endif

    //hash and index
    cudaMalloc(&(gtmp_data->hash),param->ptc_num*sizeof(int));
    cudaMalloc(&(gtmp_data->index),param->ptc_num*sizeof(int));
    
    //grid
    cudaMalloc(&(gtmp_data->grid_end),param->grid_num*sizeof(int));
    cudaMalloc(&(gtmp_data->grid_start),param->grid_num*sizeof(int));

    //mem set to zero for safety
    cudaMemset(gtmp_data->acc_drhodt,0,param->ptc_num*sizeof(float4));

    //hash and index
    cudaMemset(gtmp_data->hash,0,param->ptc_num*sizeof(int));
    cudaMemset(gtmp_data->index,0,param->ptc_num*sizeof(int));
    //grid
    cudaMemset(gtmp_data->grid_end,0,param->grid_num*sizeof(int));
    cudaMemset(gtmp_data->grid_start,0,param->grid_num*sizeof(int));

    check_gerr(__FILE__,__LINE__);
    //check_gerr();
}

void delete_gpu_tmp_data(gpu_tmp_t *gtmp_data)
{
    cudaFree(gtmp_data->acc_drhodt);

    #ifdef ZSPH_DLETA
    cudaFree(gtmp_data->dofv_grandrho);
    #endif

    #ifdef ZSPH_SIMPLE
    cudaFree(gtmp_data->dofv);
    #endif
    cudaFree(gtmp_data->hash);
    cudaFree(gtmp_data->index);
    cudaFree(gtmp_data->grid_end);
    cudaFree(gtmp_data->grid_start);
    check_gerr(__FILE__,__LINE__);
    //check_gerr();
}

void cpu_to_gpu(gpu_ptc_t *gptc_data,gpu_tmp_t *gtmp_data,cpu_input_t *cdata,cpu_param_t *param)
{

    //copy ptc data from cpu to gpu
    cudaMemcpy(gptc_data->pos,cdata->pos,param->ptc_num*sizeof(float3),cudaMemcpyHostToDevice);
    cudaMemcpy(gptc_data->vel,cdata->vel,param->ptc_num*sizeof(float3),cudaMemcpyHostToDevice);
    cudaMemcpy(gptc_data->rhop,cdata->rhop,param->ptc_num*sizeof(float2),cudaMemcpyHostToDevice);

    cudaMemcpy(gptc_data->type,cdata->type,param->ptc_num*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gptc_data->table,cdata->table,param->ptc_num*sizeof(int),cudaMemcpyHostToDevice);

    //copy tmp data from cpu to gpu where tmp data may be setting as a initial value
    cudaMemcpy(gtmp_data->acc_drhodt,cdata->acc_drhodt,param->ptc_num*sizeof(float4),cudaMemcpyHostToDevice);
    //check cuda error
    check_gerr(__FILE__,__LINE__);
}

void set_cpu_param(cpu_param_t *param, cpu_json_t *jdata)
{
    //computional setting
    param->g = jdata->g;
    param->m = jdata->rho0 * jdata->dx * jdata->dx * jdata->dx;
    param->rho0 = jdata->rho0;
    param->dx = jdata->dx;
    param->h = jdata->h_factor * jdata->dx;
    param->r = jdata->r_factor * jdata->h_factor * jdata->dx;
    param->eta = jdata->eta_factor * jdata->h_factor * jdata->dx;
    param->cs = jdata->cs_factor * sqrt(abs(jdata->g) * (jdata->zmax - jdata->zmin));
    param->delta = jdata->delta;
    param->alpha = jdata->alpha;
    param->adh = 21.0f/(16.0f*3.1415926f*param->h*param->h*param->h);

    //domain setting
    param->xmin = jdata->xmin;
    param->xmax = jdata->xmax;
    param->ymin = jdata->ymin;
    param->ymax = jdata->ymax;
    param->zmin = jdata->zmin;
    param->zmax = jdata->zmax;

    //grid setting
    param->grid_size = jdata->r_factor *jdata->h_factor * jdata->dx / jdata->grid_size_factor;
    param->grid_factor = (int)jdata->grid_size_factor;
    param->grid_xmin = jdata->xmin - param->grid_size * jdata->grid_layer_factor;
    param->grid_xmax = jdata->xmax + param->grid_size * jdata->grid_layer_factor;
    param->grid_ymin = jdata->ymin - param->grid_size * jdata->grid_layer_factor;
    param->grid_ymax = jdata->ymax + param->grid_size * jdata->grid_layer_factor;
    param->grid_zmin = jdata->zmin - param->grid_size * jdata->grid_layer_factor; 
    param->grid_zmax = jdata->zmax + param->grid_size * jdata->grid_layer_factor;
    param->grid_xdim = (int)((param->grid_xmax - param->grid_xmin) / param->grid_size);
    param->grid_ydim = (int)((param->grid_ymax - param->grid_ymin) / param->grid_size);
    param->grid_zdim = (int)((param->grid_zmax - param->grid_zmin) / param->grid_size);
    param->grid_num = param->grid_xdim * param->grid_ydim * param->grid_zdim;
    param->grid_hash_min = 0;
    param->grid_hash_max = param->grid_num - 1;

    //ptc setting
    param->ptc_num = jdata->ptc_num;
    param->water_ptc_num = jdata->water_ptc_num;
    param->air_ptc_num = jdata->air_ptc_num;
    param->wall_ptc_num = jdata->wall_ptc_num;
    param->rigid_ptc_num = jdata->rigid_ptc_num;
    
    //thread setting
    param->thread_num = jdata->thread_num;

    //gpu id
    param->gpu_id = jdata->gpu_id;

    //path setting
    param->input_path = jdata->input_path;
    param->output_path = jdata->output_path;
    param->git_hash = jdata->git_hash;  

    //time setting
    param->dt = jdata->dt;
    param->start_step = jdata->start_step;
    param->current_step = jdata->current_step;
    param->end_step = jdata->end_step;
    param->output_step = jdata->output_step;

    //check time step setting
    check_dt(param);
}

void set_gpu_param(gpu_param_t *gparam, cpu_param_t *cparam)
{
    gparam->half_dt = cparam->dt/2.0f;
    gparam->h = cparam->h;
    gparam->g = cparam->g;
    gparam->m = cparam->m;
    gparam->rho0 = cparam->rho0;
    gparam->adh = cparam->adh;
    gparam->h2 = cparam->h * cparam->h;
    gparam->eta2 = cparam->eta * cparam->eta;
    gparam->cs2 = cparam->cs * cparam->cs;
    gparam->alpha = cparam->alpha;
    gparam->delta_h_cs = cparam->delta * cparam->h * cparam->cs;
    gparam->h_cs_rho_m = cparam->h * cparam->cs * cparam->rho0 * cparam->m;

    //grid part
    gparam->grid_factor = cparam->grid_factor;
    gparam->grid_size = cparam->grid_size;
    gparam->grid_xmin = cparam->grid_xmin;
    gparam->grid_ymin = cparam->grid_ymin;
    gparam->grid_zmin = cparam->grid_zmin;
    gparam->grid_xdim = cparam->grid_xdim;
    gparam->grid_ydim = cparam->grid_ydim;
    gparam->grid_zdim = cparam->grid_zdim;
    gparam->grid_hash_min = cparam->grid_hash_min;
    gparam->grid_hash_max = cparam->grid_hash_max;

    //ptc part
    gparam->ptc_num = cparam->ptc_num;
    //std::cout << gparam->ptc_num << " " << cparam->ptc_num << std::endl;
}