#include "lib.cuh"

__global__ void cuda_ptc_hash(gpu_ptc_t *ptc_data, gpu_tmp_t *tmp_data, gpu_param_t *par)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < par->ptc_num)
    {
        float3 pos;
        pos.x = ptc_data->pos_rho[index * 4 + 0];
        pos.y = ptc_data->pos_rho[index * 4 + 1];
        pos.z = ptc_data->pos_rho[index * 4 + 2];

        int3 hash;
        hash.x = floorf((pos.x - par->grid_xmin)/par->grid_size);
        hash.y = floorf((pos.y - par->grid_ymin)/par->grid_size); 
        hash.z = floorf((pos.z - par->grid_zmin)/par->grid_size);
        tmp_data->hash[index] = hash.z * par->grid_xdim * par->grid_ydim + hash.y * par->grid_xdim + hash.x;
        if(tmp_data->hash[index] < par->grid_hash_min || tmp_data->hash[index] > par->grid_hash_max) 
        {
            ptc_data->type[index] = 100;
            ptc_data->pos_rho[index*4+0] = par->grid_xmin;
            ptc_data->pos_rho[index*4+1] = par->grid_ymin;
            ptc_data->pos_rho[index*4+2] = par->grid_zmin;
            ptc_data->pos_rho[index*4+3] = par->rho0;
            ptc_data->vel_p[index*4+0] = 0.0f;
            ptc_data->vel_p[index*4+1] = 0.0f;
            ptc_data->vel_p[index*4+2] = 0.0f;
            ptc_data->vel_p[index*4+3] = 0.0f;
            tmp_data->hash[index] = 0;
        }
        tmp_data->index[index] = index;
    }
}