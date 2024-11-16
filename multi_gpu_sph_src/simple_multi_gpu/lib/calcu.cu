#include "lib.cuh"

__global__ void hash_calcu(gpu_ptc_t *gptc_dat)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    gpu_ptc_t ptc_dat = *gptc_dat;
    if(index < par.gmax_size)
    {
        int is_ptc = ptc_dat.is_ptc[index];
        int gpu_id = ptc_dat.gpu_id[index];
        float3 pos = ptc_dat.pos[index];
        int3 tmp_hash;
        int hash = 0;
        if(is_ptc == 1 && gpu_id <= (par.gpu_id+1) && gpu_id >= (par.gpu_id-1))
        {
            tmp_hash.x = floorf((pos.x - par.grid_xmin)/par.grid_size);
            tmp_hash.y = floorf((pos.y - par.grid_ymin)/par.grid_size);
            tmp_hash.z = floorf((pos.z - par.grid_zmin)/par.grid_size);
            hash = tmp_hash.y * par.grid_xdim * par.grid_zdim + tmp_hash.z * par.grid_xdim + tmp_hash.x;
            
            if(pos.y < par.rec_l  || pos.y >= par.rec_r)
            {
                is_ptc = 0;
                gpu_id = -10;
                pos.x = par.grid_xmax;
                pos.y = par.grid_ymax;
                pos.z = par.grid_zmax;
                hash = par.grid_hash_max-1;
            }
        }
        else
        {
            is_ptc = 0;
            gpu_id = -10;
            pos.x = par.grid_xmax;
            pos.y = par.rec_r;
            pos.z = par.grid_zmax;
            hash = par.grid_hash_max-1;
        }
        gtmp_data.hash[index] =  hash;
        gtmp_data.index[index] = index;
        ptc_dat.is_ptc[index] = is_ptc;
        ptc_dat.gpu_id[index] = gpu_id;
        ptc_dat.pos[index] = pos;
    }
}

void sort_calcu_index(gpu_tmp_t *gtmp_dat,cpu_param_t *cparam)
{
    for(int i=0;i<cparam->gpu_num;i++)
    {
        cudaSetDevice(i);
        thrust::sort_by_key(thrust::device,gtmp_dat[i].hash,gtmp_dat[i].hash+cparam->gmax_size,gtmp_dat[i].index);
    }
}

__global__ void sort_calcu_data(gpu_ptc_t *new_gptc_dat,gpu_ptc_t *old_gptc_dat)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    gpu_ptc_t old_ptc_dat = *old_gptc_dat;
    gpu_ptc_t new_ptc_dat = *new_gptc_dat;
    
    __shared__ int sharedHash[257];

    if(index < par.gmax_size)
    {
        int hash = gtmp_data.hash[index];
        sharedHash[threadIdx.x + 1] = hash;
        if(index > 0 && threadIdx.x == 0)
        {
            sharedHash[0] = gtmp_data.hash[index-1];
        }
        else if(index == 0)
        {
            sharedHash[0] = hash;
        }
        __syncthreads();

        if(hash != sharedHash[threadIdx.x])
        {
            gtmp_data.grid_start[hash] = index;
            gtmp_data.grid_end[sharedHash[threadIdx.x]] = index;
        }
        if(index == 0)
        {
            gtmp_data.grid_start[hash] = 0;
        }

        //sort data
        int old_index = gtmp_data.index[index];
        new_ptc_dat.pos[index] = old_ptc_dat.pos[old_index];
        new_ptc_dat.vel[index] = old_ptc_dat.vel[old_index];
        new_ptc_dat.rhop[index] = old_ptc_dat.rhop[old_index];

        new_ptc_dat.tmp_pos[index] = old_ptc_dat.tmp_pos[old_index];
        new_ptc_dat.tmp_vel[index] = old_ptc_dat.tmp_vel[old_index];
        new_ptc_dat.tmp_rhop[index] = old_ptc_dat.tmp_rhop[old_index];

        new_ptc_dat.table[index] = old_ptc_dat.table[old_index];
        new_ptc_dat.type[index] = old_ptc_dat.type[old_index];
        new_ptc_dat.is_ptc[index] = old_ptc_dat.is_ptc[old_index];
        new_ptc_dat.gpu_id[index] = old_ptc_dat.gpu_id[old_index];
    }
}