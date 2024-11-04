#include "lib.cuh"

void cuda_sort_index(gpu_tmp_t gdata,cpu_param_t cpar)
{
    thrust::sort_by_key(thrust::device, gdata.hash, gdata.hash + cpar.ptc_num, gdata.index);
}

__global__ void cuda_sort_data(gpu_ptc_t *tdata_old,gpu_ptc_t *tdata_new)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const gpu_ptc_t data_old = *tdata_old;
    const gpu_ptc_t data_new = *tdata_new;

    __shared__ int sharedHash[257]; //cuda blockSize + 1
    if(tid < par.ptc_num)
    {
        int hash = tmp_data.hash[tid];
        sharedHash[ threadIdx.x + 1] = hash;
        if(tid > 0 && threadIdx.x == 0)
        {
            sharedHash[0] = tmp_data.hash[tid-1];
        }
        else if (tid == 0)
        {
            sharedHash[0] = hash;
        }
        __syncthreads();

        //find the grid start and end ptc
        if(hash != sharedHash[ threadIdx.x])
        {
            //printf("%d %d %d \n",tid,sharedHash[threadIdx.x],tid);
            tmp_data.grid_start[hash] = tid;
            tmp_data.grid_end[sharedHash[ threadIdx.x]] = tid;
            for(int i=sharedHash[ threadIdx.x]+1;i<hash;i++)
            {
                tmp_data.grid_start[i] = tid;
                tmp_data.grid_end[i] = tid;
            }
        }
        else if (tid == 0)
        {
            //printf("%d %d %d\n",tid,sharedHash[threadIdx.x],tid);
            tmp_data.grid_start[hash] = tid;
            for(int i=0;i<hash;i++)
            {
                tmp_data.grid_end[i] = tid;
                tmp_data.grid_start[i] = tid;
            }
        }
        else if (tid == par.ptc_num - 1)
        {
            //printf("%d %d %d\n",tid,sharedHash[threadIdx.x],tid);
            tmp_data.grid_end[hash] = tid;
            for(int i=hash+1;i<=par.grid_hash_max;i++)
            {
                tmp_data.grid_end[i] = tid;
                tmp_data.grid_start[i] = tid;
            }
        }
        
        //reorder the pos data
        int old_index = tmp_data.index[tid]; //old index
        data_new.pos[tid] = data_old.pos[old_index];

        //reorder the vel data
        data_new.vel[tid] = data_old.vel[old_index];

        //reorder the rhop data
        data_new.rhop[tid] = data_old.rhop[old_index];

        //reorder the tmp_pos data
        data_new.tmp_pos[tid] = data_old.tmp_pos[old_index];

        //reorder the tmp_vel data
        data_new.tmp_vel[tid] = data_old.tmp_vel[old_index];
        
        //reorder the tmp_rhop data
        data_new.tmp_rhop[tid] = data_old.tmp_rhop[old_index];

        data_new.type[tid] = data_old.type[old_index];
        data_new.table[tid] = data_old.table[old_index] + old_index - tid;
    }
}