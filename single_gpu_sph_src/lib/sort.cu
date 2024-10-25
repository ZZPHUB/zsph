#include "header/lib.cuh"

void cuda_sort_index(gpu_tmp_t data,cpu_param_t par)
{
    thrust::sort_by_key(thrust::device, data.hash, data.hash + par.ptc_num, data.index);
}

__global__ void cuda_sort_data(gpu_ptc_t data_old,gpu_ptc_t data_new,gpu_tmp_t tmp_data,gpu_param_t par)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    __shared__ int sharedHash[257]; //cuda blockSize + 1

    if(tid < par.ptc_num)
    {
        int hash = tmp_data.hash[tid];
        sharedHash[ threadIdx.x + 1] = hash;
        if(tid > 0 && threadIdx.x == 0)
        {
            sharedHash[0] = data_old.hash[tid-1]
        }
        else if (tid == 0)
        {
            sharedHash[0] = hash;
        }
        __synchreads();

        //find the grid start and end ptc
        if(hash != sharedHash[ threadIdx.x])
        {
            tmp_data.grid_end[hash] = tid;
            tmp_data.grid_start[sharedHash[ threadIdx.x]] = tid;
            for(int i=sharedHash[ threadIdx.x]+1;i<hash;i++)
            {
                tmp_data.grid_start[i] = tid;
                tmp_data.grid_end[i] = tid;
            }
        }
        else if (tid == 0)
        {
            tmp_data.grid_start[hash] = tid;
            for(int i=0;i<hash;i++)
            {
                tmp_data.grid_end[i] = tid;
                tmp_data.grid_start[i] = tid;
            }
        }
        elif (tid == par.ptc_num - 1)
        {
            tmp_data.grid_end[hash] = tid;
            for(int i=hash+1;i<par.hash_max;i++)
            {
                tmp_data.grid_end[i] = tid;
                tmp_data.grid_start[i] = tid;
            }
        }
        
        //reorder the pos_rho data
        int old_index = tmp_data.index[tid]; //old index
        data_new.pos_rho[tid * 4 + 0] = data_old.pos_rho[old_index * 4 + 0];
        data_new.pos_rho[tid * 4 + 1] = data_old.pos_rho[old_index * 4 + 1];
        data_new.pos_rho[tid * 4 + 2] = data_old.pos_rho[old_index * 4 + 2];
        data_new.pos_rho[tid * 4 + 3] = data_old.pos_rho[old_index * 4 + 3];

        //reorder the vel_p data
        data_new.vel_p[tid * 4 + 0] = data_old.vel_p[old_index * 4 + 0];
        data_new.vel_p[tid * 4 + 1] = data_old.vel_p[old_index * 4 + 1];
        data_new.vel_p[tid * 4 + 2] = data_old.vel_p[old_index * 4 + 2];
        data_new.vel_p[tid * 4 + 3] = data_old.vel_p[old_index * 4 + 3];

        //reorder the tmp_pos_rho data
        data_new.tmp_pos_rho[tid * 4 + 0] = data_old.tmp_pos_rho[old_index * 4 + 0];
        data_new.tmp_pos_rho[tid * 4 + 1] = data_old.tmp_pos_rho[old_index * 4 + 1];
        data_new.tmp_pos_rho[tid * 4 + 2] = data_old.tmp_pos_rho[old_index * 4 + 2];
        data_new.tmp_pos_rho[tid * 4 + 3] = data_old.tmp_pos_rho[old_index * 4 + 3];

        //reorder the tmp_vel_p data
        data_new_tmp_vel_p[tid * 4 + 0] = data_old.tmp_vel_p[old_index * 4 + 0];
        data_new_tmp_vel_p[tid * 4 + 1] = data_old.tmp_vel_p[old_index * 4 + 1];
        data_new_tmp_vel_p[tid * 4 + 2] = data_old.tmp_vel_p[old_index * 4 + 2];
        data_new_tmp_vel_p[tid * 4 + 3] = data_old.tmp_vel_p[old_index * 4 + 3];

        /*
        data_new.acc_empty[tid * 4 + 0] = data_old.acc_empty[old_index * 4 + 0];
        data_new.acc_empty[tid * 4 + 1] = data_old.acc_empty[old_index * 4 + 1];
        data_new.acc_empty[tid * 4 + 2] = data_old.acc_empty[old_index * 4 + 2];
        data_new.acc_empty[tid * 4 + 3] = data_old.acc_empty[old_index * 4 + 3];
        */

        data_new.type[tid] = data_old.type[old_index];
        data_new.table[tid] = data_old.table[old_index] + old_index - tid;
    }
}