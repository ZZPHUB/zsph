#include "lib.cuh"

int in_which_gpu(float posy, cpu_param_t *cparam, gpu_param_t *gparam)
{
    for (int i = 0; i < cparam->gpu_num; i++)
    {
        if (posy >= gparam[i].start_l && posy < gparam[i].end_r)
        {
            return i;
        }
    }
    return -1;
}

void split_in_cpu(cpu_input_t *ginput_dat, cpu_input_t *cintput_dat, cpu_param_t *cparam, gpu_param_t *gparam, gpu_socket_t **gsocket)
{
    for (int i = 0; i < cparam->gpu_num; i++)
    {
        gsocket[i]->total_ptc_num = gsocket[i]->send_ptc_num = gsocket[i]->outer_ptc_num = 0;
    }
    std::cout << "*******************************************************************************************" << std::endl;
    std::cout << "* Zsph is spliting the data for gpus..." << std::endl;
    for (int i = 0; i < cparam->ptc_num; i++)
    {
        /*
        int3 hash;
        hash.x = (int)((cintput_dat->pos[i * 3 + 0] - cparam->grid_xmin) / cparam->grid_size);
        hash.y = (int)((cintput_dat->pos[i * 3 + 1] - cparam->grid_ymin) / cparam->grid_size);
        hash.z = (int)((cintput_dat->pos[i * 3 + 2] - cparam->grid_zmin) / cparam->grid_size);
        hash.x += hash.y * cparam->grid_xdim * cparam->grid_zdim + hash.z * cparam->grid_xdim;
        */
        float posy = cintput_dat->pos[i * 3 + 1];
        int gpu_id = in_which_gpu(posy, cparam, gparam);
        if (gpu_id == -1)
        {
            std::cerr << "* Zsph error in " << __FILE__ << ":" << __LINE__ << std::endl;
            exit(1);
        }
        else
        {
            int index = gsocket[gpu_id]->total_ptc_num;
            if (gsocket[gpu_id]->total_ptc_num >= cparam->gmax_size)
            {
                std::cerr << "Zsph error in " << cparam->gmax_size << gsocket[gpu_id]->total_ptc_num << __FILE__ << ":" << __LINE__ << std::endl;
                exit(1);
            }
            ginput_dat[gpu_id].pos[index * 3 + 0] = cintput_dat->pos[i * 3 + 0];
            ginput_dat[gpu_id].pos[index * 3 + 1] = cintput_dat->pos[i * 3 + 1];
            ginput_dat[gpu_id].pos[index * 3 + 2] = cintput_dat->pos[i * 3 + 2];

            ginput_dat[gpu_id].vel[index * 3 + 0] = cintput_dat->vel[i * 3 + 0];
            ginput_dat[gpu_id].vel[index * 3 + 1] = cintput_dat->vel[i * 3 + 1];
            ginput_dat[gpu_id].vel[index * 3 + 2] = cintput_dat->vel[i * 3 + 2];

            ginput_dat[gpu_id].rhop[index * 2 + 0] = cintput_dat->rhop[i * 2 + 0];
            ginput_dat[gpu_id].rhop[index * 2 + 1] = cintput_dat->rhop[i * 2 + 1];

            ginput_dat[gpu_id].acc_drhodt[index * 4 + 0] = cintput_dat->acc_drhodt[i * 4 + 0];
            ginput_dat[gpu_id].acc_drhodt[index * 4 + 1] = cintput_dat->acc_drhodt[i * 4 + 1];
            ginput_dat[gpu_id].acc_drhodt[index * 4 + 2] = cintput_dat->acc_drhodt[i * 4 + 2];
            ginput_dat[gpu_id].acc_drhodt[index * 4 + 2] = cintput_dat->acc_drhodt[i * 4 + 2];

            ginput_dat[gpu_id].type[index] = cintput_dat->type[i];
            ginput_dat[gpu_id].table[index] = cintput_dat->table[i];
            ginput_dat[gpu_id].is_ptc[index] = 1;
            ginput_dat[gpu_id].gpu_id[index] = gpu_id;

            gsocket[gpu_id]->total_ptc_num += 1;
        }
    }
    for (int i = 0; i < cparam->gpu_num; i++)
    {
        std::cout << "* GPU" << i << " ptc num is:" << gsocket[i]->total_ptc_num << std::endl;
    }
}

void init_socket(gpu_socket_t **gsocket, cpu_param_t *cparam)
{
    for (int i = 0; i < cparam->gpu_num; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < cparam->gpu_num; i++)
    {
        cudaSetDevice(i);
        cudaMemset(gsocket[i], 0, sizeof(gpu_socket_t));
    }
    for (int i = 0; i < cparam->gpu_num; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
}

__global__ void hash_split(gpu_ptc_t *gptc_dat)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const gpu_ptc_t ptc_dat = *gptc_dat;

    if (index < par.gmax_size)
    {
        int tmp_gpu_id = ptc_dat.gpu_id[index];
        int tmp_is_ptc = ptc_dat.is_ptc[index];
        float3 pos = ptc_dat.pos[index];
        int tmp_hash = 0;
        if (tmp_is_ptc == 1 && tmp_gpu_id == par.gpu_id)
        {
            if (pos.y >= par.end_l && pos.y < par.start_r)
            {
                tmp_hash = 0;
                tmp_is_ptc = 1;
                tmp_gpu_id = par.gpu_id;
            }
            else if ((pos.y >= par.start_l && pos.y < par.end_l) || (pos.y >= par.start_r && pos.y < par.end_r))
            {
                tmp_hash = 1;
                tmp_is_ptc = 1;
                tmp_gpu_id = par.gpu_id;
            }
            else if (pos.y < par.start_l)
            {
                tmp_hash = 2;
                tmp_is_ptc = 1;
                tmp_gpu_id = par.gpu_id - 1;
            }
            else if (pos.y >= par.end_r)
            {
                tmp_hash = 2;
                tmp_is_ptc = 1;
                tmp_gpu_id = par.gpu_id + 1;
            }
        }
        else
        {
            tmp_hash = 3;
            tmp_is_ptc = 0;
            tmp_gpu_id = -10;
        }
        ptc_dat.is_ptc[index] = tmp_is_ptc;
        ptc_dat.gpu_id[index] = tmp_gpu_id;
        gtmp_data.hash[index] = tmp_hash;
        gtmp_data.index[index] = index;
    }
}

void sort_split_index(gpu_tmp_t *gtmp_dat, cpu_param_t *cparam)
{
    for (int i = 0; i < cparam->gpu_num; i++)
    {
        cudaSetDevice(i);
        thrust::sort_by_key(thrust::device, gtmp_dat[i].hash, gtmp_dat[i].hash + cparam->gmax_size, gtmp_dat[i].index);
    }
    check_gerr(__FILE__, __LINE__);
}

__global__ void sort_split_data(gpu_ptc_t *new_gptc_dat, gpu_ptc_t *old_gptc_dat)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    gpu_ptc_t new_dat = *new_gptc_dat;
    gpu_ptc_t old_dat = *old_gptc_dat;
    if (index < par.gmax_size)
    {
        int old_index = gtmp_data.index[index];

        new_dat.pos[index] = old_dat.pos[old_index];
        new_dat.vel[index] = old_dat.vel[old_index];
        new_dat.rhop[index] = old_dat.rhop[old_index];

        new_dat.tmp_pos[index] = old_dat.tmp_pos[old_index];
        new_dat.tmp_vel[index] = old_dat.tmp_vel[old_index];
        new_dat.tmp_rhop[index] = old_dat.tmp_rhop[old_index];

        new_dat.type[index] = old_dat.type[old_index];
        new_dat.table[index] = old_dat.table[old_index];

        new_dat.gpu_id[index] = old_dat.gpu_id[old_index];
        new_dat.is_ptc[index] = old_dat.is_ptc[old_index];
    }
}

__global__ void find_split_data(gpu_ptc_t *gptc_dat, gpu_socket_t *gsocket_dat)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    gpu_ptc_t ptc_dat = *gptc_dat;
    __shared__ int sharedHash[257];

    if (index < par.gmax_size)
    {
        int hash = gtmp_data.hash[index];
        sharedHash[threadIdx.x + 1] = hash;
        if (index > 0 && threadIdx.x == 0)
        {
            sharedHash[0] = gtmp_data.hash[index - 1];
        }
        else if (index == 0)
        {
            sharedHash[0] = gtmp_data.hash[index];
        }
        __syncthreads();
        if (hash != sharedHash[threadIdx.x])
        {
            if (hash == 1 && sharedHash[threadIdx.x] == 0)
            {
                gsocket_dat->pos_send = &(ptc_dat.pos[index]);
                gsocket_dat->vel_send = &(ptc_dat.vel[index]);
                gsocket_dat->rhop_send = &(ptc_dat.rhop[index]);
                gsocket_dat->tmp_pos_send = &(ptc_dat.tmp_pos[index]);
                gsocket_dat->tmp_vel_send = &(ptc_dat.tmp_vel[index]);
                gsocket_dat->tmp_rhop_send = &(ptc_dat.tmp_rhop[index]);
                gsocket_dat->type_send = &(ptc_dat.type[index]);
                gsocket_dat->table_send = &(ptc_dat.table[index]);
                gsocket_dat->gpu_id_send = &(ptc_dat.gpu_id[index]);
                gsocket_dat->is_ptc_send = &(ptc_dat.is_ptc[index]);

                // gsocket_dat->send_ptc_num = total_ptc_num - index;//index is the first element that will be send
                gsocket_dat->send_ptc_num = index;
                // gsocket_dat->send_ptc_num = index;
            }
            else if (hash == 3 && sharedHash[threadIdx.x] <= 2)
            {
                gsocket_dat->total_ptc_num = index;

                int tmp_index = index;
                gsocket_dat->pos_rec_l = &(ptc_dat.pos[tmp_index]);
                gsocket_dat->vel_rec_l = &(ptc_dat.vel[tmp_index]);
                gsocket_dat->rhop_rec_l = &(ptc_dat.rhop[tmp_index]);
                gsocket_dat->tmp_pos_rec_l = &(ptc_dat.tmp_pos[tmp_index]);
                gsocket_dat->tmp_vel_rec_l = &(ptc_dat.tmp_vel[tmp_index]);
                gsocket_dat->tmp_rhop_rec_l = &(ptc_dat.tmp_rhop[tmp_index]);
                gsocket_dat->type_rec_l = &(ptc_dat.type[tmp_index]);
                gsocket_dat->table_rec_l = &(ptc_dat.table[tmp_index]);
                gsocket_dat->is_ptc_rec_l = &(ptc_dat.is_ptc[tmp_index]);
                gsocket_dat->gpu_id_rec_l = &(ptc_dat.gpu_id[tmp_index]);

                tmp_index += (par.gmax_size - index) / 2; // alloc for receive the data form right

                gsocket_dat->pos_rec_r = &(ptc_dat.pos[tmp_index]);
                gsocket_dat->vel_rec_r = &(ptc_dat.vel[tmp_index]);
                gsocket_dat->rhop_rec_r = &(ptc_dat.rhop[tmp_index]);
                gsocket_dat->tmp_pos_rec_r = &(ptc_dat.tmp_pos[tmp_index]);
                gsocket_dat->tmp_vel_rec_r = &(ptc_dat.tmp_vel[tmp_index]);
                gsocket_dat->tmp_rhop_rec_r = &(ptc_dat.tmp_rhop[tmp_index]);
                gsocket_dat->type_rec_r = &(ptc_dat.type[tmp_index]);
                gsocket_dat->table_rec_r = &(ptc_dat.table[tmp_index]);
                gsocket_dat->is_ptc_rec_r = &(ptc_dat.is_ptc[tmp_index]);
                gsocket_dat->gpu_id_rec_r = &(ptc_dat.gpu_id[tmp_index]);
            }
        }
    }
}