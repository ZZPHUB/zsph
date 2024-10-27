#include "lib.cuh"

__global__ void cuda_prediction(gpu_ptc_t *ptc_data, gpu_tmp_t *tmp_data, gpu_param_t *par)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    float4 tmp_vel_p;
    float4 tmp_pos_rho;
    if(index < par->ptc_num)
    {
        if(ptc_data->type[index] == 1)
        {
            //load the data of pos rho vel and p
            tmp_pos_rho.x = ptc_data->pos_rho[index * 4 + 0];
            tmp_pos_rho.y = ptc_data->pos_rho[index * 4 + 1];
            tmp_pos_rho.z = ptc_data->pos_rho[index * 4 + 2];
            tmp_pos_rho.w = ptc_data->pos_rho[index * 4 + 3];
            tmp_vel_p.x = ptc_data->vel_p[index * 4 + 0];
            tmp_vel_p.y = ptc_data->vel_p[index * 4 + 1];
            tmp_vel_p.z = ptc_data->vel_p[index * 4 + 2];
            tmp_vel_p.w = ptc_data->vel_p[index * 4 + 3];

            //save the tmp data of pos rho vel and p
            ptc_data->tmp_pos_rho[index * 4 + 0] = tmp_pos_rho.x;
            ptc_data->tmp_pos_rho[index * 4 + 1] = tmp_pos_rho.y; 
            ptc_data->tmp_pos_rho[index * 4 + 2] = tmp_pos_rho.z;
            ptc_data->tmp_pos_rho[index * 4 + 3] = tmp_pos_rho.w;
            ptc_data->tmp_vel_p[index * 4 + 0] = tmp_vel_p.x;
            ptc_data->tmp_vel_p[index * 4 + 1] = tmp_vel_p.y;
            ptc_data->tmp_vel_p[index * 4 + 2] = tmp_vel_p.z;
            ptc_data->tmp_vel_p[index * 4 + 3] = tmp_vel_p.w;

            //time intergration
            tmp_vel_p.x += tmp_data->acc_drhodt[index * 4 + 0]*par->half_dt; //vx
            tmp_vel_p.y += tmp_data->acc_drhodt[index * 4 + 1]*par->half_dt; //vy
            tmp_vel_p.z += tmp_data->acc_drhodt[index * 4 + 2]*par->half_dt; //vz
            tmp_pos_rho.w += tmp_data->acc_drhodt[index * 4 + 3]*par->half_dt; //rho
            tmp_pos_rho.x += tmp_vel_p.x*par->half_dt; //x
            tmp_pos_rho.y += tmp_vel_p.y*par->half_dt; //y
            tmp_pos_rho.z += tmp_vel_p.z*par->half_dt; //z
            
            if(tmp_pos_rho.w < par->rho0)
            {
                tmp_pos_rho.w = par->rho0;
            }
            tmp_vel_p.w = par->cs2 * (tmp_pos_rho.w - par->rho0);

            ptc_data->pos_rho[index * 4 + 0] = tmp_pos_rho.x;
            ptc_data->pos_rho[index * 4 + 1] = tmp_pos_rho.y;
            ptc_data->pos_rho[index * 4 + 2] = tmp_pos_rho.z;
            ptc_data->pos_rho[index * 4 + 3] = tmp_pos_rho.w;
            ptc_data->vel_p[index * 4 + 0] = tmp_vel_p.x;
            ptc_data->vel_p[index * 4 + 1] = tmp_vel_p.y;
            ptc_data->vel_p[index * 4 + 2] = tmp_vel_p.z;
            ptc_data->vel_p[index * 4 + 3] = tmp_vel_p.w;
        }
    }
}

__global__ void cuda_correction(gpu_ptc_t *ptc_data, gpu_tmp_t *tmp_data, gpu_param_t *par)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    float4 tmp_pos_rho;
    float4 tmp_vel_p;
    if(index < par->ptc_num)
    {
        if(ptc_data->type[index] == 1)
        {
            //load the data of pos rho vel and p from gpu_ptc_t's tmp_pos_rho and tmp_vel_p
            tmp_vel_p.x = ptc_data->tmp_vel_p[index * 4 + 0];
            tmp_vel_p.y = ptc_data->tmp_vel_p[index * 4 + 1];
            tmp_vel_p.z = ptc_data->tmp_vel_p[index * 4 + 2];
            tmp_vel_p.w = ptc_data->tmp_vel_p[index * 4 + 3];
            tmp_pos_rho.x = ptc_data->tmp_pos_rho[index * 4 + 0];
            tmp_pos_rho.y = ptc_data->tmp_pos_rho[index * 4 + 1];
            tmp_pos_rho.z = ptc_data->tmp_pos_rho[index * 4 + 2];
            tmp_pos_rho.w = ptc_data->tmp_pos_rho[index * 4 + 3];
            
            //time intergration
            tmp_vel_p.x += tmp_data->acc_drhodt[index * 4 + 0]*par->half_dt; //vx
            tmp_vel_p.y += tmp_data->acc_drhodt[index * 4 + 1]*par->half_dt; //vy
            tmp_vel_p.z += tmp_data->acc_drhodt[index * 4 + 2]*par->half_dt; //vz
            tmp_pos_rho.w += tmp_data->acc_drhodt[index * 4 + 3]*par->half_dt; //rho
            tmp_pos_rho.x += tmp_vel_p.x*par->half_dt; //x
            tmp_pos_rho.y += tmp_vel_p.y*par->half_dt; //y
            tmp_pos_rho.z += tmp_vel_p.z*par->half_dt; //z

            if(tmp_pos_rho.w < par->rho0)
            {
                tmp_pos_rho.w = par->rho0;
            }
            tmp_vel_p.w = par->cs2 * (tmp_pos_rho.w - par->rho0);//p

            ptc_data->pos_rho[index * 4 + 0] = tmp_pos_rho.x;
            ptc_data->pos_rho[index * 4 + 1] = tmp_pos_rho.y;
            ptc_data->pos_rho[index * 4 + 2] = tmp_pos_rho.z;
            ptc_data->pos_rho[index * 4 + 3] = tmp_pos_rho.w;
            ptc_data->vel_p[index * 4 + 0] = tmp_vel_p.x;
            ptc_data->vel_p[index * 4 + 1] = tmp_vel_p.y;
            ptc_data->vel_p[index * 4 + 2] = tmp_vel_p.z;
            ptc_data->vel_p[index * 4 + 3] = tmp_vel_p.w;
            
        }
    }
}