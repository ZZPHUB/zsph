#include "lib.cuh"

__global__ void cuda_prediction(gpu_ptc_t *ptc_data, gpu_tmp_t *tmp_data, gpu_param_t *par)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    float3 tmp_pos;
    float3 tmp_vel;
    float2 tmp_rhop;
    float4 acc_drhodt;
    if(index < par->ptc_num)
    {
        if(ptc_data->type[index] == 1)
        {
            //load the data of pos rho vel and p
            tmp_pos = ptc_data->pos[index];
            tmp_vel = ptc_data->vel[index];
            tmp_rhop = ptc_data->rhop[index];
            acc_drhodt = tmp_data->acc_drhodt[index];

            //save the tmp data of pos rho vel and p
            ptc_data->tmp_pos[index] = tmp_pos;
            ptc_data->tmp_vel[index] = tmp_vel;
            ptc_data->tmp_rhop[index] = tmp_rhop;
            
            //time intergration
            tmp_pos.x += tmp_vel.x * par->half_dt;//x
            tmp_pos.y += tmp_vel.y * par->half_dt;//y
            tmp_pos.z += tmp_vel.z * par->half_dt;//z

            tmp_vel.x += acc_drhodt.x * par->half_dt;//vx
            tmp_vel.y += acc_drhodt.y * par->half_dt;//vy
            tmp_vel.z += acc_drhodt.z * par->half_dt;//vz

            tmp_rhop.x += acc_drhodt.w * par->half_dt;//rho
            
            if(tmp_rhop.x < par->rho0)
            {
                tmp_rhop.x = par->rho0;
            }
            tmp_rhop.y = par->cs2 * (tmp_rhop.x - par->rho0);

            ptc_data->pos[index] = tmp_pos;
            ptc_data->vel[index] = tmp_vel;
            ptc_data->rhop[index] = tmp_rhop;
        }
    }
}

__global__ void cuda_correction(gpu_ptc_t *ptc_data, gpu_tmp_t *tmp_data, gpu_param_t *par)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    float3 tmp_pos;
    float3 tmp_vel;
    float3 ptc_vel;
    float2 tmp_rhop;
    float4 acc_drhodt;
    if(index < par->ptc_num)
    {
        if(ptc_data->type[index] == 1)
        {
            //load the data of pos rho vel and p from gpu_ptc_t's tmp_pos_rho and tmp_vel_p
            tmp_pos = ptc_data->tmp_pos[index];
            tmp_vel = ptc_data->tmp_vel[index];
            ptc_vel = ptc_data->vel[index];
            tmp_rhop = ptc_data->tmp_rhop[index];
            acc_drhodt = tmp_data->acc_drhodt[index];
            
            //time intergration
            tmp_pos.x += ptc_vel.x * par->half_dt *2.0f;//x
            tmp_pos.y += ptc_vel.y * par->half_dt *2.0f;//y
            tmp_pos.z += ptc_vel.z * par->half_dt *2.0f;//z

            tmp_vel.x += acc_drhodt.x * par->half_dt *2.0f; //vx
            tmp_vel.y += acc_drhodt.y * par->half_dt *2.0f; //vy
            tmp_vel.z += acc_drhodt.z * par->half_dt *2.0f; //vz

            tmp_rhop.x += acc_drhodt.w * par->half_dt *2.0f; //rho

            if(tmp_rhop.x < par->rho0)
            {
                tmp_rhop.x = par->rho0;
            }
            tmp_rhop.y = par->cs2 * (tmp_rhop.x - par->rho0);//p

            ptc_data->pos[index] = tmp_pos;
            ptc_data->vel[index] = tmp_vel;
            ptc_data->rhop[index]= tmp_rhop;
            
        }
    }
}