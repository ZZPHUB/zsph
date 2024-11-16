
#include "equation.cuh"

__device__ int is_out(float3 pos)
{
    if(pos.x < par.grid_xmin || pos.y < par.grid_ymin || pos.z < par.grid_zmin || pos.x > par.grid_xmax || pos.y > par.grid_ymax || pos.z > par.grid_zmax)
    {
        return 1;
    }
    else 
    {
        return 0;
    }
}

__global__ void cuda_prediction(gpu_ptc_t *tptc_data)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const gpu_ptc_t ptc_data = *tptc_data;
    float3 tmp_pos_;
    float3 tmp_vel_;
    float2 tmp_rhop_;
    float4 acc_drhodt_;
    if(index < par.gmax_size)
    {
        int is_ptc = ptc_data.is_ptc[index];
        int gpu_id = ptc_data.gpu_id[index];
        if(is_ptc == 1 && gpu_id == par.gpu_id && ptc_data.type[index] == 1)
        {
            //load the pos vel and rho and p
            tmp_pos_ = ptc_data.pos[index];
            tmp_vel_ = ptc_data.vel[index];
            tmp_rhop_ = ptc_data.rhop[index];

            //load the acc and drhodt
            acc_drhodt_ = gtmp_data.acc_drhodt[index];

            //store the temp pos vel and rho and p
            ptc_data.tmp_pos[index] = tmp_pos_;
            ptc_data.tmp_vel[index] = tmp_vel_;
            ptc_data.tmp_rhop[index] = tmp_rhop_;

            //time integration 
            tmp_pos_.x += tmp_vel_.x * par.half_dt;//x
            tmp_pos_.y += tmp_vel_.y * par.half_dt;//y
            tmp_pos_.z += tmp_vel_.z * par.half_dt;//z

            tmp_vel_.x += acc_drhodt_.x * par.half_dt;//vx
            tmp_vel_.y += acc_drhodt_.y * par.half_dt;//vy
            tmp_vel_.z += acc_drhodt_.z * par.half_dt;//vz

            tmp_rhop_.x += acc_drhodt_.w * par.half_dt;//rho

            if(tmp_rhop_.x < par.rho0) tmp_rhop_.x = par.rho0;
            tmp_rhop_.y = par.cs2*(tmp_rhop_.x - par.rho0);

            //store pos vel and rho and p
            if(is_out(tmp_pos_))
            {
                ptc_data.is_ptc[index] = 0;
                tmp_pos_.x = par.grid_xmax;
                tmp_pos_.y = par.rec_r;
                tmp_pos_.z = par.grid_zmax;
            }
            ptc_data.pos[index] = tmp_pos_;
            ptc_data.vel[index] = tmp_vel_;
            ptc_data.rhop[index] = tmp_rhop_;
            
        }
    }
}

__global__ void cuda_correction(gpu_ptc_t *tptc_data)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const gpu_ptc_t ptc_data = *tptc_data;
    float3 tmp_pos_;
    float3 tmp_vel_;
    float3 ptc_vel_;
    float2 tmp_rhop_;
    float4 acc_drhodt_;
    if(index < par.gmax_size)
    {
        int is_ptc = ptc_data.is_ptc[index];
        int gpu_id = ptc_data.gpu_id[index];
        if(is_ptc == 1 && gpu_id == par.gpu_id && ptc_data.type[index] == 1)
        {
            //load the tmp pos vel rho and p
            tmp_pos_ = ptc_data.tmp_pos[index];
            tmp_vel_ = ptc_data.tmp_vel[index];
            tmp_rhop_ = ptc_data.tmp_rhop[index];

            //load the vel
            ptc_vel_ = ptc_data.vel[index];

            //load the acc and drhodt
            acc_drhodt_ = gtmp_data.acc_drhodt[index];

            //time integration
            tmp_pos_.x += ptc_vel_.x * par.half_dt * 2.0f;
            tmp_pos_.y += ptc_vel_.y * par.half_dt * 2.0f;
            tmp_pos_.z += ptc_vel_.z * par.half_dt * 2.0f;

            tmp_vel_.x += acc_drhodt_.x * par.half_dt * 2.0f;
            tmp_vel_.y += acc_drhodt_.y * par.half_dt * 2.0f;
            tmp_vel_.z += acc_drhodt_.z * par.half_dt * 2.0f;

            tmp_rhop_.x += acc_drhodt_.w * par.half_dt * 2.0f;
            if(tmp_rhop_.x < par.rho0) tmp_rhop_.x = par.rho0;
            tmp_rhop_.y = par.cs2*(tmp_rhop_.x - par.rho0);

            if(is_out(tmp_pos_))
            {
                ptc_data.is_ptc[index] = 0;
                tmp_pos_.x = par.grid_xmax;
                tmp_pos_.y = par.rec_r;
                tmp_pos_.z = par.grid_zmax;
            }
            ptc_data.pos[index] = tmp_pos_;
            ptc_data.vel[index] = tmp_vel_;
            ptc_data.rhop[index] = tmp_rhop_;
        }
    }
}