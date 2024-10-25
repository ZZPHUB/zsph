#include "header/equation.cuh"

__global__ void cuda_boundary_ns(gpu_ptc_t ptc_data, gpu_tmp_t tmp_data, gpu_param_t par)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < par.ptc_num)
    {
        float rhop_sum_tmp = 0.0f;
        float w_sum_tmp = 0.0f;
        float dofv_tmp = 0.0f;
        float3 pos,vel;
        float rho_0;
        float p_0;
        int type_0;

        type_0 = ptc_data.type[index];
        pos.x = ptc_data.pos_rho[index * 4 + 0];
        pos.y = ptc_data.pos_rho[index * 4 + 1];
        pos.z = ptc_data.pos_rho[index * 4 + 2];
        rho_0 = ptc_data.pos_rho[index * 4 + 3];

        vel.x = ptc_data.vel_p[index * 4 + 0];
        vel.y = ptc_data.vel_p[index * 4 + 1];
        vel.z = ptc_data.vel_p[index * 4 +2 ];
        p_0 = ptc_data.vel_p[index * 4 + 3];


        int type_1;
        float rho_1;
        float p_1;
        float dx,dy,dz;
        float dvx,dvy,dvz;
        float rr2,q,w,fr;
        float dvdx;
        for(int z=-3;z<=3;z++)
        {
            for(int y=-3;y<=3;y++)
            {
                int newgridHash = tmp_data.hash[index] + z*par.grid_xdim*par.grid_ydim + y*par.grid_xdim;
                if(newgridHash <= par.grid_hash_max - 3 && newgridHash >= 3)
                {
                    #define startIndex (tmp_data.grid_start[newgridHash-3])
                    #define endIndex (tmp_data.grid_end[newgridHash+3])
                    for(int i=startIndex;i<endIndex;i++)
                    {
                        if(i != index )
                        {
                            dx = ptc_data.pos_rho[i * 4 + 0];
                            dy = ptc_data.pos_rho[i * 4 + 1];
                            dz = ptc_data.pos_rho[i * 4 + 2];
                            rho_1 = ptc_data.pos_rho[i * 4 + 3];

                            dvx = ptc_data.vel_p[i * 4 + 0];
                            dvy = ptc_data.vel_p[i * 4 + 1];
                            dvz = ptc_data.vel_p[i * 4 + 2];
                            p_1 = ptc_data.vel_p[i * 4 + 3];

                            type_1 = ptc_data.type[i];

                            dx = pos - dx;
                            dy = pos - dy;
                            dz = pos - dz;
                            dvx = vel.x - dvx;
                            dvy = vel.y - dvy;
                            dvz = vel.z - dvz;
                            rr2= dx*dx + dy*dy + dz*dz;
                            dvdx = dvx * dx + dvy * dy + dvz * dz;
                            q = sqrtf(rr2)/par.h;

                            if(q <= 2.0f)
                            {
                                fr = (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f);
                                w = fr*(1.0f - q/2.0f);
                                w *= (2.0f*q + 1)*par.adh;
                                fr *= -5.0f*par.adh/par.h2;
                                if(type_0 != 1 && type_1 == 1)//type 1 is fluid and 0 is boundary
                                {
                                    rhop_sum_tmp += (p_1 - rho_1*(0.0f*dx + 0.0f*dy + (0.0f - par.g)*dz))*w;
                                    w_sum_tmp += w;
                                }
                                elif(type_0 == 1 && type_1 == 1)
                                {
                                    dofv_tmp -= fr * dvdx * par.m/rho_1;
                                }
                            }
                        }
                    }
                    #undef startIndex
                    #undef endIndex
                }
            }
        }
        tmp_data.dofv[index] = dofv_tmp;
        if(type_0 != 1)
        {
            if(fabs(w_sum_tmp) > 1.0E-8 && rhop_sum_tmp > 0.0f)
            {
                rhop_sum_tmp = rhop_sum_tmp/w_sum_tmp;
            }
            else
            {
                rhop_sum_tmp = 0.0f;
            }
            ptc_data.vel_p[index * 4 + 3] = rhop_sum_tmp;
            ptc_data.pos_rho[index * 4 + 3] = rhop_sum_tmp/par.cs2 + par.rho0;
        }
    }
}

__global__ void cuda_govering_ns(gpu_ptc_t ptc_data, gpu_tmp_t tmp_data, gpu_param_t par)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < par.ptc_num)
    {
        float4 acc_drhodt = make_float4(0.0f,0.0f,0.0f,0.0f);
        float3 pos,vel;
        float rho_0;
        float p_0;
        float dofv_0;
        int type_0; 
        
        type_0= ptc_data.type[index];
        dofv_0 = tmp_data.dofv[index];

        pos.x = ptc_data.pos_rho[index * 4 + 0];
        pos.y = ptc_data.pos_rho[index * 4 + 1];
        pos.z = ptc_data.pos_rho[index * 4 + 2];
        rho_0 = ptc_data.pos_rho[index * 4 + 3];

        vel.x = ptc_data.vel_p[index * 4 + 0];
        vel.y = ptc_data.vel_p[index * 4 + 1];
        vel.z = ptc_data.vel_p[index * 4 +2 ];
        p_0 = ptc_data.vel_p[index * 4 + 3];

        int type_1;
        float rho_1;
        float p_1;
        float dofv_1;
        float dx,dy,dz;
        float dvx,dvy,dvz;
        float rr2,q,w,fr;
        float dvdx;
        //tmp variable
        float v_tmp = 0.0f;
        for(int z=-3;z<=3;z++)
        {
            for(int y=-3;y<=3;y++)
            {
                int newgridHash = tmp_data.hash[index] + z*par.grid_xdim*par.grid_ydim + y*par.grid_xdim;
                if(newgridHash <= par.grid_hash_max - 3 && newgridHash >= 3)
                {
                    #define startIndex (tmp_data.grid_start[newgridHash-3])
                    #define endIndex (tmp_data.grid_end[newgridHash+3])
                    for(int i=startIndex;i<endIndex;i++)
                    {
                        if(i != index )
                        {
                            dx = ptc_data.pos_rho[i * 4 + 0];
                            dy = ptc_data.pos_rho[i * 4 + 1];
                            dz = ptc_data.pos_rho[i * 4 + 2];
                            rho_1 = ptc_data.pos_rho[i * 4 + 3];

                            dvx = ptc_data.vel_p[i * 4 + 0];
                            dvy = ptc_data.vel_p[i * 4 + 1];
                            dvz = ptc_data.vel_p[i * 4 + 2];
                            p_1 = ptc_data.vel_p[i * 4 + 3];

                            type_1 = ptc_data.type[i];
                            dofv_1 = tmp_data.dofv[i];

                            dx = pos - dx;
                            dy = pos - dy;
                            dz = pos - dz;
                            dvx = vel.x - dvx;
                            dvy = vel.y - dvy;
                            dvz = vel.z - dvz;
                            rr2= dx*dx + dy*dy + dz*dz;
                            dvdx = dvx * dx + dvy * dy + dvz * dz;
                            q = sqrtf(rr2)/par.h;

                            if(q <= 2.0f)
                            {
                                fr = -5.0f * par.adh * (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)/par.h2;
                                acc_drhodt.w += (rho_0*dvdx*fr + (rho_0-rho_1)*rr2*fr*par.delta_h_cs/(rr2+par.eta2))*par.m/rho_1;
                                v_tmp = -(p_0+p_1)*par.m*fr/(rho_0*rho_1);
                                if(type_0 == 1 && type_1 == 1)
                                {
                                    v_tmp += par.h_cs_rho_m*fr*(dofv_0 + dofv_1 + par.alpha*dvdx/(rr2+par.eta2))/(rho_0*rho_1);
                                }
                                acc_drhodt.x += v_tmp*dx;
                                acc_drhodt.y += v_tmp*dy;
                                acc_drhodt.z += v_tmp*dz;
                            }
                        }
                    }
                    #undef startIndex
                    #undef endIndex
                }
            }
        }
        tmp_data.acc_drhodt[index * 4 + 0] = acc_drhodt.x;
        tmp_data.acc_drhodt[index * 4 + 1] = acc_drhodt.y;
        tmp_data.acc_drhodt[index * 4 + 2] = acc_drhodt.z + par.g;
        tmp_data.acc_drhodt[index * 4 + 3] = acc_drhodt.w;
    }
}