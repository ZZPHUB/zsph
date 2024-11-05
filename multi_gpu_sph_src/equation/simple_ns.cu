#include "equation.cuh"

__global__ void cuda_boundary_ns(gpu_ptc_t *tptc_data)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const gpu_ptc_t ptc_data = *tptc_data;

    if(index < par.ptc_num)
    {
        float rhop_sum_tmp = 0.0f;
        float w_sum_tmp = 0.0f;
        float dofv_tmp = 0.0f;
        float3 pos = ptc_data.pos[index];
        float3 vel = ptc_data.vel[index];
        int type_0 = ptc_data.type[index];

        int type_1;
        float3 dx;
        float3 dv;
        float2 rhop_1;
        float rr2,q,w,fr;
        float dvdx;
        //int count = 0;
        for(int z=-par.grid_factor;z<=par.grid_factor;z++)
        {
            for(int y=-par.grid_factor;y<par.grid_factor;y++)
            {
                int cell_start = tmp_data.hash[index] + z * par.grid_xdim * par.grid_ydim + y * par.grid_xdim;
                int cell_end = cell_start + par.grid_factor;
                cell_start = cell_start - par.grid_factor;
                if(cell_start >= par.grid_hash_min && cell_end <= par.grid_hash_max)
                {
                    cell_start = tmp_data.grid_start[cell_start];
                    cell_end = tmp_data.grid_end[cell_end];
                    for(int i=cell_start;i<cell_end;i++)
                    {
                        if(i != index)
                        {
                            
                            dx = ptc_data.pos[i];
                            dv = ptc_data.vel[i];
                            rhop_1 = ptc_data.rhop[i];

                            dx.x = pos.x - dx.x;
                            dx.y = pos.y - dx.y;
                            dx.z = pos.z - dx.z;
                            dv.x = vel.x - dv.x;
                            dv.y = vel.y - dv.y;
                            dv.z = vel.z - dv.z;
                            rr2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z ;
                            dvdx = dv.x * dx.x + dv.y * dx.y + dv.z * dx.z;

                            type_1 = ptc_data.type[i];
                            rhop_1 = ptc_data.rhop[i];

                            q = sqrtf(rr2)/par.h;
                            if(q <= 2.0f)
                            {
                                //count ++;
                                fr = (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f);
                                w = fr*(1.0f - q/2.0f);
                                w *= (2.0f*q + 1.0f)*par.adh;
                                fr *= -5.0f*par.adh/(par.h2);
                                if(type_0 != 1 && type_1 == 1)
                                {
                                    rhop_sum_tmp += (rhop_1.y - rhop_1.x*(0.0f*dx.x + 0.0f*dx.y + (0.0f - par.g)*dx.z))*w;
                                    w_sum_tmp += w;
                                }
                                else if(type_0 == 1 && type_1 == 1)
                                {
                                    dofv_tmp -= fr*dvdx*par.m/rhop_1.x;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        tmp_data.dofv[index] = dofv_tmp;
        if(type_0 != 1)
        {
            if(fabs(w_sum_tmp) > 1e-8f)
            {
                rhop_sum_tmp = rhop_sum_tmp/w_sum_tmp;
            }
            else 
            {
                rhop_sum_tmp = 0.0f;
            }
            if(rhop_sum_tmp < 0.0f) rhop_sum_tmp = 0.0f;
            rhop_1.y = rhop_sum_tmp;
            rhop_1.x = rhop_sum_tmp/par.cs2 + par.rho0;
            ptc_data.rhop[index] = rhop_1;
        }
    }
}

__global__ void cuda_govering_ns(gpu_ptc_t *tptc_data)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const gpu_ptc_t ptc_data = *tptc_data;
    if(index < par.ptc_num)
    {
        float4 acc_drhodt = make_float4(0.0f,0.0f,0.0f,0.0f);
        float3 pos = ptc_data.pos[index];
        float3 vel = ptc_data.vel[index];
        float2 rhop_0 = ptc_data.rhop[index];
        int type_0 = ptc_data.type[index]; 
        float dofv_0 = tmp_data.dofv[index];
       
        int type_1;
        float2 rhop_1;
        float dofv_1;
        float3 dx;
        float3 dv;
        float rr2,q,fr;
        float dvdx;

        //tmp variable
        float v_tmp = 0.0f;
        
        for(int z=-par.grid_factor;z<=par.grid_factor;z++)
        {
            for(int y=-par.grid_factor;y<=par.grid_factor;y++)
            {
                int cell_start = tmp_data.hash[index] + z*par.grid_xdim*par.grid_ydim + y*par.grid_xdim;
                int cell_end = cell_start + par.grid_factor;
                cell_start = cell_start - par.grid_factor;
                if(cell_start >= par.grid_hash_min && cell_end <= par.grid_hash_max)
                {
                    cell_start = tmp_data.grid_start[cell_start];
                    cell_end = tmp_data.grid_end[cell_end];
                    for(int i=cell_start;i<cell_end;i++)
                    {
                        if(i != index )
                        {
                            dx = ptc_data.pos[i];
                            dv = ptc_data.vel[i];
                            rhop_1 = ptc_data.rhop[i];

                            type_1 = ptc_data.type[i];
                            dofv_1 = tmp_data.dofv[i];

                            //dx
                            dx.x = pos.x - dx.x;
                            dx.y = pos.y - dx.y;
                            dx.z = pos.z - dx.z;
                            
                            //dvx
                            dv.x = vel.x - dv.x;
                            dv.y = vel.y - dv.y;
                            dv.z = vel.z - dv.z;

                            rr2= dx.x*dx.x + dx.y*dx.y + dx.z*dx.z; //rr
                            dvdx = dv.x * dx.x + dv.y * dx.y + dv.z * dx.z; //dvdx
                            q = sqrtf(rr2)/par.h;

                            if(q <= 2.0f)
                            {
                                fr = -5.0f * par.adh * (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)/par.h2;
                                acc_drhodt.w += (rhop_0.x*dvdx*fr + (rhop_0.x-rhop_1.x)*rr2*fr*par.delta_h_cs/(rr2+par.eta2))*par.m/rhop_1.x;
                                v_tmp = -(rhop_0.y+rhop_1.y)*par.m*fr/(rhop_0.x*rhop_1.x);
                                if(type_0 == 1 && type_1 == 1)
                                {
                                    v_tmp += par.h_cs_rho_m*fr*(dofv_0 + dofv_1 + par.alpha*dvdx/(rr2+par.eta2))/(rhop_0.x*rhop_1.x);
                                }
                                acc_drhodt.x += v_tmp*dx.x;
                                acc_drhodt.y += v_tmp*dx.y;
                                acc_drhodt.z += v_tmp*dx.z;
                            }
                        }
                    }
                }
            }
        }
        acc_drhodt.z += par.g;
        tmp_data.acc_drhodt[index] = acc_drhodt;
    }
}
