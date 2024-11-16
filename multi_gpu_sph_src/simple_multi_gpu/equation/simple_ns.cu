#include "equation.cuh"

__global__ void cuda_governing_ns(gpu_ptc_t *gptc_dat)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    gpu_ptc_t ptc_dat = *gptc_dat;
    if (index < par.gmax_size)
    {
        int is_ptc_0 = ptc_dat.is_ptc[index];
        int gpu_id_0 = ptc_dat.gpu_id[index];
        if (is_ptc_0 == 1 && gpu_id_0 == par.gpu_id)
        {
            float4 acc_drhodt = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float v_tmp = 0.0f;
            int tmp_w = 0;

            float3 pos = ptc_dat.pos[index];
            float3 vel = ptc_dat.vel[index];
            int type_0 = ptc_dat.type[index];
            float2 rhop_0 = ptc_dat.rhop[index];

            /*
            int hash_max = par.hash_end_r + par.grid_factor * par.grid_zdim * par.grid_xdim;
            int hash_min = par.hash_start_l - par.grid_factor * par.grid_zdim * par.grid_xdim;
            if(hash_max >= par.grid_hash_max) hash_max = par.grid_hash_max;
            if(hash_min < par.grid_hash_min) hash_min = par.grid_hash_min;
            */
            // printf("%d,%d\n",hash_min,hash_max);

            float3 dx;
            float3 dv;
            float2 rhop_1;
            int gpu_id_1;
            int is_ptc_1;
            int type_1;
            float dvdx, rr2, q, fr;
            for (int y = -par.grid_factor; y <= par.grid_factor; y++)
            {
                for (int z = -par.grid_factor; z <= par.grid_factor; z++)
                {
                    for (int x = -par.grid_factor; x <= par.grid_factor; x++)
                    {
                        int newGridHash = gtmp_data.hash[index] + y * par.grid_zdim * par.grid_xdim + z * par.grid_xdim + x;
                        if (newGridHash >= par.grid_hash_min && newGridHash < par.grid_hash_max)
                        {
                            int cell_start = gtmp_data.grid_start[newGridHash];
                            int cell_end = gtmp_data.grid_end[newGridHash];
                            if (cell_start == -1)
                                cell_end = -1;
                            for (int i = cell_start; i < cell_end; i++)
                            {
                                dx = ptc_dat.pos[i];
                                dv = ptc_dat.vel[i];
                                rhop_1 = ptc_dat.rhop[i];
                                type_1 = ptc_dat.type[i];
                                is_ptc_1 = ptc_dat.is_ptc[i];
                                gpu_id_1 = ptc_dat.gpu_id[i];

                                if (is_ptc_1 == 1)
                                {
                                    dx.x = pos.x - dx.x;
                                    dx.y = pos.y - dx.y;
                                    dx.z = pos.z - dx.z;
                                    dv.x = vel.x - dv.x;
                                    dv.y = vel.y - dv.y;
                                    dv.z = vel.z - dv.z;
                                    rr2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
                                    dvdx = dv.x * dx.x + dv.y * dx.y + dv.z * dx.z;

                                    q = sqrtf(rr2) / par.h;
                                    if (q <= 2.0f)
                                    {
                                        tmp_w += 1;

                                        fr = -5.0f * par.adh * (1.0f - q / 2.0f) * (1.0f - q / 2.0f) * (1.0f - q / 2.0f) / par.h2;
                                        acc_drhodt.w += (rhop_0.x * dvdx * fr + (rhop_0.x - rhop_1.x) * rr2 * fr * par.delta_h_cs / (rr2 + par.eta2)) * par.m / rhop_1.x;

                                        if (type_0 == 1 && type_1 == 1)
                                        {
                                            v_tmp = -(rhop_0.y + rhop_1.y) * par.m * fr / (rhop_0.x * rhop_1.x);
                                            v_tmp += par.h_cs_rho_m * fr * (par.alpha * dvdx / (rr2 + par.eta2)) / (rhop_0.x * rhop_1.x);
                                        }
                                        else if (type_0 == 1 && type_1 != 1 && par.dx2 > rr2)
                                        {
                                            v_tmp = -2.0f * par.g * par.grid_zmax * (powf((par.dx2/rr2),6) - powf((par.dx2 / rr2),3)) / rr2;
                                        }
                                        else
                                        {
                                            v_tmp = 0.0f;
                                        }

                                        acc_drhodt.x += v_tmp * dx.x;
                                        acc_drhodt.y += v_tmp * dx.y;
                                        acc_drhodt.z += v_tmp * dx.z;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            acc_drhodt.z += par.g;
            gtmp_data.acc_drhodt[index] = acc_drhodt;
            gtmp_data.wsum[index] = tmp_w;
        }
    }
}
