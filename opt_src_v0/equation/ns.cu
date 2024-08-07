#include "sph.cuh"

using namespace std;

void SPH_NS_simpleversion(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int* sorted_particle_type, float* densitydt, float* Veldt, int* cellStart, int* cellEnd, int numParticles, int* particleHash, int timestep, float* dofv, float* rhop_sum, float* w_sum)
{
    int numThreads, numBlocks;
    //computeGridSize(numParticles, 128, numBlocks, numThreads); 
    computeGridSize(numParticles, 256, numBlocks, numThreads); 

    computeBoundary_Delta_acoustic_D<<<numBlocks,numThreads>>>(sortedPos, sortedVel, sorteddensity, sortedpressure, sorted_particle_type, cellStart, cellEnd, rhop_sum, w_sum, numParticles, particleHash, dofv);

    computeGovering_equationD<<<numBlocks,numThreads>>>(sortedPos, sortedVel, sortedpressure, sorteddensity, sorted_particle_type, densitydt, dofv, Veldt, cellStart, cellEnd, numParticles, particleHash);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "SPH_NS launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        system("pause");
    }
}

__global__ void computeBoundary_Delta_acoustic_D(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int* sorted_particle_type, int* cellStart, int* cellEnd, float* rhop_sum, float* w_sum, int numParticles, int* particleHash, float* dofv)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numParticles)
    {
        //dofv[index] = 0.0; rhop_sum[index] = 0.0; w_sum[index] = 0.0;//dofv-acoustic damper
        float rhop_sum_tmp = 0.0f;
        float w_sum_tmp = 0.0f;
        float dofv_tmp = 0.0f;
        float3 pos;
        pos.x = sortedPos[3 * index];
        pos.y = sortedPos[3 * index + 1];
        pos.z = sortedPos[3 * index + 2];
        //int3 gridPos;
        //gridPos.x = calcGridPos_x(pos.x);
        //gridPos.y = calcGridPos_y(pos.y);
        //gridPos.z = calcGridPos_z(pos.z);
        //int3 newgridPos;
        //int grid_count = 0;
        //int par_count = 0;

        for (int z = -1; z <= 1; z++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {
                    //newgridPos.x = gridPos.x + x;
                    //newgridPos.y = gridPos.y + y;
                    //newgridPos.z = gridPos.z + z;
                    //int newgridHash = calcGridHash(newgridPos);
                    //int newgridHash = calcGridHash_new(gridPos.x + x,gridPos.y + y,gridPos.z + z);
                    int newgridHash = particleHash[index] + calcGridHash_new(x,y,z);
                    if (newgridHash <= par.hash_max && newgridHash >= 0)
                    {
                        //int startIndex = cellStart[newgridHash];
                        #define startIndex (cellStart[newgridHash])
                        if (startIndex == 0xffffffff)	continue;
                        //int endIndex = cellEnd[newgridHash];
                        #define endIndex (cellEnd[newgridHash])
                        //  iterate over particles in this cell
                        for (int i = startIndex; i < endIndex; i++)
                        {
                            #undef startIndex
                            #undef endIndex
                            //int cellData = particleHash[i];
                            //if (cellData != newgridHash)  break;
                            if (i != index)	// check not colliding with self
                            {
                                
                                //float3 pos2; 
                                //float rr, drx, dry, drz;
                                //pos2.x = sortedPos[3 * i];
                                //pos2.y = sortedPos[3 * i + 1];
                                //pos2.z = sortedPos[3 * i + 2];
                                #define pos2_x sortedPos[3*i]
                                #define pos2_y sortedPos[3*i+1]
                                #define pos2_z sortedPos[3*i+2]
                                
                               /*
                                __shared__ float pos2[256*3];
                                pos2[threadIdx.x] = sortedPos[3*i];
                                pos2[threadIdx.x+256] = sortedPos[3*i+1];
                                pos2[threadIdx.x+512] = sortedPos[3*i+2];
                                #define drx (pos.x - pos2[threadIdx.x])
                                #define dry (pos.y - pos2[threadIdx.x+256])
                                #define drz (pos.z - pos2[threadIdx.x+512])
                                */
                                #define drx (pos.x - pos2_x)
                                #define dry (pos.y - pos2_y)
                                #define drz (pos.z - pos2_z)
                                #define rr (sqrtf(drx*drx + dry*dry + drz*drz))
                                //drx = pos.x - pos2.x; dry = pos.y - pos2.y; drz = pos.z - pos2.z;
                                //rr = sqrt(drx * drx + dry * dry + drz * drz);
                                float w, fr;
                                //float frx, fry, frz, factor1, factor2, factor3, factor4;
                                float q = rr / par.h;
                                //grid_count++;

                                if (rr < par.kh)
                                {
                                    if (q <= 2.0f)
                                    {
                                        w = (par.adh * powf(1.0f - q / 2.0f, 4) * (2.0f * q + 1.0f));
                                        #define factor1 (powf(1.0f - q / 2.0f, 3))
                                        #define factor2 (2.0f * q + 1.0f)
                                        #define factor3 (powf(1.0f - q / 2.0f, 4))
                                        #define factor4  (par.h * rr)
                                        
                                        fr =  (par.adh * (-2.0f * factor1 * factor2/ factor4 + 2.0f * factor3 / factor4));
                                        //#define  frx  (par.adh * (-2.0 * factor1 * factor2 * drx / factor4 + 2.0 * factor3 * drx / factor4));
                                        //#define  fry  (par.adh * (-2.0 * factor1 * factor2 * dry / factor4 + 2.0 * factor3 * dry / factor4));
                                        //#define  frz  (par.adh * (-2.0 * factor1 * factor2 * drz / factor4 + 2.0 * factor3 * drz / factor4));
                                        //par_count++;
                                    }
                                    else
                                    {
                                        w = 0.0f;
                                        fr = 0.0f;
                                        //frx = 0.0;
                                        //fry = 0.0;
                                        //frz = 0.0;
                                    }
                                    #define  frx  (fr*drx)
                                    #define  fry  (fr*dry)
                                    #define  frz  (fr*drz)
                                    if (sorted_particle_type[index] != 1 && sorted_particle_type[i] == 1)//计算边界所需变量
                                    {
                                        rhop_sum_tmp += (sortedpressure[i] - sorteddensity[i] * (0.0f * drx + 0.0f * dry + (0.0f - par.gravity) * drz)) * w;
                                        w_sum_tmp += w;
                                    }
                                    else if (sorted_particle_type[index] == 1 && sorted_particle_type[i] == 1)
                                    {
                                        #define factor5  ((sortedVel[3 * index] - sortedVel[3 * i]) * frx + (sortedVel[3 * index + 1] - sortedVel[3 * i + 1]) * fry + (sortedVel[3 * index + 2] - sortedVel[3 * i + 2]) * frz)
                                        dofv_tmp -= factor5 * par.particleMass / sorteddensity[i];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        dofv[index] = dofv_tmp;
        //if(par_count > 128) printf("the ptc :%d has %d pars and it's grid has %d ptcs!\n",index,par_count,grid_count);
        if (sorted_particle_type[index] != 1)
        {
            //if (fabs(w_sum[index]) > 1.0E-8)
            if (fabs(w_sum_tmp) > 1.0E-8f)
            {
                //sortedpressure[index] = rhop_sum[index] / w_sum[index];
                rhop_sum_tmp = rhop_sum_tmp / w_sum_tmp;
            }
            else
            {
                //sortedpressure[index] = 0;
                rhop_sum_tmp = 0.0f;
            }
            //if (sortedpressure[index] < 0)  sortedpressure[index] = 0;
            if(rhop_sum_tmp < 0.0f) rhop_sum_tmp = 0.0f;
            sortedpressure[index] = rhop_sum_tmp;
            sorteddensity[index] = rhop_sum_tmp / par.cs / par.cs + par.restDensity;
        }
    }
    #undef drx
    #undef dry
    #undef drz
    #undef rr
    #undef factor1
    #undef factor2
    #undef factor3
    #undef factor4
    #undef factor5
    #undef frx
    #undef fry
    #undef frz
}


__global__ void computeGovering_equationD(float* sortedPos, float* sortedVel, float* sortedpressure, float* sorteddensity, int* sorted_particle_type, float* densitydt, float* dofv, float* Veldt, int* cellStart, int* cellEnd, int numParticles, int* particleHash)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numParticles)
    {
        //densitydt[index] = 0.0;
        //Veldt[3 * index] = 0.0;
        //Veldt[3 * index + 1] = 0.0;
        //Veldt[3 * index + 2] = 0.0;
        float3 pos, vel;
        pos.x = sortedPos[3 * index];
        pos.y = sortedPos[3 * index + 1];
        pos.z = sortedPos[3 * index + 2];
        vel.x = sortedVel[3 * index];
        vel.y = sortedVel[3 * index + 1];
        vel.z = sortedVel[3 * index + 2];
        float pres = sortedpressure[index];	float dens = sorteddensity[index];
        //int3 gridPos = calcGridPos(pos);
        //int3 gridPos;
        //gridPos.x = calcGridPos_x(pos.x);
        //gridPos.y = calcGridPos_y(pos.y);
        //gridPos.z = calcGridPos_z(pos.z);
        //int3 newgridPos;
        float densitydt_temp = 0.0f;
        float3 veldt_temp = make_float3(0.0f, 0.0f, 0.0f);

        for (int z = -1; z <= 1; z++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {
                    //newgridPos.x = gridPos.x + x;
                    //newgridPos.y = gridPos.y + y;
                    //newgridPos.z = gridPos.z + z;
                    //int gridHash = calcGridHash(newgridPos);
                    //gridPos.x += x;
                    //gridPos.y += y;
                    //gridPos.z += z;
                    //int gridHash = calcGridHash_new(gridPos.x+x,gridPos.y+y,gridPos.z+z);
                    int gridHash = particleHash[index] + calcGridHash_new(x,y,z);

                    if (gridHash <= par.hash_max && gridHash >= 0)
                    {
                        //int startIndex = cellStart[gridHash];
                        #define startIndex cellStart[gridHash]
                        if (startIndex == 0xffffffff)	continue;
                        //int endIndex = cellEnd[gridHash];
                        #define endIndex cellEnd[gridHash]
                        //  iterate over particles in this cell
                        for (int i = startIndex; i < endIndex; i++)
                        {
                            #undef startIndex
                            #undef endIndex
                            //int cellData = particleHash[i];
                            //if (cellData != gridHash)  break;
                            if (i != index)	// check not colliding with self
                            {
                                //float3 pos2; 
                                //float rr, drx, dry, drz;
                                //pos2.x = sortedPos[3 * i];
                                //pos2.y = sortedPos[3 * i + 1];
                                //pos2.z = sortedPos[3 * i + 2];
                                #define pos_x sortedPos[3*i]
                                #define pos_y sortedPos[3*i+1]
                                #define pos_z sortedPos[3*i+2]
                                #define drx (pos.x - pos2_x)
                                #define dry (pos.y - pos2_y)
                                #define drz (pos.z - pos2_z)
                                
                               /*
                               __shared__ float pos2[256*3];
                               pos2[threadIdx.x] = sortedPos[3*i];
                               pos2[threadIdx.x+256] = sortedPos[3*i+1];
                               pos2[threadIdx.x+512] = sortedPos[3*i+2];
                                #define drx (pos.x - pos2[threadIdx.x])
                                #define dry (pos.y - pos2[threadIdx.x+256])
                                #define drz (pos.z - pos2[threadIdx.x+512])
                                */
                                #define rr (sqrtf(drx * drx + dry * dry + drz * drz))
                                //drx = pos.x - pos2.x; dry = pos.y - pos2.y; drz = pos.z - pos2.z;
                                //rr = sqrt(drx * drx + dry * dry + drz * drz);
                                if (rr < par.kh)
                                {
                                    //float frx, fry, frz
                                    float fr;
                                    //float factor1, factor2, factor3, factor4;
                                    float q = rr / par.h;
                                    if (q <= 2.0f)
                                    {
                                        #define factor1 (powf(1.0f - q / 2.0f, 3))
                                        #define factor2 (2.0f * q + 1.0f)
                                        #define factor3 (powf(1.0f - q / 2.0f, 4))
                                        #define factor4 (par.h * rr)
                                        fr = par.adh * (-2.0f * factor1 * factor2/factor4 + 2.0f * factor3/factor4);
                                        //frx = par.adh * (-2.0 * factor1 * factor2 * drx / factor4 + 2.0 * factor3 * drx / factor4);
                                        //fry = par.adh * (-2.0 * factor1 * factor2 * dry / factor4 + 2.0 * factor3 * dry / factor4);
                                        //frz = par.adh * (-2.0 * factor1 * factor2 * drz / factor4 + 2.0 * factor3 * drz / factor4);
                                    }
                                    else
                                    {
                                        fr = 0.0f;
                                        //frx = 0.0;
                                        //fry = 0.0;
                                        //frz = 0.0;
                                    }
                                    #define frx (fr * drx)
                                    #define fry (fr * dry)
                                    #define frz (fr * drz)
                                    //质量方程
                                    //factor1 = (vel.x - sortedVel[3 * i]) * frx + (vel.y - sortedVel[3 * i + 1]) * fry + (vel.z - sortedVel[3 * i + 2]) * frz;
                                    #define factor5  ((vel.x - sortedVel[3 * i]) * frx + (vel.y - sortedVel[3 * i + 1]) * fry + (vel.z - sortedVel[3 * i + 2]) * frz)
                                    densitydt_temp += dens * factor5 * par.particleMass / sorteddensity[i];
                                    
                                    //density diffusion
                                    //factor1 = drx * frx + dry * fry + drz * frz;
                                    //factor2 = rr * rr + par.eta * par.eta;
                                    //factor3 = par.delta * par.h * par.cs * factor1 / factor2;
                                    #undef factor4
                                    #define factor4 (drx * frx + dry * fry + drz * frz)
                                    #undef factor5
                                    #define factor5 (rr * rr + par.eta * par.eta)
                                    #define factor6 (par.delta * par.h * par.cs * factor4 / factor5)
                                    densitydt_temp += (dens - sorteddensity[i]) * factor6 * par.particleMass / sorteddensity[i];
                                    //动量方程
                                    //factor1 = (sortedpressure[i] + pres) / dens / sorteddensity[i];
                                    #define factor7 ((sortedpressure[i] + pres) / dens / sorteddensity[i])
                                    veldt_temp.x -= par.particleMass * factor7 * frx;
                                    veldt_temp.y -= par.particleMass * factor7 * fry;
                                    veldt_temp.z -= par.particleMass * factor7 * frz;

                                    if (sorted_particle_type[index] == 1 && sorted_particle_type[i] == 1)
                                    {
                                        //acoustic damper
                                        //factor1 = par.restDensity * par.cs * par.h;
                                        //factor2 = dofv[index] + dofv[i];
                                        #define factor8 (par.restDensity * par.cs * par.h)
                                        #define factor9 (dofv[index] + dofv[i])
                                        veldt_temp.x += factor8 * factor9 * frx * par.particleMass / sorteddensity[i] / dens;
                                        veldt_temp.y += factor8 * factor9 * fry * par.particleMass / sorteddensity[i] / dens;
                                        veldt_temp.z += factor8 * factor9 * frz * par.particleMass / sorteddensity[i] / dens;

                                        //artificial viscosity
                                        //factor1 = par.afa * par.h * par.cs;
                                        //factor2 = (vel.x - sortedVel[3 * i]) * drx + (vel.y - sortedVel[3 * i + 1]) * dry + (vel.z - sortedVel[3 * i + 2]) * drz;
                                        //factor3 = rr * rr + par.eta * par.eta;
                                        //factor4 = factor1 * factor2 / factor3;
                                        #define factor10 (par.afa * par.h * par.cs)
                                        #define factor11 ((vel.x - sortedVel[3 * i]) * drx + (vel.y - sortedVel[3 * i + 1]) * dry + (vel.z - sortedVel[3 * i + 2]) * drz)
                                        #define factor12 (rr * rr + par.eta * par.eta)
                                        #define factor13 (factor10 * factor11 / factor12)
                                        veldt_temp.x += factor13 * frx * par.restDensity / dens * par.particleMass / sorteddensity[i];
                                        veldt_temp.y += factor13 * fry * par.restDensity / dens * par.particleMass / sorteddensity[i];
                                        veldt_temp.z += factor13 * frz * par.restDensity / dens * par.particleMass / sorteddensity[i];
                                    }
                                }
                            }
                        }
                    }
                   //gridPos.x -= x;
                   // gridPos.y -= y;
                   // gridPos.z -= z;
                }
            }
        }
        densitydt[index] = densitydt_temp;
        Veldt[3 * index] = veldt_temp.x;
        Veldt[3 * index + 1] = veldt_temp.y;
        Veldt[3 * index + 2] = veldt_temp.z + par.gravity;
        #undef drx
        #undef dry
        #undef drz
        #undef rr
        #undef frx
        #undef fry
        #undef frz
        #undef factor1
        #undef factor2
        #undef factor3
        #undef factor4
        #undef factor5
        #undef factor6
        #undef factor7
        #undef factor8
        #undef factor9
        #undef factor10
        #undef factor11
        #undef factor12
        #undef factor13
    }
}