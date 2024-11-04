#include "lib.cuh"

void check_dt(cpu_param_t *param)
{
    float dt_max =0.0f;
    dt_max = 0.25 * param->h / sqrt(-param->g * (param->zmax - param->zmin));
    std::cout << "*******************************************************************************************" << std::endl;
    std::cout << "* Zsph is checking dt ..." << std::endl;    
    if(param->dt <= dt_max)
    {
        std::cout << "* Dt checking is done ..." << std::endl;
    }
    else
    {
        std::cerr << "* Dt is greater than the dt_max" << std::endl;
        exit(1);
    }
}
void check_gpu(cpu_param_t *param)
{
    std::cout << "*******************************************************************************************" << std::endl;
    std::cout << "* Zsph is checking gpu ..." << std::endl;
    cudaDeviceProp prop;  
    int deviceCount;  
    
    cudaDeviceReset();
    // get the gpu num  
    cudaGetDeviceCount(&deviceCount);  
  
    // is the gpu num equal 0  
    if (deviceCount == 0) {  
        std::cerr << "* No CUDA-capable devices are available." << std::endl;  
        exit(1);  
    }  
  
    // check the first gpu prop  
    if (cudaGetDeviceProperties(&prop, param->gpu_id) == cudaSuccess) {  
        //std::cout << "*******************************************************************************************" << std::endl;
        std::cout << "* Device Name: " << prop.name << std::endl;  
        std::cout << "* Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;  
        std::cout << "* Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;  
        std::cout << "* Registers per Block: " << prop.regsPerBlock << std::endl;  
        std::cout << "* Warp Size: " << prop.warpSize << std::endl;  
        std::cout << "* Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;  
        std::cout << "* Max Grid Size (Dimension x,y,z): " << prop.maxGridSize[0]   
                  << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;  
        std::cout << "* Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;  
        std::cout << "* Multiprocessor Count: " << prop.multiProcessorCount << std::endl;  
        std::cout << "* asyncEngineCount: " << prop.asyncEngineCount << std::endl;  
        std::cout << "* persistingL2CacheMaxSize: " << prop.persistingL2CacheMaxSize << std::endl;  
        std::cout << "* concurrentKernels: " << prop.concurrentKernels << std::endl;  
        //std::cout << "*******************************************************************************************" << std::endl;
    
    } else {  
        std::cerr << "* Failed to get CUDA device properties." << std::endl;
        exit(1);   
    }  
  
}

void check_param(cpu_param_t *cparam)
{
    std::cout << "*******************************************************************************************" << std::endl;
    std::cout << "* Zsph is checking cpu_param_t..." <<std::endl;
    std::cout << "* g:" << cparam->g << std::endl;
    std::cout << "* m:" << cparam->m << std::endl;
    std::cout << "* rho0:" << cparam->rho0 << std::endl;
    std::cout << "* dx:" << cparam->dx << std::endl;
    std::cout << "* h:" << cparam->h << std::endl;
    std::cout << "* r:" << cparam->r << std::endl;
    std::cout << "* eta:" << cparam->eta << std::endl;
    std::cout << "* cs:" << cparam->cs << std::endl;
    std::cout << "* adh:" << cparam->adh << std::endl;
    std::cout << "* delta:" << cparam->delta << std::endl;
    std::cout << "* alpha:" << cparam->alpha << std::endl;
    std::cout << "* dt:" << cparam->dt << std::endl;
    std::cout << "* start_step:" << cparam->start_step << std::endl;
    std::cout << "* current_step:" << cparam->current_step << std::endl;
    std::cout << "* end_step:" << cparam->end_step  << std::endl;
    std::cout << "* output_step:" << cparam->output_step << std::endl;
    std::cout << "* xmin:" << cparam->xmin << " xmax:" << cparam->xmax << " ymin:" << cparam->ymin << " ymax:" << cparam->ymax <<" zmin:" << cparam->zmin << " zmax:" << cparam->zmax << std::endl;
    std::cout << "* grid_size:" << cparam->grid_size << std::endl;
    std::cout << "* grid_num:" << cparam->grid_num << " grid_hash_min:" << cparam->grid_hash_min << " grid_hash_max:" << cparam->grid_hash_max << std::endl;
    std::cout << "* ptc_num:" << cparam->ptc_num << std::endl;
    std::cout << "* gpu_id" << cparam->gpu_id << std::endl;
    //std::cout << "*******************************************************************************************" << std::endl;
}

void check_json(cpu_json_t *cjson)
{
    std::cout << "*******************************************************************************************" << std::endl;
    std::cout << "* Zsph is checking gpu_param_t..." << std::endl;
    std::cout << "* NO IMPLEMENTION" << std::endl;
}

void check_gerr(const char *a,const int b)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error in " << a << ":" << b << " is:" << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}