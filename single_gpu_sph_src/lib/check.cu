#include "header/lib.cuh"

void check_dt(cpu_param_t *param)
{
    float dt_max =0.0f;
    dt_max = 0.25 * param->h / sqrt(param->g * (param->zmax - param->zmin));
    std::cout << "Zsph is checking dt ..." << std::endl;    
    if(param->dt <= dt_max)
    {
        std::cout << "Dt is less than the dt_max" << std::endl;
    }
    else
    {
        std::cerr << "Dt is greater than the dt_max" << std::endl;
        exit(1);
    }
}
void check_gpu(cpu_param_t *param)
{
    std::cout << "Zsph is checking gpu ..." << std::endl;
    cudaDeviceProp prop;  
    int deviceCount;  
    
    cudaDeviceReset();
    // get the gpu num  
    cudaGetDeviceCount(&deviceCount);  
  
    // is the gpu num equal 0  
    if (deviceCount == 0) {  
        std::cerr << "No CUDA-capable devices are available." << std::endl;  
        exit(1);  
    }  
  
    // check the first gpu prop  
    if (cudaGetDeviceProperties(&prop, param->gpu_id) == cudaSuccess) {  
        std::cout << "Device Name: " << prop.name << std::endl;  
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;  
        std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;  
        std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;  
        std::cout << "Warp Size: " << prop.warpSize << std::endl;  
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;  
        std::cout << "Max Grid Size (Dimension x,y,z): " << prop.maxGridSize[0]   
                  << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;  
        std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;  
        std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;  
        std::cout << "asyncEngineCount: " << prop.asyncEngineCount << std::endl;  
        std::cout << "persistingL2CacheMaxSize: " << prop.persistingL2CacheMaxSize << std::endl;  
        std::cout << "concurrentKernels: " << prop.concurrentKernels << std::endl;  
    
    } else {  
        std::cerr << "Failed to get CUDA device properties." << std::endl;
        exit(1);   
    }  
  
}

void check_gerr()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}