#include "equation.cuh"
#include "io.cuh"
#include "lib.cuh"

__constant__ gpu_param_t par;
__constant__ gpu_tmp_t tmp_data;

int main(void)
{
    cpu_json_t jdata;
    cpu_param_t cparam;
    cpu_input_t cdata; 
    read_json("../input/input.json", &jdata);
    set_cpu_param(&cparam, &jdata);
    alloc_cpu_data(&cdata, &cparam);
    read_vtk(&cdata, &cparam);
    check_param(&cparam);

    //for(int j=0;j<cparam.ptc_num;j++)
    //{
        //std::cout << cdata.pos_rho[j*4] << " " << cdata.pos_rho[j*4+1] << " " << cdata.pos_rho[j*4+2] << " " << cdata.pos_rho[j*4+3]<< std::endl;
    //}

    check_gpu(&cparam); 
    cudaDeviceReset();
    cudaSetDevice(cparam.gpu_id);

    gpu_param_t h_gparam;
    gpu_ptc_t h_old_gptc_data;
    gpu_ptc_t h_new_gptc_data;
    gpu_tmp_t h_gtmp_data;

    set_gpu_param(&h_gparam, &cparam);

    alloc_gpu_ptc_data(&h_old_gptc_data, &cparam);
    alloc_gpu_ptc_data(&h_new_gptc_data, &cparam);
    alloc_gpu_tmp_data(&h_gtmp_data, &cparam);

    cpu_to_gpu(&h_old_gptc_data,&h_gtmp_data,&cdata,&cparam);

    gpu_ptc_t *d_old_gptc_data; cudaMalloc(&d_old_gptc_data,sizeof(gpu_ptc_t));
    gpu_ptc_t *d_new_gptc_data; cudaMalloc(&d_new_gptc_data,sizeof(gpu_ptc_t));

    cudaMemcpy(d_old_gptc_data,&h_old_gptc_data,sizeof(gpu_ptc_t),cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_gptc_data,&h_new_gptc_data,sizeof(gpu_ptc_t),cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(par,&h_gparam,sizeof(gpu_param_t));
    cudaMemcpyToSymbol(tmp_data,&h_gtmp_data,sizeof(gpu_tmp_t));

    check_gerr(__FILE__,__LINE__);
    
    cpu_thread_t cthread[cparam.thread_num];
    mul_thread_creat(cthread,&cparam);

    int grid = (cparam.ptc_num+255)/256;
    int block = 256;

    for(int i=cparam.start_step;i<cparam.end_step;i++)
    {
        std::cout << "Zsph is calculating the " << i << " steps... " ;
        auto time_prob_0 = std::chrono::high_resolution_clock::now();

        cuda_ptc_hash<<<grid,block>>>(d_old_gptc_data);
        //check_gerr(__FILE__,__LINE__);
        cuda_sort_index(h_gtmp_data,cparam);
        //check_gerr(__FILE__,__LINE__);
        cuda_sort_data<<<grid,block>>>(d_old_gptc_data,d_new_gptc_data);
        //check_gerr(__FILE__,__LINE__);
        cuda_boundary_ns<<<grid,block>>>(d_new_gptc_data);
        //check_gerr(__FILE__,__LINE__);
        cuda_govering_ns<<<grid,block>>>(d_new_gptc_data);
        //check_gerr(__FILE__,__LINE__);
        cuda_prediction<<<grid,block>>>(d_new_gptc_data);
        check_gerr(__FILE__,__LINE__);

        cudaDeviceSynchronize();
        auto time_prob_1 = std::chrono::high_resolution_clock::now();

        cuda_ptc_hash<<<grid,block>>>(d_new_gptc_data);
       // check_gerr(__FILE__,__LINE__);
        cuda_sort_index(h_gtmp_data,cparam);
        cuda_sort_data<<<grid,block>>>(d_new_gptc_data,d_old_gptc_data);
        //check_gerr(__FILE__,__LINE__);
        cuda_boundary_ns<<<grid,block>>>(d_old_gptc_data);
        //check_gerr(__FILE__,__LINE__);
        cuda_govering_ns<<<grid,block>>>(d_old_gptc_data);
        //check_gerr(__FILE__,__LINE__);
        cuda_correction<<<grid,block>>>(d_old_gptc_data);
        check_gerr(__FILE__,__LINE__);
        
        auto time_prob_2 = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> prediction_time = time_prob_1 - time_prob_0;
        std::chrono::duration<double> correction_time = time_prob_2 - time_prob_1;
        std::cout << "prediton step use:" << prediction_time.count() <<" correction step use:" << correction_time.count() << std::endl;

        int expect_0 = 0;
        int &ref_0 = expect_0;
        bool flag = true;

        if(i%cparam.output_step==0)
        {
            //std::cout << "current i is:" << i << " current ref_0 is:" << ref_0 << std::endl;
            flag = true;
            while(flag)
            {

                for(int j=0;j<cparam.thread_num;j++)
                {
                    ref_0 = 0;
                    if(cthread[j].write_flag.compare_exchange_strong(ref_0,0))
                    {
                        cparam.current_step = i;
                        cthread[j].current_step = i;
                        /*
                        cudaMemcpy(cthread[j].data.pos,h_old_gptc_data.pos,cparam.ptc_num*3*sizeof(float),cudaMemcpyDeviceToHost);
                        cudaMemcpy(cthread[j].data.vel,h_old_gptc_data.vel,cparam.ptc_num*3*sizeof(float),cudaMemcpyDeviceToHost);
                        cudaMemcpy(cthread[j].data.rhop,h_old_gptc_data.rhop,cparam.ptc_num*2*sizeof(float),cudaMemcpyDeviceToHost);
                        cudaMemcpy(cthread[j].data.type,h_old_gptc_data.type,cparam.ptc_num*sizeof(int),cudaMemcpyDeviceToHost);
                        cudaMemcpy(cthread[j].data.table,h_old_gptc_data.table,cparam.ptc_num*sizeof(int),cudaMemcpyDeviceToHost);
                        
                        check_gerr(__FILE__,__LINE__);
                        */
                        gpu_to_cpu(&(cthread[j].data),&h_old_gptc_data,&h_gtmp_data,&cparam);
                        flag = false;
                        //std::cout << "ref_0 is :" << ref_0 << std::endl;
                        cthread[j].write_flag.compare_exchange_strong(ref_0,1);
                    }
                }
            }
            
            
            /*
            std::string filename = cparam.output_path+"/zsph"+std::to_string(i)+".vtk";
            cudaMemcpy(cdata.pos_rho,h_old_gptc_data.pos_rho,sizeof(float)*4*cparam.ptc_num,cudaMemcpyDeviceToHost);
            cudaMemcpy(cdata.vel_p,h_old_gptc_data.vel_p,sizeof(float)*4*cparam.ptc_num,cudaMemcpyDeviceToHost);
            cudaMemcpy(cdata.type,h_old_gptc_data.type,sizeof(int)*cparam.ptc_num,cudaMemcpyDeviceToHost);
            cudaMemcpy(cdata.table,h_old_gptc_data.table,sizeof(int)*cparam.ptc_num,cudaMemcpyDeviceToHost);
            write_vtk(filename,&cdata,&cparam);
            */
            
        }
    }    
    delete_cpu_data(&cdata);
    delete_gpu_ptc_data(&h_old_gptc_data);
    delete_gpu_ptc_data(&h_new_gptc_data);
    delete_gpu_tmp_data(&h_gtmp_data);
    cudaDeviceReset();

    return 0;
}