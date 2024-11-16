#include "lib.cuh"
#include "io.cuh"
#include "equation.cuh"

__constant__ gpu_tmp_t gtmp_data;
__constant__ gpu_param_t par;

int main(void)
{
    cpu_json_t jdata;
    cpu_param_t cparam;
    cpu_input_t cintput_dat;

    //read setting from json
    read_json("../input/input.json",&jdata);

    //set cpu param
    set_cpu_param(&cparam,&jdata);

    //check param 
    check_param(&cparam);

    //alloc data for reading from vtk
    alloc_cpu_input(&cintput_dat,&cparam);

    //read data from vtk
    read_vtk(&cintput_dat,&cparam);

    /****************set gpu param for each gpu***************/
    gpu_param_t h_gparam[cparam.gpu_num];
    set_gpu_param(h_gparam,&cparam);

    /**************split data to sub data*********************/
    cpu_input_t ginput_dat[cparam.gpu_num];
    alloc_gpu_intput(ginput_dat,&cparam);

    //gpu_socket_t h_socket[cparam.gpu_num];
    /*
    gpu_socket_t *gsocket[cparam.gpu_num];
    for(int i=0;i<cparam.gpu_num;i++)
    {
        cudaMallocManaged(&(gsocket[i]),sizeof(gpu_socket_t));
    }
    */

   gpu_socket_t *h_socket_dat[cparam.gpu_num];
   gpu_socket_t *d_socket_dat[cparam.gpu_num];
   for(int i=0;i<cparam.gpu_num;i++)
   {
        h_socket_dat[i] = (gpu_socket_t *)calloc(1,sizeof(gpu_socket_t));
        cudaSetDevice(i);
        cudaMalloc(&(d_socket_dat[i]),sizeof(gpu_socket_t));
   }

    split_in_cpu(ginput_dat,&cintput_dat,&cparam,h_gparam,h_socket_dat);
    
    gpu_ptc_t h_old_gptc_data[cparam.gpu_num];
    gpu_ptc_t h_new_gptc_data[cparam.gpu_num];
    //alloc_gptc_data
    alloc_gptc_data(h_old_gptc_data,&cparam);
    alloc_gptc_data(h_new_gptc_data,&cparam);

    gpu_tmp_t h_gtmp_data[cparam.gpu_num];
    //alloc_gtmp_data
    alloc_gtmp_data(h_gtmp_data,&cparam,h_gparam);

    //cpu to gpu
    cpu_to_gpu(h_old_gptc_data,h_gtmp_data,ginput_dat,&cparam,h_socket_dat);

    gpu_ptc_t* d_old_gptc_data[cparam.gpu_num];
    gpu_ptc_t* d_new_gptc_data[cparam.gpu_num];
    for(int i=0;i<cparam.gpu_num;i++)
    {
        cudaSetDevice(i);
        cudaMalloc(&(d_old_gptc_data[i]),sizeof(gpu_ptc_t));
        cudaMalloc(&(d_new_gptc_data[i]),sizeof(gpu_ptc_t));
        cudaMemcpy(d_old_gptc_data[i],&(h_old_gptc_data[i]),sizeof(gpu_ptc_t),cudaMemcpyHostToDevice);
        cudaMemcpy(d_new_gptc_data[i],&(h_new_gptc_data[i]),sizeof(gpu_ptc_t),cudaMemcpyHostToDevice);
        
        cudaMemcpyToSymbol(gtmp_data,&(h_gtmp_data[i]),sizeof(gpu_tmp_t));
        cudaMemcpyToSymbol(par,&(h_gparam[i]),sizeof(gpu_param_t));
        check_gerr(__FILE__,__LINE__);

    }

    int block = 256;
    int grid = (int)((cparam.gmax_size+255)/256);

    cpu_thread_t cthread[cparam.thread_num];
    mul_thread_creat(cthread,&cparam);

    bool flag = true;
    int expect_0 = 0;
    int &ref_0 = expect_0;

    init_p2p(&cparam);

    for(int step = cparam.start_step;step<cparam.end_step;step++)
    {
        std::cout << "* Zsph is at " << step << " steps..." << std::endl;
        //**********************************split*********************************//
        //init
        init_socket(d_socket_dat,&cparam);
        
        //hash
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            hash_split<<<grid,block>>>(d_old_gptc_data[i]);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);
        //sort index by hash
        sort_split_index(h_gtmp_data,&cparam);
        
        //sort data
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            sort_split_data<<<grid,block>>>(d_new_gptc_data[i],d_old_gptc_data[i]);
            //check_gerr(__FILE__,__LINE__);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);

        //find the address and num of send data,the and updata the total_ptc_num
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            find_split_data<<<grid,block>>>(d_new_gptc_data[i],d_socket_dat[i]);
            //check_gerr(__FILE__,__LINE__);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);

        //synchronize for data transport
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }

        //**********************************data transport*************************//
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            cudaMemcpy(h_socket_dat[i],d_socket_dat[i],sizeof(gpu_socket_t),cudaMemcpyDeviceToHost);
        }

        /*
        for(int i=0;i<cparam.gpu_num;i++)
        {
            std::cout << "* Gpu "<< i << " :" << std::setw(10) << h_socket_dat[i]->send_ptc_num << std::setw(10) << h_socket_dat[i]->outer_ptc_num \
            << std::setw(10) << h_socket_dat[i]->total_ptc_num << std::setw(10) << cparam.gmax_size << std::endl;
        }*/

        //copy data to gpus
        peer_to_peer(h_socket_dat,&cparam);
        check_mulgerr(__FILE__,__LINE__,&cparam);

        //************************************write file*************************** */
        if(step%cparam.output_step==0)
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
                        cparam.current_step = step;
                        cthread[j].current_step = step;
                        gpu_to_cpu(cthread[0].data,h_new_gptc_data,h_gtmp_data,&cparam,h_socket_dat);
                        flag = false;
                        //std::cout << "ref_0 is :" << ref_0 << std::endl;
                        cthread[j].write_flag.compare_exchange_strong(ref_0,1);
                    }
                }
            }
        }

        //*****************************calculate***********************************//
        //hash
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            hash_calcu<<<grid,block>>>(d_new_gptc_data[i]);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);

        //sort index by hash
        sort_calcu_index(h_gtmp_data,&cparam);

        //sort data and find cell start and end
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            cudaMemset(h_gtmp_data[i].grid_start,-1,sizeof(int)*h_gparam[i].grid_hash_max);
            cudaMemset(h_gtmp_data[i].grid_end,-1,sizeof(int)*h_gparam[i].grid_hash_max);
            sort_calcu_data<<<grid,block>>>(d_old_gptc_data[i],d_new_gptc_data[i]);
            //check_gerr(__FILE__,__LINE__);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);

        //governing
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            cuda_governing_ns<<<grid,block>>>(d_old_gptc_data[i]);
            //check_gerr(__FILE__,__LINE__);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);

        
        //time integration for prediction
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            cuda_prediction<<<grid,block>>>(d_old_gptc_data[i]);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);
        

        /************************************************************************************************************************************************************/

        //**********************************split********************************
        //init
        init_socket(d_socket_dat,&cparam);
        
        //hash
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            hash_split<<<grid,block>>>(d_old_gptc_data[i]);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);
        //sort index by hash
        sort_split_index(h_gtmp_data,&cparam);
        
        //sort data
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            sort_split_data<<<grid,block>>>(d_new_gptc_data[i],d_old_gptc_data[i]);
            //check_gerr(__FILE__,__LINE__);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);

        //find the address and num of send data,the and updata the total_ptc_num
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            find_split_data<<<grid,block>>>(d_new_gptc_data[i],d_socket_dat[i]);
            //check_gerr(__FILE__,__LINE__);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);

        //synchronize for data transport
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }

        //**********************************data transport*************************//
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            cudaMemcpy(h_socket_dat[i],d_socket_dat[i],sizeof(gpu_socket_t),cudaMemcpyDeviceToHost);
        }

        /*
        for(int i=0;i<cparam.gpu_num;i++)
        {
            std::cout << "* Gpu "<< i << " :" << std::setw(10) << h_socket_dat[i]->send_ptc_num << std::setw(10) << h_socket_dat[i]->outer_ptc_num \
            << std::setw(10) << h_socket_dat[i]->total_ptc_num << std::setw(10) << cparam.gmax_size << std::endl;
        }*/

        //copy data to gpus
        peer_to_peer(h_socket_dat,&cparam);
        check_mulgerr(__FILE__,__LINE__,&cparam);

        //*****************************calculate***********************************//
        //hash
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            hash_calcu<<<grid,block>>>(d_new_gptc_data[i]);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);

        //sort index by hash
        sort_calcu_index(h_gtmp_data,&cparam);

        //sort data and find cell start and end
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            cudaMemset(h_gtmp_data[i].grid_start,-1,sizeof(int)*h_gparam[i].grid_hash_max);
            cudaMemset(h_gtmp_data[i].grid_end,-1,sizeof(int)*h_gparam[i].grid_hash_max);
            sort_calcu_data<<<grid,block>>>(d_old_gptc_data[i],d_new_gptc_data[i]);
            //check_gerr(__FILE__,__LINE__);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);

        //governing
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            cuda_governing_ns<<<grid,block>>>(d_old_gptc_data[i]);
            //check_gerr(__FILE__,__LINE__);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);

        
        //time integration for prediction
        for(int i=0;i<cparam.gpu_num;i++)
        {
            cudaSetDevice(i);
            cuda_correction<<<grid,block>>>(d_old_gptc_data[i]);
        }
        check_mulgerr(__FILE__,__LINE__,&cparam);
    }
    
    for(int i=0;i<cparam.thread_num;i++)
    {
	    cthread[i].end_flag = 1;
    }
    sleep(2);
    
   
}
