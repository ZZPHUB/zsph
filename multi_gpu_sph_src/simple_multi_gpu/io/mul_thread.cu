#include "io.cuh"

void mul_thread_creat(cpu_thread_t *thread_pool,cpu_param_t *param)
{
    for(int i=0;i<param->thread_num;i++)
    {
        thread_pool[i].data = (cpu_output_t *)calloc(param->gpu_num,sizeof(cpu_output_t));
        for(int j=0;j<param->gpu_num;j++)
        {
            thread_pool[i].data[j].pos = new float[param->gmax_size * 3];

            // vel
            #ifdef ZSPH_OUTPUT_VEL
                thread_pool[i].data[j].vel = new float[param->gmax_size * 3];
            #endif

            // rho and p
            #if (defined ZSPH_OUTPUT_RHO) || (defined ZSPH_OUTPUT_P)
                thread_pool[i].data[j].rhop = new float[param->gmax_size * 2];
            #endif

            // is ptc
            #ifdef ZSPH_OUTPUT_ISPTC
                thread_pool[i].data[j].is_ptc = new int[param->gmax_size];
            #endif

            // gpu id
            #ifdef ZSPH_OUTPUT_GPUID
                thread_pool[i].data[j].gpu_id = new int[param->gmax_size];
            #endif

            // acc and drhodt
            #if (defined ZSPH_OUTPUT_ACC) || (defined ZSPH_OUTPUT_DRHODT)
                thread_pool[i].data[j].acc_drhodt = new float[param->gmax_size * 4];
            #endif

            // type
            #ifdef ZSPH_OUTPUT_TYPE
                thread_pool[i].data[j].type = new int[param->gmax_size];
            #endif

            // table
            #ifdef ZSPH_OUTPUT_TABLE
                thread_pool[i].data[j].table = new int[param->gmax_size];
            #endif

            // wsum
            #ifdef ZSPH_OUTPUT_WSUM
                thread_pool[i].data[j].wsum = new float[param->gmax_size];
            #endif

            // hash
            #ifdef ZSPH_OUTPUT_HASH
                thread_pool[i].data[j].hash = new int[param->gmax_size];
            #endif
        }

        thread_pool[i].write_flag = 0;
        thread_pool[i].end_flag = 0;
        
        thread_pool[i].thread = std::thread(mul_thread_output,&thread_pool[i],param);
    }
}

void mul_thread_output(cpu_thread_t *thread,cpu_param_t *param)
{
    int expect_1 = 1;
    int expect_2 = 2;
    int expect_end = 1;
    int &ref_1 = expect_1;
    int &ref_2 = expect_2;
    int &ref_end = expect_end;
    std::string fname;
    while (true)
    {
        if(thread->write_flag.compare_exchange_strong(ref_1,2))
        {
            for(int i=0;i<param->gpu_num;i++)
            {
                fname = param->output_path + "/zsph-gpu" + std::to_string(i)+ "-" + std::to_string(thread->current_step)+".vtk";
                write_vtk(fname.c_str(),&(thread->data[i]));
            }
            thread->write_flag.compare_exchange_strong(ref_2,0);
        }
        ref_1 = 1;
        ref_2 = 2;
        if(thread->end_flag.compare_exchange_strong(ref_end,0))
        {
            for(int i=0;i<param->gpu_num;i++)
            {
                free(thread->data[i].pos);
                
                #ifdef ZSPH_OUTPUT_VEL
                free(thread->data[i].vel);
                #endif

                #if (defined ZSPH_OUTPUT_RHO) || (defined ZSPH_OUTPUT_P)
                free(thread->data[i].rhop);
                #endif
                
                #if (defined ZSPH_OUTPUT_ACC) || (defined ZSPH_OUTPUT_DRHODT)
                free(thread->data[i].acc_drhodt);
                #endif
                
                #ifdef ZSPH_OUTPUT_TABLE
                free(thread->data[i].table);
                #endif
                
                #ifdef ZSPH_OUTPUT_TYPE
                free(thread->data[i].type);
                #endif

                #ifdef ZSPH_OUTPUT_ISPTC
                free(thread->data[i].is_ptc);
                #endif

                #ifdef ZSPH_OUTPUT_GPUID
                free(thread->data[i].gpu_id);
                #endif

                #ifdef ZSPH_OUTPUT_HASH
                free(thread->data[i].hash);
                #endif

                #ifdef ZSPH_OUTPUT_WSUM
                free(thread->data[i].wsum);
                #endif
            }
            exit(0);
        }
        ref_end = 1;
    }   
}