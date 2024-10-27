#include "io.cuh"

void mul_thread_creat(cpu_thread_t *thread_pool,cpu_param_t *param)
{
    for(int i=0;i<param->thread_num;i++)
    {
        thread_pool[i].data.pos_rho = new float[param->ptc_num*4];
        thread_pool[i].data.vel_p = new float[param->ptc_num*4];
        thread_pool[i].data.acc_drhodt = new float[param->ptc_num*4];
        thread_pool[i].data.type = new int[param->ptc_num];
        thread_pool[i].data.table = new int[param->ptc_num];
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
            fname = param->output_path + "/zsph" + std::to_string(thread->current_step)+".vtk";
            write_vtk(fname.c_str(),&(thread->data), param);
            thread->write_flag.compare_exchange_strong(ref_2,0);
        }
        ref_1 = 1;
        ref_2 = 2;
        if(thread->end_flag.compare_exchange_strong(ref_end,0))
        {
            delete[] thread->data.pos_rho;
            delete[] thread->data.vel_p;
            delete[] thread->data.acc_drhodt;
            delete[] thread->data.type;
            delete[] thread->data.table;
            exit(0);
        }
        ref_end = 1;
    }   
}