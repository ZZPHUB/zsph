#include "lib.cuh"

void set_cpu_param(cpu_param_t *param, cpu_json_t *jdata)
{
    // computional setting
    param->g = jdata->g;
    param->m = jdata->rho0 * jdata->dx * jdata->dx * jdata->dx;
    param->rho0 = jdata->rho0;
    param->dx = jdata->dx;
    param->h = jdata->h_factor * jdata->dx;
    param->r = jdata->r_factor * jdata->h_factor * jdata->dx;
    param->eta = jdata->eta_factor * jdata->h_factor * jdata->dx;
    param->cs = jdata->cs_factor * sqrt(abs(jdata->g) * (jdata->zmax - jdata->zmin));
    param->delta = jdata->delta;
    param->alpha = jdata->alpha;
    param->adh = 21.0f / (16.0f * 3.1415926f * param->h * param->h * param->h);

    // domain setting
    param->xmin = jdata->xmin - 1.1f*param->dx;
    param->xmax = jdata->xmax + 1.1f*param->dx;
    param->ymin = jdata->ymin - 1.1f*param->dx;
    param->ymax = jdata->ymax + 1.1f*param->dx;
    param->zmin = jdata->zmin - 1.1f*param->dx;
    param->zmax = jdata->zmax + 1.1f*param->dx;

    // grid setting
    param->grid_size = jdata->r_factor * jdata->h_factor * jdata->dx / jdata->grid_size_factor;
    param->grid_factor = (int)jdata->grid_size_factor;

    // ptc setting
    param->ptc_num = jdata->ptc_num;

    // thread setting
    param->thread_num = jdata->thread_num;

    // gpu num
    param->gpu_num = jdata->gpu_num;
    param->gmax_size = (int)(2.0f*(float)param->ptc_num / param->gpu_num);

    // path setting
    param->input_path = jdata->input_path;
    param->output_path = jdata->output_path;
    param->git_hash = jdata->git_hash;

    // time setting
    param->dt = jdata->dt;
    param->start_step = jdata->start_step;
    param->current_step = jdata->current_step;
    param->end_step = jdata->end_step;
    param->output_step = jdata->output_step;

    // check time step setting
    check_dt(param);
}

void set_gpu_param(gpu_param_t *gparam, cpu_param_t *cparam)
{
    //int grid_xzdim = cparam->grid_xdim * cparam->grid_ydim;
    //int hash_num = (int)(cparam->grid_ydim/cparam->gpu_num);
    float sub_len = (cparam->ymax - cparam->ymin)/((float)cparam->gpu_num);
    for (int i = 0; i < cparam->gpu_num; i++)
    {

        gparam[i].half_dt = cparam->dt / 2.0f;
        gparam[i].h = cparam->h;
        gparam[i].g = cparam->g;
        gparam[i].m = cparam->m;
        gparam[i].rho0 = cparam->rho0;
        gparam[i].adh = cparam->adh;
        gparam[i].h2 = cparam->h * cparam->h;
        gparam[i].eta2 = cparam->eta * cparam->eta;
        gparam[i].cs2 = cparam->cs * cparam->cs;
        gparam[i].alpha = cparam->alpha;
        gparam[i].delta_h_cs = cparam->delta * cparam->h * cparam->cs;
        gparam[i].h_cs_rho_m = cparam->h * cparam->cs * cparam->rho0 * cparam->m;
        gparam[i].dx2 = cparam->dx * cparam->dx;

        // grid part
        gparam[i].grid_factor = cparam->grid_factor;
        gparam[i].grid_size = cparam->grid_size;

        gparam[i].gmax_size = cparam->gmax_size;

        //different from other gpu
        gparam[i].gpu_id = i;
        gparam[i].gpu_num = cparam->gpu_num;
        gparam[i].start_l = cparam->ymin + ((float)i)*sub_len;
        gparam[i].end_l = gparam[i].start_l + cparam->r;
        gparam[i].end_r = cparam->ymin + ((float)(i+1))*sub_len;
        gparam[i].start_r = gparam[i].end_r - cparam->r;

        if(i == 0) gparam[i].end_l = gparam[i].start_l;
        else if(i == cparam->gpu_num-1) gparam[i].start_r = gparam[i].end_r;

        gparam[i].grid_xmin = cparam->xmin - cparam->r;
        gparam[i].grid_ymin = gparam[i].start_l - 2.0f*cparam->r;
        gparam[i].grid_zmin = cparam->zmin - cparam->r;
        
        gparam[i].grid_xmax = cparam->xmax + cparam->r;
        gparam[i].grid_ymax = gparam[i].end_r + 2.0f*cparam->r;
        gparam[i].grid_zmax = cparam->zmax + cparam->r;

        gparam[i].grid_xdim = (int)((gparam[i].grid_xmax - gparam[i].grid_xmin)/gparam[i].grid_size);
        gparam[i].grid_ydim = (int)((gparam[i].grid_ymax - gparam[i].grid_ymin)/gparam[i].grid_size);
        gparam[i].grid_zdim = (int)((gparam[i].grid_zmax - gparam[i].grid_zmin)/gparam[i].grid_size);
        gparam[i].grid_hash_min = 0;
        gparam[i].grid_hash_max = gparam[i].grid_xdim * gparam[i].grid_ydim * gparam[i].grid_zdim;

        gparam[i].rec_l = gparam[i].start_l - cparam->r;
        gparam[i].rec_r = gparam[i].end_r + cparam->r;
    }
    std::cout << "*******************************************************************************************" << std::endl;
    std::cout << "* Zsph is check gpu spliting param..." << std::endl;    
    for(int i=0;i<cparam->gpu_num;i++)
    {
        #define SW std::setw(10)
        std::cout << "* gpu" << i << " configure is:" << SW << gparam[i].grid_ymin ;
        std::cout << SW << gparam[i].rec_l;
        std::cout << SW << gparam[i].start_l;
        std::cout << SW << gparam[i].end_l;
        std::cout << SW << gparam[i].start_r;
        std::cout << SW << gparam[i].end_r;
        std::cout << SW << gparam[i].rec_r;
        std::cout << SW << gparam[i].grid_ymax << std::endl;
        #undef SW
    }
    std::cout << "* Spliting checing is done..." << std::endl;
}

