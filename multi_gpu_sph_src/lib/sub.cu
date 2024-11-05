#include "lib.cuh"

void sub_domain_init(gpu_buff_t *gbuff,cpu_param_t *param)
{
    int direction_dim = param->grid_ydim;
    for(int i=0;i<param->sub_num;i++)
    {
        if(i > 0)
        {
            gbuff[i].left = 1;
            int tmp_dim = (int)(direction_dim/param->sub_num)*i;
            gbuff[i].hash_mid_l = tmp_dim * param->grid_xdim * param->grid_zdim;
            gbuff[i].hash_min_l = (tmp_dim - param->grid_factor)*param->grid_xdim*param->gird_zdim;
            gbuff[i].hash_max_l = (tmp_dim + param->grid_factor)*param->grid_xdim*param->grid_zdim;
        }
        else
        {
            gbuff[i].left = 0;
            gbuff[i].hash_min_l = param->grid_hash_min;
            gbuff[i].hash_min_l = param->grid_hash_min;
            gbuff[i].hash_max_l = param->grid_hash_min;
        }

        if(i<param->sub_num-1)
        {
            gbuff[i].right = 1;
            int tmp_dim = (int)(direction_dim/param->sub_num)*(i+1);
            gbuff[i].hash_mid_l = tmp_dim * param->grid_xdim * param->grid_zdim;
            gbuff[i].hash_min_l = (tmp_dim - param->grid_factor)*param->grid_xdim*param->gird_zdim;
            gbuff[i].hash_max_l = (tmp_dim + param->grid_factor)*param->grid_xdim*param->grid_zdim;
        }
        {
            gbuff[i].right = 0;
            gbuff[i].hash_mid_l = param->grid_hash_max;
            gbuff[i].hash_min_l = param->grid_hash_max;
            gbuff[i].hash_max_l = param->grid_hash_max;
        }
        gbuff[i].gpu_id = i;
        gbuff[i].out_buff = 1;
    }
}