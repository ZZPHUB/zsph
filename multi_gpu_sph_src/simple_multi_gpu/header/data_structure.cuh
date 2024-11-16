#ifndef __DATA_STRUCTURE_CUH__
#define __DATA_STRUCTURE_CUH__  

#include "std_header.cuh"

typedef struct 
{
    /* cpu json data */
    float g;
    float m;
    float rho0;
    float dx;
    float h_factor;
    float r_factor;
    float eta_factor;
    float cs_factor;
    float delta;
    float alpha;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float zmin;
    float zmax;
    float grid_layer_factor;
    float grid_size_factor;

    float dt;
    int start_step;
    int current_step;
    int end_step;
    int output_step;

    //ptc part
    int ptc_num;//num of ptc
    int water_ptc_num;//num of water ptc
    int air_ptc_num;//num of air ptc
    int wall_ptc_num;//num of wall ptc
    int rigid_ptc_num;//num of rigid ptc

    //thread num
    int thread_num;

    //total gpu num
    int gpu_num;

    //infomation about zsph
    std::string input_path;
    std::string output_path;
    std::string git_hash;
}cpu_json_t;

typedef struct 
{
    //simulation part
    float g; //gravity
    float m; //ptc mass
    float rho0; //initial density
    float dx; //initial ptc spacing
    float h; //initial smoothing length
    float r; //ptc support domain radix
    float eta; //eta = factor*h.initial value equal to particle spacing by a factor
    float cs; //initial sound speed
    float adh; //adh = W(0.0,h)
    float delta; //delta term factor
    float alpha; //artificial viscosity term factor

    //time part
    float dt;//time step
    int start_step;
    int current_step;
    int end_step;
    int output_step;

    //domain part
    float xmin,xmax,ymin,ymax,zmin,zmax;

    //grid part
    int grid_factor; //to search the neighbor gird
    float grid_size; //single grid size

    //ptc part
    int ptc_num;//num of ptc

    //thread num
    int thread_num;

    //gpu param setting
    int gmax_size;
    //total gpu_num
    int gpu_num;

    //infomation about zsph
    std::string input_path;
    std::string output_path;
    std::string git_hash;
}cpu_param_t;

typedef struct
{
    //simulation part
    float half_dt;
    float h;
    float g;
    float m;
    float rho0;
    float adh;
    float h2; //pow(h,2).h is the smoothing length
    float eta2; //pow(eta,2).eta is the value equal to particle spacing by a factor
    float cs2; //pow(cs,2).cs is the sound speed
    float alpha; //artificial term factor
    float delta_h_cs;//delta_h_cs = delta*h*cs
    float h_cs_rho_m;//h_cs_rho_m = h*cs*rho*m
    float dx2;//dx2 = dx*dx + dy*dy + dz*dz;
    
    //grid part
    float grid_size;
    int grid_factor;
    float grid_xmin,grid_ymin,grid_zmin;
    float grid_xmax,grid_ymax,grid_zmax;
    //int grid_num; //do not use
    int grid_xdim,grid_ydim,grid_zdim;
    int grid_hash_min;
    int grid_hash_max;

    //gpu id
    int gpu_id;
    int gpu_num;
    int gmax_size;
    /*
    int hash_start_l;
    int hash_end_l;
    int hash_start_r;
    int hash_end_r;
    */
    float start_l;
    float end_l;
    float start_r;
    float end_r;
    float rec_l;
    float rec_r;

    float left_val;
    float right_val;
}gpu_param_t;

typedef struct
{
    float *pos;
    float *vel;
    float *rhop;
    float *acc_drhodt;
    int *type;
    int *table;
    int *is_ptc;
    int *gpu_id;
}cpu_input_t;

typedef struct
{
    float *pos;
    float *vel;
    float *rhop;
    float *acc_drhodt;
    int *type;
    int *table;
    int *hash;
    float *wsum; 
    int *is_ptc;
    int *gpu_id;
    int ptc_num;   
}cpu_output_t;

typedef struct 
{   
    //the data array's ptr,and the data where transfered to the nearest gpus
    float3 *pos_send;
    float3 *vel_send;
    float2 *rhop_send;
    float3 *tmp_pos_send;
    float3 *tmp_vel_send;
    float2 *tmp_rhop_send;
    int *type_send;
    int *gpu_id_send;
    int *table_send;
    int *is_ptc_send;

    //the data array's ptc,and the data ptr is the ptr that where receive data frome left and right nearest gpus
    float3 *pos_rec_l;
    float3 *vel_rec_l;
    float2 *rhop_rec_l;
    float3 *tmp_pos_rec_l;
    float3 *tmp_vel_rec_l;
    float2 *tmp_rhop_rec_l;
    int *type_rec_l;
    int *gpu_id_rec_l;
    int *table_rec_l;
    int *is_ptc_rec_l;

    float3 *pos_rec_r;
    float3 *vel_rec_r;
    float2 *rhop_rec_r;
    float3 *tmp_pos_rec_r;
    float3 *tmp_vel_rec_r;
    float2 *tmp_rhop_rec_r;
    int *type_rec_r;
    int *gpu_id_rec_r;
    int *table_rec_r;
    int *is_ptc_rec_r;

    int total_ptc_num;
    int send_ptc_num;
    int outer_ptc_num;
//    int buffr_ptc_num;
//    int inner_ptc_num;
}gpu_socket_t;


typedef struct
{
    float3 *pos;
    float3 *vel;
    float2 *rhop;
    float3 *tmp_pos; //to save pos for time integration
    float3 *tmp_vel; //to save vel for time integration
    float2 *tmp_rhop; //to save rho and p for time intergration
    int *type;
    int *table;
    int *is_ptc;
    int *gpu_id;
}gpu_ptc_t;

typedef struct 
{
    float4 *acc_drhodt;
//#ifdef ZSPH_DELTA
    //float4 *dofv_grandrho;
//#elif ZSPH_SIMPLE
    float *dofv;
//#endif
    int *index;
    int *hash;
    //int *is_ptc;
    //int *gpu_id;
    float *wsum;
    int *grid_start;
    int *grid_end;
}gpu_tmp_t;

typedef struct
{
    int current_step;
    cpu_output_t *data;
    std::atomic<int> write_flag;
    std::atomic<int> end_flag;
    std::thread thread;
}cpu_thread_t;

extern __constant__ gpu_param_t par;
extern __constant__ gpu_tmp_t gtmp_data;

#endif