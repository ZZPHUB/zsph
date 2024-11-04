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

    //gpu id
    int gpu_id;

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
    float grid_xmin,grid_ymin,grid_zmin;//min position of grid in xyz direction
    float grid_xmax,grid_ymax,grid_zmax;//max position of grid in xyz direction
    int grid_xdim,grid_ydim,grid_zdim;//num of grid in xyz direction
    int grid_num;//num of grid.grid_num = grid_xdim*grid_ydim*gird_zdim
    int grid_hash_min;//grid_hash_min = 0
    int grid_hash_max;//grid_hash_max = grid_num-1

    //ptc part
    int ptc_num;//num of ptc
    int water_ptc_num;//num of water ptc
    int air_ptc_num;//num of air ptc
    int wall_ptc_num;//num of wall ptc
    int rigid_ptc_num;//num of rigid ptc

    //thread num
    int thread_num;

    //gpu id
    int gpu_id;

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
    
    //grid part
    float grid_size;
    int grid_factor;
    float grid_xmin,grid_ymin,grid_zmin;
    //int grid_num; //do not use
    int grid_xdim,grid_ydim,grid_zdim;
    int grid_hash_min;
    int grid_hash_max;

    //ptc part
    int ptc_num;
}gpu_param_t;

typedef struct
{
    float *pos;
    float *vel;
    float *rhop;
    float *acc_drhodt;
    int *type;
    int *table;
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
}cpu_output_t;

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
}gpu_ptc_t;

typedef struct 
{
    float4 *acc_drhodt;
    float *dofv;
    int *index;
    int *hash;
    int *grid_start;
    int *grid_end;
}gpu_tmp_t;

typedef struct
{
    int current_step;
    cpu_output_t data;
    std::atomic<int> write_flag;
    std::atomic<int> end_flag;
    std::thread thread;
}cpu_thread_t;

extern __constant__ gpu_param_t par;
extern __constant__ gpu_tmp_t tmp_data;

#endif