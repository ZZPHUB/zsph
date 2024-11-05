#include "io.cuh"
using json = nlohmann::json;


void read_json(std::string file_name, cpu_json_t *json_data)
{

    //read json
    std::ifstream fjson(file_name.c_str());
    json jdata = json::parse(fjson);

    json_data->g = jdata["g"];
    //json_data->m = jdata["m"];
    json_data->rho0 = jdata["rho0"];
    json_data->dx = jdata["dx"];
    json_data->h_factor = jdata["h_factor"];
    json_data->r_factor = jdata["r_factor"];
    json_data->eta_factor = jdata["eta_factor"];
    json_data->cs_factor = jdata["cs_factor"];
    json_data->delta = jdata["delta"];
    json_data->alpha = jdata["alpha"];
    json_data->xmin = jdata["xmin"];
    json_data->xmax = jdata["xmax"];
    json_data->ymin = jdata["ymin"];
    json_data->ymax = jdata["ymax"];
    json_data->zmin = jdata["zmin"];
    json_data->zmax = jdata["zmax"];
    json_data->grid_layer_factor = jdata["grid_layer_factor"];
    json_data->grid_size_factor = jdata["grid_size_factor"];

    json_data->dt = jdata["dt"];
    json_data->start_step = jdata["start_step"];
    json_data->current_step = jdata["current_step"];//current_step is changed when write back
    json_data->end_step = jdata["end_step"];
    json_data->output_step = jdata["output_step"];

    json_data->ptc_num = jdata["ptc_num"];
    json_data->water_ptc_num = jdata["water_ptc_num"];
    json_data->air_ptc_num = jdata["air_ptc_num"];
    json_data->wall_ptc_num = jdata["wall_ptc_num"];
    json_data->rigid_ptc_num = jdata["rigid_ptc_num"];

    json_data->thread_num = jdata["thread_num"];

    //json_data->gpu_id = jdata["gpu_id"];
    json_data->gpu_num = jdata["gpu_num"];

    json_data->input_path = jdata["input_path"];
    json_data->output_path = jdata["output_path"];
    json_data->git_hash = jdata["git_hash"];
    
}

