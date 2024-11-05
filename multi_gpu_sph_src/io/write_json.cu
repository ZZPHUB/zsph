#include "io.cuh"
using json = nlohmann::json;

void write_json(cpu_param_t *param, cpu_json_t *jdata)
{
    std::ifstream fin((jdata->input_path+"intput.json").c_str());
    json out_json = json::parse(fin);
    out_json["current_step"] = param->current_step;
    fin.close();

    std::ofstream fout((jdata->output_path+"output.json").c_str());
    fout << out_json;
    fout.close();
}