#include "io.hpp"

void write_vtk(std::vector<vector_t> &ptc, int step)
{
    std::string filename = "output/step_" + std::to_string(step) + ".vtk";
    std::ofstream ofile;
    ofile.open(filename);

    ofile << "# vtk DataFile Version 3.0" << std::endl;
    ofile << "zsph data" << std::endl;
    ofile << "ASCII" << std::endl;
    ofile << "DATASET UNSTRUCTURED_GRID" << std::endl;
    ofile << "POINTS " << ptc.size() << " float" << std::endl;
    for (int i = 0; i < ptc.size(); i++)
    {
        ofile << ptc[i].x << " " << ptc[i].y << " " << ptc[i].z << std::endl;
    }
}