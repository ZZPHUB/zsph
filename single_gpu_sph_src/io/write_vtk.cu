#include "header/io.cuh"

float trans(float x)
{
    unsigned int unsigned_x = *(unsigned int *)(&x);
    unsigned char b[4];
    b[0] = (char)((unsigned_x & 0xff000000) >> 24);
    b[1] = (char)((unsigned_x & 0x00ff0000) >> 16);
    b[2] = (char)((unsigned_x & 0x0000ff00) >> 8);
    b[3] = (char)((unsigned_x & 0x000000ff) >> 0);
    float X = *(float *)b;
    return X;
}

int trans(int x)
{
    unsigned int unsigned_x = *(unsigned int *)(&x);
    unsigned char b[4];
    b[0] = (char)((unsigned_x & 0xff000000) >> 24);
    b[1] = (char)((unsigned_x & 0x00ff0000) >> 16);
    b[2] = (char)((unsigned_x & 0x0000ff00) >> 8);
    b[3] = (char)((unsigned_x & 0x000000ff) >> 0);
    int X = *(float *)b;
    return X;
}

void write_vtk(std::string filename,cpu_data_t *data, cpu_param_t *param)
{
    std::ofstream ofile;
    ofile.open(file_name.c_str(), std::ios::out);

    ofile << "# vtk DataFile Version 3.0" << std::endl;
    ofile << "zsph data" << std::endl;
    ofile << "BINARY" << std::endl;
    ofile << "DATASET UNSTRUCTURED_GRID" << std::endl;
    ofile << "POINTS " << param->ptc_num << " float" << std::endl;

    // ptc position
    for (int i = 0; i < param->ptc_num; i++)
    {
        float x[3];
        x[0] = trans(data->pos_rho[i * 4 + 0]);
        x[1] = trans(data->pos_rho[i * 4 + 1]);
        x[2] = trans(data->pos_rho[i * 4 + 2]);
        ofile.write((char *)x, 3*sizeof(float));
    }

    // ptc data
    ofile << "POINT_DATA " << param->ptc_num << std::endl;

#ifdef ZSPH_DEBUG 
    ofile << "SCALARS rho float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < param->ptc_num; i++)
    {
        float rho = trans(data->pos_rho[i * 4 + 3])
        ofile.write((char *)&rho, sizeof(float));
    }
#endif

    ofile << "SCALARS p float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < param->ptc_num; i++)
    {
        float p = trans(data->vel_p[i * 4 + 3]);
        ofile.write((char *)&p, sizeof(float));
    }

    ofile << "SCALARS type int 1" << std::endl;
    for (int i = 0; i < param->ptc_num; i++)
    {
        int type = trans(data->type[i]);
        ofile.write((char *)&type, sizeof(int));
    }

    ofile << "VECTORS vel float" << std::endl;
    for (int i = 0; i < param->ptc_num; i++)
    {
        float vel[3];
        vel[0] = trans(data->vel_p[i * 4 + 0]);
        vel[1] = trans(data->vel_p[i * 4 + 1]);
        vel[2] = trans(data->vel_p[i * 4 + 2]);
        ofile.write((char *)vel, 3*sizeof(float));
    }
    
#ifdef ZSPH_DEBUG
    ofile << "VECTORS acc float" << std::endl;
    for (int i = 0; i < param->ptc_num; i++)
    {
        float acc[3];
        acc[0] = trans(data->acc_empty[i * 4 + 0]);
        acc[1] = trans(data->acc_empty[i * 4 + 1]);
        acc[2] = trans(data->acc_empty[i * 4 + 2]);
        ofile.write((char *)acc, sizeof(float));
    }
#endif
    ofile.close();
}

