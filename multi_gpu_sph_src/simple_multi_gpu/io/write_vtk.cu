#include "io.cuh"

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
    int X = *(int *)b;
    return X;
}

void write_vtk(std::string filename, cpu_output_t *data)
{
//BINARY FORMAT OUTPUT
#ifdef ZSPH_BINARY
    std::ofstream ofile;
    ofile.open(filename.c_str(), std::ios::out);

    ofile << "# vtk DataFile Version 3.0" << std::endl;
    ofile << "zsph data" << std::endl;
    ofile << "BINARY" << std::endl;
    ofile << "DATASET UNSTRUCTURED_GRID" << std::endl;
    ofile << "POINTS " << data->ptc_num << " float" << std::endl;

    // ptc position
    for (int i = 0; i < data->ptc_num; i++)
    {
        float x[3];
        x[0] = trans(data->pos[i * 3 + 0]);
        x[1] = trans(data->pos[i * 3 + 1]);
        x[2] = trans(data->pos[i * 3 + 2]);
        // std::cout << data->pos_rho[i*4]<< " " << data->pos_rho[i*4+1] <<" " << data->pos_rho[i*4+2] << std::endl;
        ofile.write((char *)x, 3 * sizeof(float));
    }

    // ptc data
    ofile << "POINT_DATA " << data->ptc_num << std::endl;

#ifdef ZSPH_OUPUT_RHO
    ofile << "SCALARS rho float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        float rho = trans(data->rhop[i * 2 ])
        ofile.write((char *)&rho, sizeof(float));
    }
#endif

#ifdef ZSPH_OUTPUT_P
    ofile << "SCALARS p float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        float p = trans(data->rhop[i * 2 + 1]);
        ofile.write((char *)&p, sizeof(float));
    }
#endif

#ifdef ZSPH_OUTPUT_TYPE
    ofile << "SCALARS type int 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        int type = trans(data->type[i]);
        ofile.write((char *)&type, sizeof(int));
    }
#endif 

#ifdef ZSPH_OUTPUT_HASH
    ofile << "SCALARS hash int 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for(int i=0;i < data->ptc_num; i++)
    {
        int tmphash = trans(data->hash[i]);
        ofile.write((char *)&tmphash,sizeof(int));
    }
#endif

#ifdef ZSPH_OUTPUT_GPUID
    ofile << "SCALARS gpuid int 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for(int i=0;i < data->ptc_num; i++)
    {
        int tmpgpuid = trans(data->gpu_id[i]);
        ofile.write((char *)&tmpgpuid,sizeof(int));
    }
#endif

#ifdef ZSPH_OUTPUT_ISPTC
    ofile << "SCALARS is_ptc int 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for(int i=0;i < data->ptc_num; i++)
    {
        int tmpisptc = trans(data->is_ptc[i]);
        ofile.write((char *)&tmpisptc,sizeof(int));
    }
#endif

#ifdef ZSPH_OUTPUT_WSUM
    ofile << "SCALARS wsum float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for(int i=0;i < data->ptc_num; i++)
    {
        float tmpwsum = trans(data->wsum[i]);
        ofile.write((char *)&tmpwsum,sizeof(float));
    }
#endif

#ifdef ZSPH_OUTPUT_VEL
    ofile << "VECTORS vel float" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        float vel[3];
        vel[0] = trans(data->vel[i *3 + 0]);
        vel[1] = trans(data->vel[i *3 + 1]);
        vel[2] = trans(data->vel[i *3 + 2]);
        ofile.write((char *)vel, 3 * sizeof(float));
    }
#endif

#ifdef ZSPH_OUTPUT_ACC
    ofile << "VECTORS acc float" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        float acc[3];
        acc[0] = trans(data->acc_drhodt[i * 4 + 0]);
        acc[1] = trans(data->acc_drhodt[i * 4 + 1]);
        acc[2] = trans(data->acc_drhodt[i * 4 + 2]);
        ofile.write((char *)acc, 3*sizeof(float));
    }
#endif
    ofile.close();
#endif

// ASCII FORMAT OUTPUT
#ifdef ZSPH_ASCII
    std::ofstream ofile;
    ofile.open(filename.c_str(), std::ios::out);

    ofile << "# vtk DataFile Version 3.0" << std::endl;
    ofile << "zsph data" << std::endl;
    ofile << "ASCII" << std::endl;
    ofile << "DATASET UNSTRUCTURED_GRID" << std::endl;
    ofile << "POINTS " << data->ptc_num << " float" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        ofile << data->pos[i * 3 + 0] << " " << data->pos[i *3 + 1] << " " << data->pos[i * 3 + 2] << std::endl;
    }
    ofile << "POINT_DATA " << data->ptc_num << std::endl;

#ifdef ZSPH_OUTPUT_RHO
    ofile << "SCALARS rho float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        ofile << data->rhop[i *2 ] << std::endl;
    }
#endif

#ifdef ZSPH_OUTPUT_P
    ofile << "SCALARS p float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        ofile << data->rhop[i * 2 + 1] << std::endl;
    }
#endif 

#ifdef ZSPH_OUTPUT_HASH
    ofile << "SCALARS hash int 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for(int i=0;i < data->ptc_num; i++)
    {
        ofile << data->hash[i] << std::endl;
    }
#endif

#ifdef ZSPH_OUTPUT_GPUID
    ofile << "SCALARS gpuid int 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for(int i=0;i < data->ptc_num; i++)
    {
        ofile << data->gpu_id[i] << std::endl;
    }
#endif

#ifdef ZSPH_OUTPUT_ISPTC
    ofile << "SCALARS is_ptc int 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for(int i=0;i < data->ptc_num; i++)
    {
        ofile << data->is_ptc[i] << std::endl;
    }
#endif

#ifdef ZSPH_OUTPUT_WSUM
    ofile << "SCALARS wsum float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for(int i=0;i < data->ptc_num; i++)
    {
        ofile << data->wsum[i] << std::endl;
    }
#endif

#ifdef ZSPH_OUTPUT_TYPE
    ofile << "SCALARS type int 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        ofile << data->type[i] << std::endl;
    }
#endif

#ifdef ZSPH_OUTPUT_VEL
    ofile << "VECTORS vel float" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        ofile << data->vel[i * 3 + 0] << " " << data->vel[i * 3 + 1] << " " << data->vel[i * 3 + 2] << std::endl;
    }
#endif

#ifdef ZSPH_OUTPUT_ACC
    ofile << "VECTORS acc float" << std::endl;
    for (int i = 0; i < data->ptc_num; i++)
    {
        ofile << data->acc_drhodt[i * 4 + 0] << " " << data->acc_drhodt[i * 4 + 1] << " " << data->acc_drhodt[i * 4 + 2] << std::endl;
    }
#endif
    ofile.close();
#endif
}
