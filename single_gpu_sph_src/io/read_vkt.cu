#include "io.cuh"

void read_vtk(cpu_data_t *data, cpu_param_t *param)
{
    std::string file_name = param->input_path + "/input.vtk";
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(file_name.c_str());
    reader->ReadAllScalarsOn();
    reader->ReadAllVectorsOn();
    reader->Update();
    vtkUnstructuredGrid *vtkdata;
    vtkdata = reader->GetOutput();
    // assert vtk file's ptc num equals to ptc_num setting
    assert(vtkdata->GetNumberOfPoints() == param->ptc_num);
    vtkPointData *pointdata = vtkdata->GetPointData();
    vtkDoubleArray *p_data = nullptr;vtkDoubleArray *vel_data = nullptr ;
    vtkDoubleArray *acc_data = nullptr;vtkDoubleArray *rho_data = nullptr;
    vtkIntArray *type_data = nullptr; vtkIntArray *table_data = nullptr;
    vtkDataArray *p_array = nullptr;
    vtkDataArray *vel_array =nullptr;
    vtkDataArray *acc_array = nullptr;
    vtkDataArray *rho_array = nullptr;
    vtkDataArray *type_array = nullptr;
    vtkDataArray *table_array = nullptr;

    p_array = pointdata->GetScalars("p");
    if(p_array != nullptr) p_data = vtkDoubleArray::SafeDownCast(p_array);
    //assert(p_data != nullptr);

    vel_array = pointdata->GetVectors("vel"); 
    if(vel_array != nullptr) vel_data = vtkDoubleArray::SafeDownCast(vel_array); 
    //assert(vel_data != nullptr);

    acc_array = pointdata->GetVectors("acc");
    if(acc_array != nullptr) acc_data = vtkDoubleArray::SafeDownCast(acc_array);
    //assert(acc_data != nullptr);

    rho_array = pointdata->GetScalars("rho");
    if(rho_array != nullptr) rho_data = vtkDoubleArray::SafeDownCast(rho_array);
    //assert(rho_data != nullptr);

    type_array = pointdata->GetScalars("type");
    if(type_array !=nullptr) type_data = vtkIntArray::SafeDownCast(type_array);
    //assert(type_data != nullptr);

    table_array = pointdata->GetScalars("table");
    if(table_array != nullptr) table_data = vtkIntArray::SafeDownCast(table_array);
    //assert(table_data != nullptr);

    int index = 0;
    for (vtkIdType i = 0; i < vtkdata->GetNumberOfPoints(); i++)
    {
        double p = 0.0;
        double rho = param->rho0;
        double vel[3] = {0.0};
        double pos[3] = {0.0};
        double acc[3] = {0.0};
        int type = 0;
        int table = 0;
   

        if(p_data != nullptr) p_data->GetTuple(i, &p);
        if(rho_data != nullptr) rho_data->GetTuple(i, &rho);
        if(vel_data != nullptr) vel_data->GetTuple(i, vel);
        if(acc_data != nullptr) acc_data->GetTuple(i, acc);
        type = type_data->GetValue(i); //must have type , so cannot be nullptr
        if(table_data != nullptr) table = table_data->GetValue(i);
        vtkdata->GetPoint(i, pos); //must have pos , so cannot be nullptr

        // get data of pos and rho
        data->pos_rho[index*4] = (float)pos[0];
        data->pos_rho[index*4 + 1] = (float)pos[1];
        data->pos_rho[index*4 + 2] = (float)pos[2];
        data->pos_rho[index*4 + 3] = (float)rho;
        // get data of vel and p
        data->vel_p[index*4] = (float)vel[0];
        data->vel_p[index*4 + 1] = (float)vel[1];
        data->vel_p[index*4 + 2] = (float)vel[2];
        data->vel_p[index*4 + 3] = (float)p;
        // get data of acc and drhodt
        data->acc_drhodt[index*4] = (float)acc[0];
        data->acc_drhodt[index*4 + 1] = (float)acc[1];
        data->acc_drhodt[index*4 + 2] = (float)acc[2];
        data->acc_drhodt[index*4 + 3] = 0.0f;
        // get data of type and table
        data->type[index] = type;
        data->table[index] = table;

        index++;
    }
}