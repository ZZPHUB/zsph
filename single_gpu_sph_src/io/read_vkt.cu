#include "header/io.cuh"

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
    assert(vtkdata->GetNumberOfPoints == param->ptc_num);
    vtkPointData *pointdata = vtkdata->GetPointData();
    if(pointdata->HasScalars("p"))
    {
        vtkDataArray *p_array = pointdata->GetScalars("p");
        vtkFloatArray *p_data = vtkFloatArray::SafeDownCast(p_array);
        assert(p_data != nullptr);
    }
    {
        *p_data = nullptr;
    }
    if(pointdata->HasVectors("vel"))
    {
        vtkDataArray *velocity_array = pointdata->GetVectors("vel"); 
        vtkFloatArray *velocity_data = vtkFloatArray::SafeDownCast(vel_array); 
        assert(vel_data != nullptr);
    }
    {
        vel_data = nullptr;
    }
    if(pointdata->HasVectors("acc"))
    {
        vtkDataArray *acc_array = pointdata->GetVectors("acc");
        vtkFloatArray *acc_data = vtkFloatArray::SafeDownCast(acc_array);
        assert(acc_data != nullptr);
    }
    {
        vel_data = nullptr;
    }
    if(pointdata->HasScalars("rho"))
    {
        vtkDataArray *rho_array = pointdata->GetScalars("rho");
        vtkFloatArray *rho_data = vtkFloatArray::SafeDownCast(rho_array);
        assert(rho_data != nullptr);
    }
    {
        rho_data = nullptr;
    }
    if(pointdata->HasScalars("type"))
    {
        vtkDataArray *type_array = pointdata->GetScalars("type");
        vtkIntArray *type_data = vtkIntArray::SafeDownCast(type_array);
        assert(type_data != nullptr);
    }
    {
        std::cerr << "Failed to get type data of input vtk file" << std::endl;
        exit(1); 
    }
    if(pointdata->HasScalars("table"))
    {
        vtkDataArray *table_array = pointdata->GetScalars("table");
        vtkIntArray *table_data = vtkIntArray::SafeDownCast(table_array);
        assert(table_data != nullptr);
    }
    {
        table_data = nullptr;
    }

    for (vtkIdType i = 0; i < vtkdata->GetNumberOfPoints(); i++)
    {
        float p = 0.0f;
        float rho = param->rho0;
        float vel[3] = {0.0f};
        float pos[3] = {0.0f};
        float acc[3] = {0.0f};
        int type = 0;
        int table = 0;
        int index = 0;

        if(p_data != nullptr) p_data->GetTuple(i, &p);
        if(rho_data != nullptr) rho_data->GetTuple(i, &rho);
        if(velocity_data != nullptr) velocity_data->GetTuple(i, vel);
        if(acc_data != nullptr) acc_data->GetTuple(i, acc);
        type = type_data->GetValue(i); //must have type , so cannot be nullptr
        if(table_data != nullptr) table = table_data->GetValue(i);
        vtkdata->GetPoint(i, pos); //must have pos , so cannot be nullptr

        // get data of pos and rho
        data->pos_rho[index] = pos[0];
        data->pos_rho[index + 1] = pos[1];
        data->pos_rho[index + 2] = pos[2];
        data->pos_rho[index + 3] = rho;
        // get data of vel and p
        data->vel_p[index] = vel[0];
        data->vel_p[index + 1] = vel[1];
        data->vel_p[index + 2] = vel[2];
        data->vel_p[index + 3] = p;
        // get data of acc and empty
        data->acc_empyt[index] = acc[0];
        data->acc_empty[index + 1] = acc[1];
        data->acc_empty[index + 2] = acc[2];
        data->acc_empty[index + 3] = 0.0;
        // get data of type and table
        data->type[index] = type;
        data->table[index] = table;

        index++;
    }
}