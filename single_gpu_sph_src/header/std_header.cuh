#ifndef __STD_HEADER_CUH__
#define __STD_HEADER_CUH__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>

#ifdef SPH_VTK
#include <vtkUnstructuredGridReader.h>
#include <vtkSmartPointer.h>
#include <vtkType.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPointSet.h>
#include <vtkDataSetReader.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
//#include <vtkIntArray.h>
#include <vtkFloatArray.h>
#endif

#endif