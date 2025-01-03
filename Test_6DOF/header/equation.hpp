#ifndef __EQUATION_HPP__
#define __EQUATION_HPP__

#include "data_structure.hpp"
#include "std_header.hpp"
#include "lib.hpp"

extern vector_t gravity_center(const std::vector<vector_t> &ptc);
extern tensor_t inertia_tensor(const std::vector<vector_t> &ptc, const rigid_t &rigid);
extern void cal_acc_alpha(rigid_t &rigid);
extern void time_int(std::vector<vector_t> &ptc, rigid_t &rigid);
#endif