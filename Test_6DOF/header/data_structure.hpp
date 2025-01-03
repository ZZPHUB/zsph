#ifndef __DATA_STRUCTURE_HPP__
#define __DATA_STRUCTURE_HPP__

using real = float;

struct vector_t
{
    real x;
    real y;
    real z;
};

struct quaternion_t
{
    real a;
    real b;
    real c;
    real d;
};

struct tensor_t
{
    real x1;
    real x2;
    real x3;
    
    real y1;
    real y2;
    real y3;

    real z1;
    real z2;
    real z3;
};

struct rigid_t
{
    vector_t c_pos;
    vector_t F;
    vector_t T;
    vector_t vel;
    vector_t omega;
    vector_t acc;
    vector_t alpha;
    tensor_t I;
    quaternion_t q;
    real mass;
    int ptc_num;
    int rigid_id;
};


#endif