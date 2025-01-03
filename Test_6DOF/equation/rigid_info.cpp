#include "equation.hpp"

vector_t gravity_center(const std::vector<vector_t> &ptc)
{
    vector_t center = {0.0f, 0.0f, 0.0f};
    for(int i = 0; i < ptc.size(); i++)
    {
        center.x += ptc[i].x;
        center.y += ptc[i].y;
        center.z += ptc[i].z;
    }
    center.x /= (real)ptc.size();
    center.y /= (real)ptc.size();
    center.z /= (real)ptc.size();
    return center;
}

tensor_t inertia_tensor(const std::vector<vector_t> &ptc, const rigid_t &rigid)
{
    vector_t center = rigid.c_pos;
    tensor_t I = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for(int i = 0; i < ptc.size(); i++)
    {
        I.x1 += (ptc[i].y - center.y) * (ptc[i].y - center.y) + (ptc[i].z - center.z) * (ptc[i].z - center.z);
        I.x2 -= (ptc[i].x - center.x) * (ptc[i].y - center.y);
        I.x3 -= (ptc[i].x - center.x) * (ptc[i].z - center.z);

        I.y1 = I.x2;
        I.y2 += (ptc[i].x - center.x) * (ptc[i].x - center.x) + (ptc[i].z - center.z) * (ptc[i].z - center.z);
        I.y3 -= (ptc[i].y - center.y) * (ptc[i].z - center.z);

        I.z1 = I.x3;
        I.z2 = I.y3;
        I.z3 += (ptc[i].x - center.x) * (ptc[i].x - center.x) + (ptc[i].y - center.y) * (ptc[i].y - center.y);
    }
    I = I * (rigid.mass / (real)ptc.size());
    return I;
}