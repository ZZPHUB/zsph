#include "equation.hpp"

void cal_acc_alpha(rigid_t &rigid)
{
    vector_t acc = {0.0f, 0.0f, 0.0f};   
    rigid.acc.x = rigid.F.x / rigid.mass;
    rigid.acc.y = rigid.F.y / rigid.mass;
    rigid.acc.z = rigid.F.z / rigid.mass;

    vector_t alpha = {0.0f, 0.0f, 0.0f};

    /*
    quaternion_t tmp = inv(rigid.q) * rigid.T * rigid.q;
    vector_t T_n = {tmp.b, tmp.c, tmp.d};
    tmp = inv(rigid.q) * rigid.omega * rigid.q;
    vector_t omega_n = {tmp.b, tmp.c, tmp.d};

    alpha = inv(rigid.I)*(T_n - cross(omega_n, rigid.I*omega_n));
    tmp = inv(rigid.q) * alpha * rigid.q;
    vector_t tmp_alpha = {tmp.b, tmp.c, tmp.d};
    */

    tensor_t I_inv = inv(rigid.I);
    tensor_t R = q2t(rigid.q);
    tensor_t R_T = trans(R);
    
    alpha = (R * I_inv * R_T) * (rigid.T - cross(rigid.omega, (R*rigid.I*R_T)*rigid.omega));
    //if(fabs(alpha.x) <= 1e-4f) alpha.x = 0.0f;
    //if(fabs(alpha.y) <= 1e-4f) alpha.y = 0.0f;
    //if(fabs(alpha.z) <= 1e-4f) alpha.z = 0.0f;
    rigid.alpha = alpha;

    //std::cout << "  alpha:   " << alpha.x << " " << alpha.y << " " << alpha.z << std::endl; 
    //std::cout << "tmp_alpha: " << tmp_alpha.x << " " << tmp_alpha.y << " " << tmp_alpha.z << std::endl;
}