#include "equation.hpp"

void time_int(std::vector<vector_t> &ptc, rigid_t &rigid)
{
    real dt = 0.001f;

    for(int i=0;i<ptc.size();i++)
    {
        vector_t tmp_pos = ptc[i] - rigid.c_pos;
        vector_t tmp_vel = rigid.vel + cross(rigid.omega, tmp_pos);
        ptc[i] = ptc[i] + tmp_vel * dt;
    }
    
    rigid.q = rigid.q + 0.5 * rigid.omega * rigid.q * dt;
    rigid.q = rigid.q * (1.0f/(real)sqrt(rigid.q.a*rigid.q.a + rigid.q.b*rigid.q.b + rigid.q.c*rigid.q.c + rigid.q.d*rigid.q.d));
    rigid.c_pos = rigid.c_pos + rigid.vel * dt;
    rigid.vel = rigid.vel + rigid.acc * dt;
    rigid.omega = rigid.omega + rigid.alpha * dt;

    rigid.acc = rigid.acc * 0.0f;
    rigid.alpha = rigid.alpha * 0.0f;
}