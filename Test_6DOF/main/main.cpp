#include "main.hpp"

int main()
{

    rigid_t rigid;
    rigid.c_pos = {0.0f, 0.0f, 0.0f};
    rigid.vel = {0.0f, 0.0f, 0.0f};
    rigid.acc = {0.0f, 0.0f, 0.0f};
    rigid.omega = {0.0f, 0.0f, 0.0f};
    rigid.alpha = {0.0f, 0.0f, 0.0f};
    rigid.q = {1.0f, 0.0f, 0.0f, 0.0f};
    rigid.mass = 10.0f;

    rigid.F = {1000.0f, 1000.0f, 0.0f};
    rigid.T = {0.0f, 1000.0f, 0.0f};

    std::vector<vector_t> ptc;
    for(int i = 0; i < 21; i++)
    {
        for(int j=0;j < 21; j++)
        {
            for(int k=0;k < 21;k++)
            {
                real tmp_x = (real)i * 0.1f;
                real tmp_y = (real)j * 0.1f; 
                real tmp_z = (real)k * 0.1f;
                vector_t tmp_ptc = {tmp_x, tmp_y, tmp_z};
                ptc.push_back(tmp_ptc);
            }
        }
    }

    rigid.c_pos = gravity_center(ptc);
    rigid.I = inertia_tensor(ptc, rigid);

    std::cout << "gravity center: " << rigid.c_pos.x << " " << rigid.c_pos.y << " " << rigid.c_pos.z << std::endl;
    std::cout << "inertia tensor: " << rigid.I.x1 << " " << rigid.I.x2 << " " << rigid.I.x3 << std::endl;
    std::cout << "                " << rigid.I.y1 << " " << rigid.I.y2 << " " << rigid.I.y3 << std::endl;
    std::cout << "                " << rigid.I.z1 << " " << rigid.I.z2 << " " << rigid.I.z3 << std::endl;
    
    /*
    cal_acc_alpha(rigid);
    real tmp_alpha = rigid.T.y / rigid.I.y2;
    std::cout << "tmp_alpha:   " << tmp_alpha << std::endl;
    std::cout << "rigid.alpha: " << rigid.alpha.y << std::endl;
    */
   
 
    for(int i=0;i<100000;i++)
    {
        std::cout << "i = " << i << std::endl;
        rigid.T = rigid.T * (0.5f - sin((real)i*0.01f*M_PI));
        rigid.F.x = rigid.F.x * (0.5f - sin((real)i*0.01f*M_PI));
        rigid.F.y = rigid.F.y * (0.5f - cos((real)i*0.01f*M_PI));
        if(i%200 == 0)
        {
            write_vtk(ptc, i);
        }
        cal_acc_alpha(rigid);

        time_int(ptc, rigid);
        real tmp = rigid.q.a * rigid.q.a + rigid.q.b * rigid.q.b + rigid.q.c * rigid.q.c + rigid.q.d * rigid.q.d;
        
        std::cout << "q = " <<tmp << std::endl;
        std::cout << "q: " << rigid.q.a << " " << rigid.q.b << " " << rigid.q.c << " " << rigid.q.d << std::endl;
        std::cout << "acc: " << rigid.acc.x << " " << rigid.acc.y << " " << rigid.acc.z << std::endl;
        std::cout << "alpha: " << rigid.alpha.x << " " << rigid.alpha.y << " " << rigid.alpha.z << std::endl;
        std::cout << "vel: " << rigid.vel.x << " " << rigid.vel.y << " " << rigid.vel.z << std::endl;
        std::cout << "omega: " << rigid.omega.x << " " << rigid.omega.y << " " << rigid.omega.z << std::endl;
        std::cout << "c_pos: " << rigid.c_pos.x << " " << rigid.c_pos.y << " " << rigid.c_pos.z << std::endl;


        /*
        auto tmp1 = q2t(rigid.q);
        auto tmp2 = q2t(inv(rigid.q));
        auto tmp3 = tmp1 * tmp2;
        std::cout << "tmp3: " << tmp3.x1 << " " << tmp3.x2 << " " << tmp3.x3 << std::endl;
        std::cout << "      " << tmp3.y1 << " " << tmp3.y2 << " " << tmp3.y3 << std::endl;
        std::cout << "      " << tmp3.z1 << " " << tmp3.z2 << " " << tmp3.z3 << std::endl;
        */
    }

    
    return 0;
}