#include "lib.hpp"

/* vector operation */
vector_t operator+(const vector_t& a,const vector_t& b)
{
    vector_t result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
}

vector_t operator-(const vector_t& a,const vector_t& b)
{
    vector_t result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

vector_t operator*(const vector_t& a,const real& b)
{
    vector_t result;
    result.x = a.x * b;
    result.y = a.y * b;
    result.z = a.z * b;
    return result;
}

vector_t operator*(const real& a,const vector_t& b)
{
    return b * a;
}

vector_t cross(const vector_t& a,const vector_t& b)
{
    vector_t result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

real operator*(const vector_t& a,const vector_t& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* quaternion operations */
quaternion_t operator+(const quaternion_t& a,const quaternion_t& b)
{
    quaternion_t result;
    result.a = a.a + b.a;
    result.b = a.b + b.b;
    result.c = a.c + b.c;
    result.d = a.d + b.d;
    return result;
}

quaternion_t operator-(const quaternion_t& a,const quaternion_t& b)
{
    quaternion_t result;
    result.a = a.a - b.a;
    result.b = a.b - b.b;
    result.c = a.c - b.c;
    result.d = a.d - b.d;
    return result;
}

quaternion_t operator*(const quaternion_t& a,const real& b)
{
    quaternion_t result;
    result.a = a.a * b;
    result.b = a.b * b;
    result.c = a.c * b;
    result.d = a.d * b;
    return result;
}

quaternion_t operator*(const real& a,const quaternion_t& b)
{
    return b * a;
}

quaternion_t operator*(const quaternion_t& a,const quaternion_t& b)
{
    quaternion_t result;
    result.a = a.a * b.a - a.b * b.b - a.c * b.c - a.d * b.d;
    result.b = a.b * b.a + a.a * b.b - a.d * b.c + a.c * b.d;
    result.c = a.c * b.a + a.d * b.b + a.a * b.c - a.b * b.d;
    result.d = a.d * b.a - a.c * b.b + a.b * b.c + a.a * b.d;
    return result;
}

quaternion_t operator*(const quaternion_t& a,const vector_t& b)
{
    quaternion_t tmp_b;
    tmp_b.a = 0.0f;
    tmp_b.b = b.x;
    tmp_b.c = b.y;
    tmp_b.d = b.z;
    quaternion_t result = a * tmp_b;
    return result;
}

quaternion_t operator*(const vector_t& a,const quaternion_t& b)
{
    quaternion_t tmp_a;
    tmp_a.a = 0.0f;
    tmp_a.b = a.x;
    tmp_a.c = a.y;
    tmp_a.d = a.z;
    quaternion_t result = tmp_a * b;
    return result;
}

quaternion_t inv(const quaternion_t& a)
{
    quaternion_t result;
    result.a = a.a;
    result.b = -a.b;
    result.c = -a.c;
    result.d = -a.d;
    return result;
}

tensor_t q2t(const quaternion_t& q)
{
    tensor_t result;
    result.x1 = 1.0f - 2.0f * q.c * q.c - 2.0f * q.d * q.d;
    result.x2 = 2.0f * q.b * q.c - 2.0f * q.a * q.d;
    result.x3 = 2.0f * q.a * q.c + 2.0f * q.b * q.d;

    result.y1 = 2.0f * q.b * q.c + 2.0f * q.a * q.d;
    result.y2 = 1.0f - 2.0f * q.b * q.b - 2.0f * q.d * q.d;
    result.y3 = 2.0f * q.c * q.d - 2.0f * q.a * q.b;

    result.z1 = 2.0f * q.b * q.d - 2.0f * q.a * q.c;
    result.z2 = 2.0f * q.a * q.b + 2.0f * q.c * q.d;
    result.z3 = 1.0f - 2.0f * q.b * q.b - 2.0f * q.c * q.c;

    return result;
}

/* tensor operation */
tensor_t operator+(const tensor_t& a,const tensor_t& b)
{
    tensor_t result;
    result.x1 = a.x1 + b.x1;
    result.x2 = a.x2 + b.x2;
    result.x3 = a.x3 + b.x3;
    result.y1 = a.y1 + b.y1;
    result.y2 = a.y2 + b.y2;
    result.y3 = a.y3 + b.y3;
    result.z1 = a.z1 + b.z1;
    result.z2 = a.z2 + b.z2;
    result.z3 = a.z3 + b.z3;
    return result;
}

tensor_t operator-(const tensor_t& a,const tensor_t& b)
{
    tensor_t result;
    result.x1 = a.x1 - b.x1;
    result.x2 = a.x2 - b.x2;
    result.x3 = a.x3 - b.x3;
    result.y1 = a.y1 - b.y1;
    result.y2 = a.y2 - b.y2;
    result.y3 = a.y3 - b.y3;
    result.z1 = a.z1 - b.z1;
    result.z2 = a.z2 - b.z2;
    result.z3 = a.z3 - b.z3;
    return result;
}

tensor_t operator*(const tensor_t& a,const real& b)
{
    tensor_t result;
    result.x1 = a.x1 * b;
    result.x2 = a.x2 * b;
    result.x3 = a.x3 * b;
    result.y1 = a.y1 * b;
    result.y2 = a.y2 * b;
    result.y3 = a.y3 * b;
    result.z1 = a.z1 * b;
    result.z2 = a.z2 * b;
    result.z3 = a.z3 * b;
    return result;
}

tensor_t operator*(const real& a,const tensor_t& b)
{
    return b * a;
}

tensor_t operator*(const tensor_t& a,const tensor_t& b)
{
    tensor_t result;

    result.x1 = a.x1 * b.x1 + a.x2 * b.y1 + a.x3 * b.z1;
    result.x2 = a.x1 * b.x2 + a.x2 * b.y2 + a.x3 * b.z2;
    result.x3 = a.x1 * b.x3 + a.x2 * b.y3 + a.x3 * b.z3;

    result.y1 = a.y1 * b.x1 + a.y2 * b.y1 + a.y3 * b.z1;
    result.y2 = a.y1 * b.x2 + a.y2 * b.y2 + a.y3 * b.z2;
    result.y3 = a.y1 * b.x3 + a.y2 * b.y3 + a.y3 * b.z3;

    result.z1 = a.z1 * b.x1 + a.z2 * b.y1 + a.z3 * b.z1;
    result.z2 = a.z1 * b.x2 + a.z2 * b.y2 + a.z3 * b.z2;
    result.z3 = a.z1 * b.x3 + a.z2 * b.y3 + a.z3 * b.z3;

    return result;
}

vector_t operator*(const tensor_t& a,const vector_t& b)
{
    vector_t result;
    result.x = a.x1 * b.x + a.x2 * b.y + a.x3 * b.z;
    result.y = a.y1 * b.x + a.y2 * b.y + a.y3 * b.z;
    result.z = a.z1 * b.x + a.z2 * b.y + a.z3 * b.z;
    return result;
}

real det(const tensor_t& a)
{
    real result = a.x1 * a.y2 * a.z3  + a.x2 * a.y3 * a.z1 + a.x3 * a.y1 * a.z2 \
                - a.x3 * a.y2 * a.z1  - a.x1 * a.y3 * a.z2 - a.x2 * a.y1 * a.z3;
    return result;
}

tensor_t trans(const tensor_t& a)
{
    tensor_t result = a;
    result.x2 = a.y1;
    result.x3 = a.z1;
    result.y3 = a.z2;

    result.y1 = a.x2;
    result.z1 = a.x3;
    result.z2 = a.y3;

    return result;
}
tensor_t inv(const tensor_t& a)
{
    real det_a = det(a);
    tensor_t result;
    if(fabs(det_a) >= 1e-8)
    {
        result.x1 = a.y2 * a.z3 - a.y3 * a.z2;
        result.x2 = a.y3 * a.z1 - a.y1 * a.z3;
        result.x3 = a.y1 * a.z2 - a.y2 * a.z1;

        result.y1 = a.x3 * a.z2 - a.x2 * a.z3;
        result.y2 = a.x1 * a.z3 - a.x3 * a.z1;
        result.y3 = a.x2 * a.z1 - a.x1 * a.z2;

        result.z1 = a.x2 * a.y3 - a.x3 * a.y2;
        result.z2 = a.x3 * a.y1 - a.x1 * a.y3;
        result.z3 = a.x1 * a.y2 - a.x2 * a.y1;   

        result = result * (1.0f/det_a);
    }
    else
    {
        result = result * 0.0f;
    }
    //std::cout << "det = " << det_a << std::endl;
    return result;
    
}
