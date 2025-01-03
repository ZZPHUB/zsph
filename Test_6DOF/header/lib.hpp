#ifndef __LIB_HPP__
#define __LIB_HPP__

#include "data_structure.hpp"
#include "std_header.hpp"

#include "lib.hpp"

/* vector operation */
extern vector_t operator+(const vector_t& a,const vector_t& b);

extern vector_t operator-(const vector_t& a,const vector_t& b);

extern vector_t operator*(const vector_t& a,const real& b);

extern vector_t operator*(const real& a,const vector_t& b);

extern real operator*(const vector_t& a,const vector_t& b);

extern vector_t cross(const vector_t& a,const vector_t& b);


/* quaternion operations */
extern quaternion_t operator+(const quaternion_t& a,const quaternion_t& b);

extern quaternion_t operator-(const quaternion_t& a,const quaternion_t& b);

extern quaternion_t operator*(const quaternion_t& a,const real& b);

extern quaternion_t operator*(const real& a,const quaternion_t& b);

extern quaternion_t operator*(const quaternion_t& a,const quaternion_t& b);

extern quaternion_t operator*(const quaternion_t& a,const vector_t& b);

extern quaternion_t operator*(const vector_t& a,const quaternion_t& b);

extern quaternion_t inv(const quaternion_t& a);


/* tensor operation */
extern tensor_t operator+(const tensor_t& a,const tensor_t& b);

extern tensor_t operator-(const tensor_t& a,const tensor_t& b);

extern tensor_t operator*(const tensor_t& a,const real& b);

extern tensor_t operator*(const real& a,const tensor_t& b);

extern tensor_t operator*(const tensor_t& a,const tensor_t& b);

extern vector_t operator*(const tensor_t& a,const vector_t& b);

extern real det(const tensor_t& a);

extern tensor_t inv(const tensor_t& a);

extern tensor_t q2t(const quaternion_t& q);

extern tensor_t trans(const tensor_t& a);


#endif