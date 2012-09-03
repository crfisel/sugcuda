/*
 * randoms.h
 *
 *  Created on: Nov 14, 2011
 *      Author: C. Richard Fisel
 */

#ifndef RANDOMS_H_
#define RANDOMS_H_

__global__ void setup_kernel(curandStateXORWOW_t* );

__global__ void generate_floats(curandStateXORWOW_t* , float* , float );

__global__ void generate_ints(curandStateXORWOW_t* , int* );

__global__ void generate_shorts(curandStateXORWOW_t* , short* , short );

__global__ void initialize_gridbits(curandStateXORWOW_t* , int* );

#endif //RANDOMS_H
