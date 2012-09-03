/*
 * randoms.h
 *
 *  Created on: Nov 14, 2011
 *      Author: C. Richard Fisel
 */

#ifndef RANDOMS_H_
#define RANDOMS_H_

__global__ void setup_kernel(curandState* );

__global__ void generate_floats(curandState* , float* , float );

__global__ void generate_ints(curandState* , int* , int );

__global__ void generate_shorts(curandState* , short* , short );

__global__ void generate_bits(curandState* , BitWiseType* );

#endif //RANDOMS_H
