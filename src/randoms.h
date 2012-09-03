/*
 * randoms.h
 *
 *  Created on: Nov 14, 2011
 *      Author: C. Richard Fisel
 */

#ifndef RANDOMS_H_
#define RANDOMS_H_

__global__ void initialize_food(unsigned int* , float* , float );

__global__ void initialize_agentbits(unsigned int* , int* );

__global__ void fill_positions(unsigned int* , short* , short* );

__global__ void initialize_gridbits(unsigned int* , int* );

#endif //RANDOMS_H
