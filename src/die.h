/*
 * die.h
 *
 *  Created on: Dec 4, 2011
 *      Author: C. Richard Fisel
 */

#ifndef DIE_H_
#define DIE_H_

__global__ void register_deaths(short* , short* , int* , short* , float* , float* , int* , int* , int* , const int , int* , int* , int* );

__global__ void register_deaths_fs(short* , short* , int* , short* , float* , float* , int* , int* , int* , const int);

int die(short* , short* , int* , short* , float* , float* , int* , int* , int* , const int , int* , int* , int* );

#endif /* DIE_H_ */
