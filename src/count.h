/*
 * count.h
 *
 *  Created on: Nov 22, 2011
 *      Author: C. Richard Fisel
 */

#ifndef COUNT_H_
#define COUNT_H_

__global__ void count_occupancy(short* , short* , int* , int* , int* , const int , int* , int* , int* );

__global__ void count_occupancy_fs(short* , short* , int* , int* , int* , const int );

#endif /* COUNT_H_ */
