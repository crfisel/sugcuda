/*
 * move.h
 *
 *  Created on: Nov 24, 2011
 *      Author: C. Richard Fisel
 */

#ifndef MOVE_H_
#define MOVE_H_

__global__ void best_move_by_traversal(short* , short* , int* ,	float* , float* , 
	int* , int* , int* , const int , int* , int* , int* );

__global__ void best_move_by_traversal_fs(short* , short* , int* , float* , float* , 
	int* , int* , int* , const int );

#endif /* MOVE_H_ */

