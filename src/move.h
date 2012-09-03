/*
 * move.h
 *
 *  Created on: Nov 24, 2011
 *      Author: sammy
 */

#ifndef MOVE_H_
#define MOVE_H_

__global__ void best_move_by_traversal(short* , short* , int* ,	float* , float* , 
	short* , short* , short* , int* , int* , int* , const int , int* , int* , int* );

__global__ void best_move_by_traversal_fs(short* , short* , int* , float* , float* , 
	short* , short* , short* , int* , int* , const int );

int move (short* , short* , int* , float* , float* ,short* , short* , short* , 
	int* , int* , int* , const int , int* , int* , int* );

#endif /* MOVE_H_ */

