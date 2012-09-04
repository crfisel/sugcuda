/*
 * exercise_locks.h
 *
 *  Created on: Dec 17, 2011
 *      Author: C. Richard Fisel
 */

#ifndef EXERCISE_LOCKS_H_
#define EXERCISE_LOCKS_H_

enum operation {COUNT, MOVE, DIE, MATE};

int exercise_locks (operation , curandState* , short* , short* , int* ,
	float* , float* , float* , float* ,
	int* , int* , int* , int* , int* , int* , int* , int* , int* , int* , int* , int* );

#endif /* EXERCISE_LOCKS_H_ */
