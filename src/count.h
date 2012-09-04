/*
 * count.h
 *
 *  Created on: Nov 22, 2011
 *      Author: C. Richard Fisel
 */

#ifndef COUNT_H_
#define COUNT_H_

__global__ void count_occupancy(short* psaX, short* psaY, int* pigBits, int* pigResidents, int* piaActiveQueue,
		int* piaDeferredQueue, const int ciActiveQueueSize, int* piDeferredQueueSize, int* piLockSuccesses);

__global__ void count_occupancy_fs(short* psaX, short* psaY, int* pigBits, int* pigResidents,
		int* piaActiveQueue, const int ciActiveQueueSize);

#endif /* COUNT_H_ */
