/*
 * move.h
 *
 *  Created on: Nov 24, 2011
 *      Author: C. Richard Fisel
 */

#ifndef MOVE_H_
#define MOVE_H_

__global__ void best_move_by_traversal(short* psaX, short* psaY, int* piaBits, int* pigBits, int* pigResidents, float* pfaSugar, float* pfaSpice,
		int* piaActiveQueue, int* piaDeferredQueue, const int ciActiveQueueSize, int* piDeferredQueueSize, int* piLockSuccesses, int* piStaticAgents);

__global__ void best_move_by_traversal_fs(short* psaX, short* psaY, int* piaBits, int* pigBits, int* pigResidents,
		float* pfaSugar, float* pfaSpice, int* piaActiveQueue, const int ciActiveQueueSize, int* piStaticAgents);

#endif /* MOVE_H_ */

