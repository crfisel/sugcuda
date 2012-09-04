/*
 * die.h
 *
 *  Created on: Dec 4, 2011
 *      Author: C. Richard Fisel
 */

#ifndef DIE_H_
#define DIE_H_

__global__ void register_deaths(short* psaX, short* psaY, int* piaBits, int* pigBits, int* pigResidents, float* pfaSugar, float* pfaSpice,
		int* piaActiveQueue, int* piaDeferredQueue, const int ciActiveQueueSize, int* piDeferredQueueSize, int* piLockSuccesses);

__global__ void register_deaths_fs(short* psaX, short* psaY, int* piaBits,
		int* pigBits, int* pigResidents, float* pfaSugar, float* pfaSpice,
		int* piaActiveQueue, const int ciActiveQueueSize);

#endif /* DIE_H_ */
