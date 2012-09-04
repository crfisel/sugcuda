/*
 * mate.h
 *
 *  Created on: Dec 4, 2011
 *      Author: C. Richard Fisel
 */

#ifndef MATE_H_
#define MATE_H_

#include <curand.h>
#include <curand_kernel.h>

//__noinline__
__device__ bool is_fertile(int iAgentID, int* piaBits, short* psaX);
//__noinline__
__device__ bool is_acceptable_mate(int iMateID, int* piaBits, short* psaX);
//__noinline__
__global__ void mate_masked(short* psaX, short* psaY, int* piaBits, int* pigBits, int* pigResidents,
		float* pfaSugar, float* pfaSpice, float* pfaInitialSugar, float* pfaInitialSpice,
		int* piaActiveQueue, int* piaDeferredQueue, const int ciActiveQueueSize, int* piPopulation,
		curandState* paStates, int* piDeferredQueueSize, int* piLockSuccesses);
/*
__global__ void mate_once(short* psaX, short* psaY, int* piaBits, int* pigBits, int* pigResidents,
		float* pfaSugar, float* pfaSpice, float* pfaInitialSugar, float* pfaInitialSpice,
		int* piaActiveQueue, int* piaDeferredQueue, const int ciActiveQueueSize, int* piPopulation,
		curandState* paStates, int* piDeferredQueueSize, int* piLockSuccesses);
*/

#endif /* MATE_H_ */
