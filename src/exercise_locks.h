/*
 * exercise_locks.h
 *
 *  Created on: Dec 17, 2011
 *      Author: C. Richard Fisel
 */

#ifndef EXERCISE_LOCKS_H_
#define EXERCISE_LOCKS_H_

enum operation {COUNT, MOVE, DIE, MATE};
#define MAX_OPCODE 4

__global__ void initialize_queue(int* piaQueue, int iQueueLength);

int exercise_locks(operation routine, short* psaX, short* psaY,int* piaAgentBits, int* pigGridBits, int* pigResidents,
		float* pfaSugar, float* pfaSpice, float* pfaInitialSugar, float* pfaInitialSpice,
		int* piaQueueA, int* piaQueueB, int* piPopulation, int* pihPopulation,
		int* piDeferredQueueSize, int* pihDeferredQueueSize, curandState* paStates,
		int* piLockSuccesses, int* pihLockSuccesses, int* piStaticAgents, int* pihStaticAgents);

#endif /* EXERCISE_LOCKS_H_ */
