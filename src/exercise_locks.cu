/*
 * exercise_locks.cu
 *
 *  Created on: Dec 17, 2011
 *      Author: C. Richard Fisel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "exercise_locks.h"
#include "move.h"
#include "count.h"
#include "mate.h"
#include "die.h"


int exercise_locks(operation routine, curandState* paStates, short* psaX, short* psaY, int* piaAgentBits,
	float* pfaSugar, float* pfaSpice, float* pfaInitialSugar, float* pfaInitialSpice,
	int* pigGridBits, int* pigResidents, int* piaQueueA, int* piPopulation, int* pihPopulation, int* piaQueueB,
	int* piDeferredQueueSize, int* piLockSuccesses, int* pihDeferredQueueSize, int* pihLockSuccesses,
	int* piStaticAgents, int* pihStaticAgents)
{
	int status = EXIT_SUCCESS;
	
	// sync the host and device readings of population
	CUDA_CALL(cudaMemcpy(pihPopulation,piPopulation,sizeof(int),cudaMemcpyDeviceToHost));

	// fill the agent queue with increasing (later random) id's
	int* piahTemp = (int*) malloc(pihPopulation[0]*sizeof(int));
	for (int i = 0; i < pihPopulation[0]; i++) {
		//		piahTemp[i] = rand() % pihPopulation[0];
		piahTemp[i] = i;
	}
	CUDA_CALL(cudaMemcpy(piaQueueA,piahTemp,pihPopulation[0]*sizeof(int),cudaMemcpyHostToDevice));

	// blank the deferred queue with all bits=-1
	CUDA_CALL(cudaMemset(piaQueueB,0xFF,pihPopulation[0]*sizeof(int)));

	// zero the deferred queue size
	CUDA_CALL(cudaMemset(piDeferredQueueSize,0,sizeof(int)));

	// zero the successful locks counter
	CUDA_CALL(cudaMemset(piLockSuccesses,0,sizeof(int)));

	// zero the static agents counter
	CUDA_CALL(cudaMemset(piStaticAgents,0,sizeof(int)));

	// call first iteration on parallel version with locking
	int hiNumBlocks = (pihPopulation[0]+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
	switch (routine) {
		case COUNT:
			count_occupancy<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pigGridBits,pigResidents,
				piaQueueA,pihPopulation[0],piaQueueB,piDeferredQueueSize,piLockSuccesses);
			break;
		case MOVE:
			best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
				pigGridBits,pigResidents,piaQueueA,pihPopulation[0],piaQueueB,piDeferredQueueSize,piLockSuccesses,piStaticAgents);
			CUDA_CALL(cudaMemcpy(pihStaticAgents,piStaticAgents,sizeof(int),cudaMemcpyDeviceToHost));
//			printf ("static agents:%d \n",pihStaticAgents[0]);
			break;
		case MATE:
			mate_masked<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(paStates,psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,
				pigGridBits,pigResidents,piaQueueA,pihPopulation[0],piPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses);
			break;
		case DIE:
			register_deaths<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
				pigGridBits,pigResidents,piaQueueA,pihPopulation[0],piaQueueB,piDeferredQueueSize,piLockSuccesses);
			break;
		default:
			break;
	}
	CUDA_CALL(cudaDeviceSynchronize());

	// check if any agents had to be deferred
	CUDA_CALL(cudaMemcpy(pihDeferredQueueSize,piDeferredQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
//	printf ("primary deferrals:%d \n",pihDeferredQueueSize[0]);
	CUDA_CALL(cudaMemcpy(pihLockSuccesses,piLockSuccesses,sizeof(int),cudaMemcpyDeviceToHost));
//	printf ("successful locks:%d \n",pihLockSuccesses[0]);


	// handle the deferred queue until it is empty
	int ihActiveQueueSize = pihPopulation[0];
	bool hQueue = true;
	while (pihDeferredQueueSize[0] > 10 && pihDeferredQueueSize[0] < ihActiveQueueSize) {
		ihActiveQueueSize = pihDeferredQueueSize[0];
		hiNumBlocks = (ihActiveQueueSize+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
		CUDA_CALL(cudaMemset(piDeferredQueueSize,0,sizeof(int)));
		if (hQueue) {
			// switch the two queues and handle deferred agents
			CUDA_CALL(cudaMemset(piaQueueA,0xFF,pihPopulation[0]*sizeof(int)));
			switch (routine) {
				case COUNT:
					count_occupancy<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pigGridBits,pigResidents,
						piaQueueB,ihActiveQueueSize,piaQueueA,piDeferredQueueSize,piLockSuccesses);
				break;
				case MOVE:
					best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
						pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize,piaQueueA,piDeferredQueueSize,piLockSuccesses,piStaticAgents);
					CUDA_CALL(cudaMemcpy(pihStaticAgents,piStaticAgents,sizeof(int),cudaMemcpyDeviceToHost));
//					printf ("static agents (cumulative):%d \n",pihStaticAgents[0]);
					break;
				case MATE:
					mate_masked<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(paStates,psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,
						pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize,piPopulation,piaQueueA,piDeferredQueueSize,piLockSuccesses);
					break;
				case DIE:
					register_deaths<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
						pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize,piaQueueA,piDeferredQueueSize,piLockSuccesses);
					break;
				default:
					break;
			}
		} else {
			// switch queues the other way
			CUDA_CALL(cudaMemset(piaQueueB,0xFF,pihPopulation[0]*sizeof(int)));
			switch (routine) {
				case COUNT:
					count_occupancy<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pigGridBits,pigResidents,
						piaQueueA,ihActiveQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
				break;
				case MOVE:
					best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
						pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses,piStaticAgents);
					CUDA_CALL(cudaMemcpy(pihStaticAgents,piStaticAgents,sizeof(int),cudaMemcpyDeviceToHost));
//					printf ("static agents (cumulative):%d \n",pihStaticAgents[0]);
					break;
				case MATE:
					mate_masked<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(paStates,psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,
						pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize,piPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses);
					break;
				case DIE:
					register_deaths<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
						pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
					break;
				default:
					break;
			}
		}
		CUDA_CALL(cudaDeviceSynchronize());
		hQueue = !hQueue;
		CUDA_CALL(cudaMemcpy(pihDeferredQueueSize,piDeferredQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
//		printf ("secondary deferrals:%d \n",pihDeferredQueueSize[0]);
		CUDA_CALL(cudaMemcpy(pihLockSuccesses,piLockSuccesses,sizeof(int),cudaMemcpyDeviceToHost));
//		printf ("successful locks (cumulative):%d \n",pihLockSuccesses[0]);
	} 

	// for persistent lock failures, use the failsafe version (once)
	if (pihDeferredQueueSize[0] <= 10 || pihDeferredQueueSize[0] >= ihActiveQueueSize) {
		ihActiveQueueSize = pihDeferredQueueSize[0];
		if (hQueue) {
			switch (routine) {
				case COUNT:
					count_occupancy_fs<<<1,1>>>(psaX,psaY,pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize);
				break;
				case MOVE:
					best_move_by_traversal_fs<<<1,1>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pigGridBits,
						pigResidents,piaQueueB,ihActiveQueueSize,piStaticAgents);
					CUDA_CALL(cudaMemcpy(pihStaticAgents,piStaticAgents,sizeof(int),cudaMemcpyDeviceToHost));
//					printf ("static agents (cumulative):%d \n",pihStaticAgents[0]);
					break;
				case DIE:
					register_deaths_fs<<<1,1>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
						pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize);
					break;
				default:
					break;
			}
		} else {
			switch (routine) {
				case COUNT:
					count_occupancy_fs<<<1,1>>>(psaX,psaY,pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize);
				break;
				case MOVE:
					best_move_by_traversal_fs<<<1,1>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pigGridBits,
						pigResidents,piaQueueA,ihActiveQueueSize,piStaticAgents);
					CUDA_CALL(cudaMemcpy(pihStaticAgents,piStaticAgents,sizeof(int),cudaMemcpyDeviceToHost));
//					printf ("static agents (cumulative):%d \n",pihStaticAgents[0]);
					break;
				case DIE:
					register_deaths_fs<<<1,1>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
						pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize);
					break;
				default:
					break;
			}
		}
		CUDA_CALL(cudaDeviceSynchronize());
	}

	// cleanup
	free(piahTemp);

	return status;
}
