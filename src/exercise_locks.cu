/*
 * exercise_locks.cu
 *
 *  Created on: Dec 17, 2011
 *      Author: C. Richard Fisel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "exercise_locks.h"
#include "move.h"
#include "count.h"
#include "die.h"


int exercise_locks(short routine, short* psaX, short* psaY, int* piaAgentBits, float* pfaSugar, float* pfaSpice, short* psaAge, int* pigGridBits, 
	int* pigResidents, int* piaQueueA, int* piPopulation, int* pihPopulation, int* piaQueueB, int* piDeferredQueueSize, int* piLockSuccesses)
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

	// blank the deferred queue with all bits=1
	CUDA_CALL(cudaMemset(piaQueueB,0xFF,pihPopulation[0]*sizeof(int)));

	// zero the deferred queue size
	CUDA_CALL(cudaMemset(piDeferredQueueSize,0,sizeof(int)));

	// zero the successful locks counter
	CUDA_CALL(cudaMemset(piLockSuccesses,0,sizeof(int)));

	// call first iteration on parallel version with locking
	int hiNumBlocks = (pihPopulation[0]+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
	switch (routine) {
		case COUNT:
			count_occupancy<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pigGridBits,pigResidents,
				piaQueueA,pihPopulation[0],piaQueueB,piDeferredQueueSize,piLockSuccesses);
			break;
		case MOVE:
			best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
				pigGridBits,pigResidents,piaQueueA,pihPopulation[0],piaQueueB,piDeferredQueueSize,piLockSuccesses);
			break;
		case DIE:
			register_deaths<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,
				pigGridBits,pigResidents,piaQueueA,pihPopulation[0],piaQueueB,piDeferredQueueSize,piLockSuccesses);
			break;
		default:
			break;
	}
	cudaDeviceSynchronize();

	// check if any agents had to be deferred
	int* pihDeferredQueueSize = (int*) malloc(sizeof(int));
	CUDA_CALL(cudaMemcpy(pihDeferredQueueSize,piDeferredQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
	printf ("primary deferrals:%d \n",pihDeferredQueueSize[0]);
	int* pihLockSuccesses = (int*) malloc(sizeof(int));
	CUDA_CALL(cudaMemcpy(pihLockSuccesses,piLockSuccesses,sizeof(int),cudaMemcpyDeviceToHost));
	printf ("successful locks:%d \n",pihLockSuccesses[0]);


	// handle the deferred queue until it is empty
	int ihActiveQueueSize = pihPopulation[0];
	bool hQueue = true;
	while (pihDeferredQueueSize[0] > 10 && pihDeferredQueueSize[0] < ihActiveQueueSize) {
		ihActiveQueueSize = pihDeferredQueueSize[0];
		hiNumBlocks = (ihActiveQueueSize+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
		CUDA_CALL(cudaMemset(piDeferredQueueSize,0,sizeof(int)));
		CUDA_CALL(cudaMemset(piLockSuccesses,0,sizeof(int)));
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
						pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize,piaQueueA,piDeferredQueueSize,piLockSuccesses);
					break;
				case DIE:
					register_deaths<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,
						pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize,piaQueueA,piDeferredQueueSize,piLockSuccesses);
					break;
				default:
					break;
			}
		} else {
			// switch the other way
			CUDA_CALL(cudaMemset(piaQueueB,0xFF,pihPopulation[0]*sizeof(int)));
			switch (routine) {
				case COUNT:
					count_occupancy<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pigGridBits,pigResidents,
						piaQueueA,ihActiveQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
				break;
				case MOVE:
					best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
						pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
					break;
				case DIE:
					register_deaths<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,
						pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
					break;
				default:
					break;
			}
		}
		cudaDeviceSynchronize();
		hQueue = !hQueue;
		CUDA_CALL(cudaMemcpy(pihDeferredQueueSize,piDeferredQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
		printf ("secondary deferrals:%d \n",pihDeferredQueueSize[0]);
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
						pigResidents,piaQueueB,ihActiveQueueSize);
					break;
				case DIE:
					register_deaths_fs<<<1,1>>>(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,
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
						pigResidents,piaQueueA,ihActiveQueueSize);
					break;
				case DIE:
					register_deaths_fs<<<1,1>>>(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,
						pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize);
					break;
				default:
					break;
			}
		}
		cudaDeviceSynchronize();
	}

	// cleanup
	free(pihLockSuccesses);
	free(pihDeferredQueueSize);
	free(piahTemp);

	return status;
}
