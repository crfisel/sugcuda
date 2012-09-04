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
#include "constants.h"
#include "exercise_locks.h"
#include "randoms.h"
#include "move.h"
#include "count.h"
#include "mate.h"
#include "die.h"


__global__ void initialize_queue(int* piaQueue, int iQueueLength)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	if (iAgentID < iQueueLength) piaQueue[iAgentID] = iAgentID;
}

int exercise_locks(operation routine, short* psaX, short* psaY, int* piaAgentBits, int* pigGridBits, int* pigResidents,
		float* pfaSugar, float* pfaSpice, float* pfaInitialSugar, float* pfaInitialSpice,
		int* piaQueueA, int* piaQueueB, int* piPopulation, int* pihPopulation,
		int* piDeferredQueueSize, int* pihDeferredQueueSize, curandState* paStates,
		int* piLockSuccesses, int* pihLockSuccesses, int* piStaticAgents, int* pihStaticAgents)
{
	int status;
	if (routine < MAX_OPCODE) {
		status = EXIT_SUCCESS;

		// sync the host and device readings of population
		CUDA_CALL(cudaMemcpy(pihPopulation,piPopulation,sizeof(int),cudaMemcpyDeviceToHost));

		// fill the agent queue with increasing id's
		int ihNumBlocks = (pihPopulation[0]+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
		initialize_queue<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(piaQueueA,pihPopulation[0]);

		// blank the deferred queue with all bits=-1
		CUDA_CALL(cudaMemset(piaQueueB,0xFF,pihPopulation[0]*sizeof(int)));

		// zero the deferred queue size
		CUDA_CALL(cudaMemset(piDeferredQueueSize,0,sizeof(int)));

		// zero the successful locks counter
		CUDA_CALL(cudaMemset(piLockSuccesses,0,sizeof(int)));

		// zero the static agents counter
		CUDA_CALL(cudaMemset(piStaticAgents,0,sizeof(int)));

		CUDA_CALL(cudaDeviceSynchronize());

		// call first iteration on parallel version with locking
		switch (routine) {
		case COUNT:
			count_occupancy<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pigGridBits,pigResidents,
					piaQueueA,piaQueueB,pihPopulation[0],piDeferredQueueSize,piLockSuccesses);
			break;
		case MOVE:
			best_move_by_traversal<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
					pfaSugar,pfaSpice,piaQueueA,piaQueueB,pihPopulation[0],piDeferredQueueSize,piLockSuccesses,piStaticAgents);
			//	CUDA_CALL(cudaMemcpy(pihStaticAgents,piStaticAgents,sizeof(int),cudaMemcpyDeviceToHost));
			//	printf ("static agents:%d \n",pihStaticAgents[0]);
			break;
		case MATE:
			mate_masked<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
					pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,
					piaQueueA,piaQueueB,pihPopulation[0],piPopulation,paStates,piDeferredQueueSize,piLockSuccesses);
			break;
		case DIE:
			register_deaths<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
					pfaSugar,pfaSpice,piaQueueA,piaQueueB,pihPopulation[0],piDeferredQueueSize,piLockSuccesses);
			break;
		default:
			printf("Error: opcode does not exist\n");
			break;
		}
		CUDA_CALL(cudaDeviceSynchronize());

		// check if any agents had to be deferred
		CUDA_CALL(cudaMemcpy(pihDeferredQueueSize,piDeferredQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
		//	printf ("primary deferrals:%d \n",pihDeferredQueueSize[0]);
		// CUDA_CALL(cudaMemcpy(pihLockSuccesses,piLockSuccesses,sizeof(int),cudaMemcpyDeviceToHost));
		//	printf ("successful locks:%d \n",pihLockSuccesses[0]);


		// handle the deferred queue until it is empty
		int ihActiveQueueSize = pihPopulation[0];
		bool hQueueSwitched = true;
		// bounce queues until there are too few agents, or until deferrals stop decreasing
		while (pihDeferredQueueSize[0] > 10 && pihDeferredQueueSize[0] < ihActiveQueueSize) {
			ihActiveQueueSize = pihDeferredQueueSize[0];
			ihNumBlocks = (ihActiveQueueSize+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
			CUDA_CALL(cudaMemset(piDeferredQueueSize,0,sizeof(int)));
			if (hQueueSwitched) {
				// switch the two queues and handle deferred agents
				CUDA_CALL(cudaMemset(piaQueueA,0xFF,pihPopulation[0]*sizeof(int)));
				switch (routine) {
				case COUNT:
					count_occupancy<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pigGridBits,pigResidents,
							piaQueueB,piaQueueA,ihActiveQueueSize,piDeferredQueueSize,piLockSuccesses);
					break;
				case MOVE:
					best_move_by_traversal<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
							pfaSugar,pfaSpice,piaQueueB,piaQueueA,ihActiveQueueSize,piDeferredQueueSize,piLockSuccesses,piStaticAgents);
					//		CUDA_CALL(cudaMemcpy(pihStaticAgents,piStaticAgents,sizeof(int),cudaMemcpyDeviceToHost));
					//					printf ("static agents (cumulative):%d \n",pihStaticAgents[0]);
					break;
				case MATE:
					mate_masked<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
							pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,
							piaQueueB,piaQueueA,ihActiveQueueSize,piPopulation,paStates,piDeferredQueueSize,piLockSuccesses);
					break;
				case DIE:
					register_deaths<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
							pfaSugar,pfaSpice,piaQueueB,piaQueueA,ihActiveQueueSize,piDeferredQueueSize,piLockSuccesses);
					break;
				default:
					printf("Error: opcode does not exist\n");
					break;
				}
			} else {
				// switch queues the other way
				CUDA_CALL(cudaMemset(piaQueueB,0xFF,pihPopulation[0]*sizeof(int)));
				switch (routine) {
				case COUNT:
					count_occupancy<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pigGridBits,pigResidents,
							piaQueueA,piaQueueB,ihActiveQueueSize,piDeferredQueueSize,piLockSuccesses);
					break;
				case MOVE:
					best_move_by_traversal<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
							pfaSugar,pfaSpice,piaQueueA,piaQueueB,ihActiveQueueSize,piDeferredQueueSize,piLockSuccesses,piStaticAgents);
					//			CUDA_CALL(cudaMemcpy(pihStaticAgents,piStaticAgents,sizeof(int),cudaMemcpyDeviceToHost));
					//					printf ("static agents (cumulative):%d \n",pihStaticAgents[0]);
					break;
				case MATE:
					mate_masked<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
							pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,
							piaQueueA,piaQueueB,ihActiveQueueSize,piPopulation,paStates,piDeferredQueueSize,piLockSuccesses);
					break;
				case DIE:
					register_deaths<<<ihNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
							pfaSugar,pfaSpice,piaQueueA,piaQueueB,ihActiveQueueSize,piDeferredQueueSize,piLockSuccesses);
					break;
				default:
					printf("Error: opcode does not exist\n");
					break;
				}
			}
			CUDA_CALL(cudaDeviceSynchronize());
			hQueueSwitched = !hQueueSwitched;
			CUDA_CALL(cudaMemcpy(pihDeferredQueueSize,piDeferredQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
			//		printf ("secondary deferrals:%d \n",pihDeferredQueueSize[0]);
			//			CUDA_CALL(cudaMemcpy(pihLockSuccesses,piLockSuccesses,sizeof(int),cudaMemcpyDeviceToHost));
			//		printf ("successful locks (cumulative):%d \n",pihLockSuccesses[0]);
		}

		// for persistent lock failures, use the fail-safe version (once)
		if (pihDeferredQueueSize[0] > 0 && (pihDeferredQueueSize[0] <= 10 || pihDeferredQueueSize[0] >= ihActiveQueueSize)) {
			ihActiveQueueSize = pihDeferredQueueSize[0];
			if (hQueueSwitched) {
				switch (routine) {
				case COUNT:
					count_occupancy_fs<<<1,1>>>(psaX,psaY,pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize);
					break;
				case MOVE:
					best_move_by_traversal_fs<<<1,1>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
							pfaSugar,pfaSpice,piaQueueB,ihActiveQueueSize,piStaticAgents);
//					CUDA_CALL(cudaMemcpy(pihStaticAgents,piStaticAgents,sizeof(int),cudaMemcpyDeviceToHost));
					//					printf ("static agents (cumulative):%d \n",pihStaticAgents[0]);
					break;
				case DIE:
					register_deaths_fs<<<1,1>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,pfaSugar,pfaSpice,
							piaQueueB,ihActiveQueueSize);
					break;
				default:
					printf("Error: opcode does not exist\n");
					break;
				}
			} else {
				switch (routine) {
				case COUNT:
					count_occupancy_fs<<<1,1>>>(psaX,psaY,pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize);
					break;
				case MOVE:
					best_move_by_traversal_fs<<<1,1>>>(psaX,psaY,piaAgentBits,pigGridBits,pigResidents,
							pfaSugar,pfaSpice,piaQueueA,ihActiveQueueSize,piStaticAgents);
					//				CUDA_CALL(cudaMemcpy(pihStaticAgents,piStaticAgents,sizeof(int),cudaMemcpyDeviceToHost));
					//					printf ("static agents (cumulative):%d \n",pihStaticAgents[0]);
					break;
				case DIE:
					register_deaths_fs<<<1,1>>>(psaX,psaY,piaAgentBits,	pigGridBits,pigResidents,pfaSugar,pfaSpice,
							piaQueueA,ihActiveQueueSize);
					break;
				default:
					printf("Error: opcode does not exist\n");
					break;
				}
			}
			CUDA_CALL(cudaDeviceSynchronize());
		}

	} else {
		status = EXIT_OPCODE_OVERFLOW;
	}
	return status;
}
