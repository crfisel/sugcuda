#include <cuda.h>
#include <stdio.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "move.h"
#include "die.h"

// this kernel has one thread per agent
__global__ void register_deaths(short* psaX, short* psaY, int* piaAgentBits, short* psaAge, 
		float* pfaSugar, float* pfaSpice, int* pigGridBits, int* pigResidents, int* piaActiveQueue, 
		const int ciActiveQueueSize, int* piaDeferredQueue, int* piDeferredQueueSize, int* piLockSuccesses)
{
	bool lockFailed = false;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		int iAgentID = piaActiveQueue[iOffset];

		// if agent is alive
		if (psaX[iAgentID] > -1) {

			// check for death by old age or starvation
			// reinterpret piaAgentBits[iAgentID] bitwise
			AgentBitWise abwBits;
			abwBits.asInt = piaAgentBits[iAgentID];

			if ((psaAge[iAgentID] > 64+abwBits.asBits.deathAge) || (pfaSpice[iAgentID] < 0.0f) || (pfaSpice[iAgentID] < 0.0f)) {

				// lock address to register death - if lock fails, defer
				// current agent's address in the grid
				int iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];

				// unpack grid bits
				GridBitWise gbwBits;
				gbwBits.asInt = pigGridBits[iAddy];

				// test if square is locked
				if (gbwBits.asBits.isLocked != 0) {
					// if so, lock failed
					lockFailed = true;

				} else {
					// if not, make a copy, but indicating locked
					GridBitWise gbwBitsCopy = gbwBits;
					gbwBitsCopy.asBits.isLocked = 1;

					// now lock the current address if possible
					int iLocked = atomicCAS(&(pigGridBits[iAddy]),gbwBits.asInt,gbwBitsCopy.asInt);
	
					// test if the lock failed
					if (iLocked != gbwBits.asInt) {
						lockFailed = true;
							
					} else {
						// at this point, square is locked and a valid copy of its bits are in gbwBitsCopy (because locked)
							int iFlag = atomicAdd(piLockSuccesses,1);

						// before inserting new resident, check for nonzero occupancy
						if (gbwBitsCopy.asBits.occupancy <= 0) {
									
							// if invalid, unlock with no changes
							iFlag = atomicExch(&(pigGridBits[iAddy]),gbwBits.asInt);
								
							// and indicate an error
							printf("underflow occ %d at x:%d y:%d agent %d res %d\n",
								gbwBitsCopy.asBits.occupancy,psaX[iAgentID],psaY[iAgentID],iAgentID,pigResidents[iAddy*MAX_OCCUPANCY]);

						} else {
							remove_resident(&(gbwBitsCopy.asInt),iAddy,pigResidents,iAgentID);

							// mark agent as dead
							psaX[iAgentID] *= -1;
							
							// unlock and update global occupancy values
							gbwBitsCopy.asBits.isLocked = 0;
							iFlag = atomicExch(&(pigGridBits[iAddy]),gbwBitsCopy.asInt);
						}
					}
				}
				// if a death occurred, but lock failures prevented registering it, defer
				if (lockFailed) {
					int iFlag = atomicAdd(piDeferredQueueSize,1);
					piaDeferredQueue[iFlag]=iAgentID;
				}
			}
		}
	}
	return;
}

// this "failsafe" kernel has one thread, for persistent lock failures
__global__ void register_deaths_fs(short* psaX, short* psaY, int* piaAgentBits, short* psaAge,
		float* pfaSugar, float* pfaSpice, int* pigGridBits, int* pigResidents, 
		int* piaActiveQueue, const int ciActiveQueueSize)
{
	
	// only the 1,1 block is active
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// iterate through the active queue
		for (int iOffset = 0; iOffset < ciActiveQueueSize; iOffset++) {

			// get the iAgentID from the active agent queue
			int iAgentID = piaActiveQueue[iOffset];

			// if agent is alive
			if (psaX[iAgentID] > -1) {

				// reinterpret piaAgentBits bitwise for death age
				AgentBitWise abwBits;
				abwBits.asInt = piaAgentBits[iAgentID];
				// check for death by old age or starvation
				if ((psaAge[iAgentID] > 64+(abwBits.asBits.deathAge)) || (pfaSpice[iAgentID] < 0.0f) || (pfaSpice[iAgentID] < 0.0f)) {
					
					// current agent's address in the grid
					int iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];
					
					// unpack grid bits
					GridBitWise gbwBits;
					gbwBits.asInt = pigGridBits[iAddy];

					// before removing resident, check for nonzero occupancy
					if (gbwBits.asBits.occupancy <= 0) {
									
						// if invalid, indicate an error
						printf("under occ %d at x:%d y:%d agent %d\n",gbwBits.asBits.occupancy,psaX[iAgentID],psaY[iAgentID],iAgentID);

					} else {
						remove_resident(&(gbwBits.asInt),iAddy,pigResidents,iAgentID);

						// mark agent as dead
						psaX[iAgentID] *= -1;
								
						// update global occupancy values
						int iFlag = atomicExch(&(pigGridBits[iAddy]),gbwBits.asInt);
					}
				}
			}
		}
	}
	return;
}

int die(short* psaX, short* psaY, int* piaAgentBits, short* psaAge, float* pfaSugar, float* pfaSpice, int* pigGridBits, 
		int* pigResidents, int* piaQueueA, const int iQueueSize, int* piaQueueB, int* piDeferredQueueSize, int* piLockSuccesses)
{
	int status = EXIT_SUCCESS;

	// fill the agent queue with increasing (later random) id's
	int* piahTemp = (int*) malloc(iQueueSize*sizeof(int));
	for (int i = 0; i < iQueueSize; i++) {
		//		piahTemp[i] = rand() % iQueueSize;
		piahTemp[i] = i;
	}
	CUDA_CALL(cudaMemcpy(piaQueueA,piahTemp,iQueueSize*sizeof(int),cudaMemcpyHostToDevice));

	// blank the deferred queue with all bits=1
	CUDA_CALL(cudaMemset(piaQueueB,0xFF,iQueueSize*sizeof(int)));

	// zero the deferred queue size
	CUDA_CALL(cudaMemset(piDeferredQueueSize,0,sizeof(int)));

	// zero the successful locks counter
	CUDA_CALL(cudaMemset(piLockSuccesses,0,sizeof(int)));

	// register deaths for agents
	int hiNumBlocks = (iQueueSize+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
	register_deaths<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,
			pigGridBits,pigResidents,piaQueueA,iQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
	cudaDeviceSynchronize();

	// check if any agents had to be deferred
	int* pihDeferredQueueSize = (int*) malloc(sizeof(int));
	CUDA_CALL(cudaMemcpy(pihDeferredQueueSize,piDeferredQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
	printf ("primary deferrals:%d \n",pihDeferredQueueSize[0]);
	int* pihLockSuccesses = (int*) malloc(sizeof(int));
	CUDA_CALL(cudaMemcpy(pihLockSuccesses,piLockSuccesses,sizeof(int),cudaMemcpyDeviceToHost));
	printf ("successful locks:%d \n",pihLockSuccesses[0]);


	// handle the deferred queue until it is nearly empty
	int ihActiveQueueSize = iQueueSize;
	bool hQueue = true;
	while (pihDeferredQueueSize[0] > 10 && pihDeferredQueueSize[0] < ihActiveQueueSize) {
		ihActiveQueueSize = pihDeferredQueueSize[0];
		hiNumBlocks = (ihActiveQueueSize+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
		CUDA_CALL(cudaMemset(piDeferredQueueSize,0,sizeof(int)));
		CUDA_CALL(cudaMemset(piLockSuccesses,0,sizeof(int)));
		if (hQueue) {
			CUDA_CALL(cudaMemset(piaQueueA,0xFF,iQueueSize*sizeof(int)));
			register_deaths<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,
					pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize,piaQueueA,piDeferredQueueSize,piLockSuccesses);

		} else {
			CUDA_CALL(cudaMemset(piaQueueB,0xFF,iQueueSize*sizeof(int)));
			register_deaths<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,
					pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
		}
		cudaDeviceSynchronize();
		hQueue = !hQueue;
		CUDA_CALL(cudaMemcpy(pihDeferredQueueSize,piDeferredQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
		printf ("secondary deferrals:%d \n",pihDeferredQueueSize[0]);
	} 

	// for persistent lock failures, use the failsafe version
	if (pihDeferredQueueSize[0] <= 10 || pihDeferredQueueSize[0] >= ihActiveQueueSize) {
		ihActiveQueueSize = pihDeferredQueueSize[0];
		if (hQueue) {
			register_deaths_fs<<<1,1>>>(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,
					pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize);

		} else {
			register_deaths_fs<<<1,1>>>(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,
					pigGridBits,pigResidents,piaQueueA,ihActiveQueueSize);
		}
		cudaDeviceSynchronize();
	}

	// cleanup
	free(pihLockSuccesses);
	free(pihDeferredQueueSize);
	free(piahTemp);

	return status;
} 
