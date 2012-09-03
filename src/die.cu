#include <cuda.h>
#include <stdio.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "die.h"

// this kernel has one thread per agent
__global__ void register_deaths(short* psaX, short* psaY, int* piaAgentBits, short* psaAge, 
		float* pfaSugar, float* pfaSpice, int* pigGridBits, int* pigResidents, int* piaActiveQueue, 
		const int ciActiveQueueSize, int* piaDeferredQueue, int* piDeferredQueueSize, int* piLockSuccesses)
{
	int iAgentID;
	int iFlag = 0;
	bool lockFailed = false;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		iAgentID = piaActiveQueue[iOffset];

		// if agent is alive
		if (psaX[iAgentID] > -1) {

			// check for death by old age
			// reinterpret piaAgentBits bitwise for death age
			AgentBitWise abwBits;
			abwBits.asInt = piaAgentBits[iAgentID];
			if ((psaAge[iAgentID] > 64+(abwBits.asBits.deathAge)) ||
					// check for starvation
					(pfaSpice[iAgentID] < 0.0f) || (pfaSpice[iAgentID] < 0.0f)) {
		//printf("age %d death age %d sugar %f spice %f\n",psaAge[iAgentID],60+2*((&piaAgentBits[iAgentID])->deathAge),pfaSugar[iAgentID],pfaSpice[iAgentID]);
				// lock address to register death - if lock fails, defer

				// current agent's address in the grid
				int iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];
				GridBitWise gbwBits;
				gbwBits.asInt = pigGridBits[iAddy];

				// test if address is locked
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
						// at this point, the square is locked and a valid copy of its bits are in gbwBitsCopy (because locked)
						iFlag = atomicAdd(piLockSuccesses,1);

						// decrement occupancy at old address
						short sOldOcc = gbwBitsCopy.asBits.occupancy;
						gbwBitsCopy.asBits.occupancy--;
						if (sOldOcc > 0) {
							// find match starting at end of list
							short k = gbwBitsCopy.asBits.occupancy;
							// remove current id - if it is not at the end, replace it with the one from the end and store -1 at the end
							if (pigResidents[iAddy*MAX_OCCUPANCY+k] == iAgentID) {
								pigResidents[iAddy*MAX_OCCUPANCY+k] = -1;
							} else {
								while (pigResidents[iAddy*MAX_OCCUPANCY+k] != iAgentID && k >= 0) {k--;}
								if (k != gbwBitsCopy.asBits.occupancy) {
									pigResidents[iAddy*MAX_OCCUPANCY+k] = 
										pigResidents[iAddy*MAX_OCCUPANCY+gbwBitsCopy.asBits.occupancy];
									pigResidents[iAddy*MAX_OCCUPANCY+gbwBitsCopy.asBits.occupancy] = -1;
								}
							}
						} else {
							// in case of bugs (i.e. old occupancy was already zero), report problem
							printf ("underflow occupancy %d at x:%d y:%d agent id %d \n",sOldOcc,psaX[iAgentID],psaY[iAgentID],iAgentID);
						}

						// unlock and update global occupancy values
						gbwBitsCopy.asBits.isLocked = 0;
						iFlag = atomicExch(&(pigGridBits[iAddy]),gbwBitsCopy.asInt);

						// mark agent as dead
						psaX[iAgentID] *= -1;	
					}
				}

				// if a move was warranted, but lock failures prevented it, defer
				if (lockFailed) {
					// if either lock failed or either agent was already locked, add the agent to the deferred queue
					iFlag = atomicAdd(piDeferredQueueSize,1);
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
	int iAgentID;

	// only the 1,1 block is active
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// iterate through the active queue
		for (int iOffset = 0; iOffset < ciActiveQueueSize; iOffset++) {

			// get the iAgentID from the active agent queue
			iAgentID = piaActiveQueue[iOffset];

			// if agent is alive
			if (psaX[iAgentID] > -1) {

				// check for death by old age
				// reinterpret piaAgentBits bitwise for death age
				AgentBitWise abwBits;
				abwBits.asInt = piaAgentBits[iAgentID];
				if ((psaAge[iAgentID] > 64+(abwBits.asBits.deathAge)) ||
						// check for starvation
						(pfaSpice[iAgentID] < 0.0f) || (pfaSpice[iAgentID] < 0.0f)) {
					//printf("age %d sugar %f spice %f\n",psaAge[iAgentID],pfaSugar[iAgentID],pfaSpice[iAgentID]);
					// current agent's address in the grid
					int iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];
					//	printf("death %d at %d:%d\n",iAgentID,psaX[iAgentID],psaY[iAgentID]);

					// unpack grid bits
					GridBitWise gbwBits;
					gbwBits.asInt = pigGridBits[iAddy];
					// decrement occupancy at old address
					short sOldOcc = gbwBits.asBits.occupancy;
					gbwBits.asBits.occupancy--;
					if (sOldOcc > 0) {
						// find match starting at end of list
						short k = gbwBits.asBits.occupancy;
						// remove current id - if it is not at the end, replace it with the one from the end and store -1 at end
						if (pigResidents[iAddy*MAX_OCCUPANCY+k] == iAgentID) {
							pigResidents[iAddy*MAX_OCCUPANCY+k] = -1;
						} else {
							while (pigResidents[iAddy*MAX_OCCUPANCY+k] != iAgentID && k >= 0) {k--;}
							if (k != gbwBits.asBits.occupancy) {
								pigResidents[iAddy*MAX_OCCUPANCY+k] = 
										pigResidents[iAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy];
								pigResidents[iAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy] = -1;
							} 
						}
					} else {
						// in case of bugs (i.e. old occupancy was already zero), report problem
						printf ("underflow occupancy %d at x:%d y:%d agent id %d \n",sOldOcc,psaX[iAgentID],psaY[iAgentID],iAgentID);
					}
					// mark agent as dead
					psaX[iAgentID] *= -1;	
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

	// find best move for agents at the head of their square's resident list
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


	// handle the deferred queue until it is empty
	int ihActiveQueueSize = iQueueSize;
	bool hQueue = true;
	while (pihDeferredQueueSize[0] > 100 && pihDeferredQueueSize[0] < ihActiveQueueSize) {
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
