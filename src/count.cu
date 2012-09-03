#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwisetype.h"
#include "count.h"

__global__ void count_occupancy(short* psaX, short* psaY, short* psgOccupancy, int* pigResidents, int* pigLocks, 
		int* piaActiveQueue, const int ciActiveQueueSize, int* piaDeferredQueue, int* piDeferredQueueSize, 
		int* piLockSuccesses)
{
	int iAgentID;
	int iAddy;
	int iTemp;
	short sOldOcc;


	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		iAgentID = piaActiveQueue[iOffset];

		// work with live agents only
		if (psaX[iAgentID] > -1) {

			// current agent's address in the grid
			iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];

			// lock the current address if possible
			iTemp = atomicCAS(&(pigLocks[iAddy]), 0, 1);
			if (iTemp == 0) {

				// increment square occupancy and counter of successful locks
				sOldOcc = psgOccupancy[iAddy]++;
				iTemp = atomicAdd(piLockSuccesses,1);

				// check for overflow
				if (sOldOcc < MAX_OCCUPANCY) {

					// insert the resident at the next position in the pigResidents list
					pigResidents[iAddy*MAX_OCCUPANCY+sOldOcc] = iAgentID;

				} else {

					// indicate an occupancy overflow
					printf ("overflow at x:%d y:%d \n",psaX[iAgentID], psaY[iAgentID]);
					psgOccupancy[iAddy] = MAX_OCCUPANCY+1;
				}

				// unlock the square
				iTemp = atomicExch(&(pigLocks[iAddy]),0);
			}
			else {

				// otherwise, add the agent to the deferred queue
				iTemp = atomicAdd(piDeferredQueueSize,1);
				piaDeferredQueue[iTemp]=iAgentID;
			}
		}
	}
	return;
}

// this "failsafe" kernel has one thread, for persistent lock failures
__global__ void count_occupancy_fs(short* psaX, short* psaY, short* psgOccupancy, int* pigResidents,
		int* piaActiveQueue, const int ciActiveQueueSize)
{
	int iAgentID;
	int iAddy;
	short sOldOcc;

	// only the 1,1 block is active
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// iterate through the active queue
		for (int iOffset = 0; iOffset < ciActiveQueueSize; iOffset++) {

			// get agent id
			iAgentID = piaActiveQueue[iOffset];

			// work with live agents only
			if (psaX[iAgentID] > -1) {

				// current agent's address in the grid
				iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];

				// no locks necessary, so increment square occupancy
				sOldOcc = psgOccupancy[iAddy]++;

				// check for overflow
				if (sOldOcc < MAX_OCCUPANCY) {

					// insert the resident at the next position in the pigResidents list
					pigResidents[iAddy*MAX_OCCUPANCY+sOldOcc] = iAgentID;

				} else {

					// indicate an occupancy overflow
					printf ("overflow at x:%d y:%d \n",psaX[iAgentID], psaY[iAgentID]);
					psgOccupancy[iAddy] = MAX_OCCUPANCY+1;
				}

			}
		}
	}
	return;
}

int count(short* psaX, short* psaY, short* psgOccupancy, int* pigResidents, int* pigLocks, 
		int* piaQueueA, const int iQueueSize, int* piaQueueB, int* piDeferredQueueSize, int* piLockSuccesses)
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

	// count agents at each grid location
	int hiNumBlocks = (iQueueSize+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
	count_occupancy<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,psgOccupancy,pigResidents,pigLocks,
			piaQueueA,iQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
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
			count_occupancy<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,psgOccupancy,pigResidents,pigLocks,
					piaQueueB,ihActiveQueueSize,piaQueueA,piDeferredQueueSize,piLockSuccesses);
		}
		else {
			CUDA_CALL(cudaMemset(piaQueueB,0xFF,iQueueSize*sizeof(int)));
			count_occupancy<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,psgOccupancy,pigResidents,pigLocks,
					piaQueueA,ihActiveQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
		}
		cudaDeviceSynchronize();
		hQueue = !hQueue;
		CUDA_CALL(cudaMemcpy(pihDeferredQueueSize,piDeferredQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
		printf ("secondary deferrals:%d \n",pihDeferredQueueSize[0]);
	}

	// for persistent lock failures, use the failsafe version
	if (pihDeferredQueueSize[0] <= 100 || pihDeferredQueueSize[0] >= ihActiveQueueSize) {
		ihActiveQueueSize = pihDeferredQueueSize[0];
		if (hQueue) {
			count_occupancy_fs<<<1,1>>>(psaX,psaY,psgOccupancy,pigResidents,piaQueueB,ihActiveQueueSize);

		} else {
			count_occupancy_fs<<<1,1>>>(psaX,psaY,psgOccupancy,pigResidents,piaQueueA,ihActiveQueueSize);
		}
		cudaDeviceSynchronize();
	}

	// check for overflows
	short* psghOccupancy = (short*) malloc(GRID_SIZE*GRID_SIZE*sizeof(short));
	CUDA_CALL(cudaMemcpy(psghOccupancy,psgOccupancy,GRID_SIZE*GRID_SIZE*sizeof(short),cudaMemcpyDeviceToHost));
	for (int k = 0; k < GRID_SIZE*GRID_SIZE; k++) {
		if (psghOccupancy[k] > MAX_OCCUPANCY) {
			printf ("Occupancy overflow at square %d\n",k);
			status = OCCUPANCY_OVERFLOW;
		}
	}

	// cleanup
	free(pihLockSuccesses);
	free(pihDeferredQueueSize);
	free(psghOccupancy);
	free(piahTemp);

	return status;
}
