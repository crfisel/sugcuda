#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwisetype.h"
#include "mate.h"

__device__ bool isFertile(int iAgentID, BitWiseType* pbaBits, short* psaAge)
{
	bool ydResult = (psaX[iAgentID] > -1) &&
			(psaAge[iAgentID] > ((&pbaBits[iAgentID])->startFertilityAge + 12)) &&
			(psaAge[iAgentID] < (2*((&pbaBits[iAgentID])->endFertilityAge) + 40));
	return ydResult;
}
__global__ void mate(short* psaX, short* psaY, short* psgOccupancy, int* pigResidents, int* pigLocks, 
		int* piaActiveQueue, const int ciActiveQueueSize, int* piaDeferredQueue, int* piDeferredQueueSize, 
		int* piLockSuccesses)
{
	int iAgentID;
	int iMateID;
	int iAddy;
	int iTemp;
	short sOldOcc;
	bool pregnant = false;
	bool lockFailed = false;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		iAgentID = piaActiveQueue[iOffset];

		// live, fertile, solvent female agents only
		if (((&pbaBits[iAgentID])->isFemale) == 1) {
			if (isFertile(iAgentID)) {
				if	((pfaSugar[iAgentID] > pfaInitialSugar[iAgentID]) &&
						(pfaSpice[iAgentID] > pfaInitialSpice[iAgentID])) {
					for (short sXTry = psaX[iAgentID] - 1; sXTry < psaX[iAgentID] + 2; sXTry++) {
						for (short sYTry = psaY[iAgentID] - 1; sYTry < psaY[iAgentID] + 2; sYTry++) {
							short sOccTry = 0;
							while (sOccTry < MAX_OCCUPANCY && !pregnant) {
								// get the potential mate's id
								int iMateID = pigResidents[(sXTry*GRID_SIZE+sYTry)*MAX_OCCUPANCY+sOccTry];
								// make sure this is not an "empty" placeholder
								if (iMateID > -1) {
									// make sure he's male, alive and fertile
									if ((&pbaBits[iAgentID])->isFemale == 0 && isFertile(iMateID)) {
										// if he's unlocked...
										short sHisAge = psaAge[iMateID];
										if (sHisAge > 0) {
											// lock him if possible by changing his age to negative
											int iTemp = atomicCAS(&(psaAge[iMateID]),sHisAge,-sHisAge);
											if (iTemp == sHisAge) {
												// now he's locked, check his solvency
												if	((pfaSugar[iMateID] > pfaInitialSugar[iMateID]) &&
														(pfaSpice[iMateID] > pfaInitialSpice[iMateID])) {
													// ok he's a keeper, make a baby...
													pregnant = true;
												} else {
													// unlock him
													iTemp = atomicExch(&(psaAge[iMateID]),abs(sHisAge));
												}
											} else {
												lockFailed = true;
											}
										}
									}
								}
							}
						}
					}
					// TODO: if lock failed anywhere, defer this agent
				}
			}
		}
	}



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
	while (pihDeferredQueueSize[0] > 10 && pihDeferredQueueSize[0] < ihActiveQueueSize) {
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
	if (pihDeferredQueueSize[0] <= 10 || pihDeferredQueueSize[0] >= ihActiveQueueSize) {
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
