#include <cuda.h>
#include <stdio.h>
#include "symbolic_constants.h"
#include "bitwisetype.h"
#include "move.h"

// this kernel has one thread per agent, each traversing the local neighborhood prescribed by its vision
// NOTE: NUM_AGENTS is an int, GRID_SIZE is a short
__global__ void best_move_by_traversal(short* psaX, short* psaY, BitWiseType* pbaBits, 
		float* pfaSugar, float* pfaSpice, short* psgSugar, short* psgSpice,
		short* psgOccupancy, int* pigResidents, int* pigLocks, int* piaActiveQueue,
		const int ciActiveQueueSize, int* piaDeferredQueue, int* piDeferredQueueSize,
		int* piLockSuccesses)
{
	int iAgentID;
	int iTemp = 0;
	int iFlag = 0;
	int iLockedOld = 0;
	int iLockedNew = 0;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		iAgentID = piaActiveQueue[iOffset];

		// work with live agents only
		if (psaX[iAgentID] > -1) {

			// begin finding best move, searching around current (center) square
			short sXCenter = psaX[iAgentID];
			short sYCenter = psaY[iAgentID];
			short sXStore = sXCenter;
			short sYStore = sYCenter;

			// scale values by agent's current sugar and spice levels,
			// converting to a duration using metabolic rates
			// add .01 to current values to avoid div by zero
			float sugarScale = 1.0f/(pfaSugar[iAgentID]+0.01f)/((&pbaBits[iAgentID])->metSugar+1);
			float spiceScale = 1.0f/(pfaSpice[iAgentID]+0.01f)/((&pbaBits[iAgentID])->metSpice+1);

			// search limits based on vision
			short sXMin = sXCenter-(&pbaBits[iAgentID])->vision-1;
			short sXMax = sXCenter+(&pbaBits[iAgentID])->vision+1;
			short sYMin = sYCenter-(&pbaBits[iAgentID])->vision-1;
			short sYMax = sYCenter+(&pbaBits[iAgentID])->vision+1;

			// calculate the value of the current square,
			// weighting its sugar and spice by need, metabolism, and occupancy
			int iTemp = sXCenter*GRID_SIZE+sYCenter;
			float fBest = psgSpice[iTemp]*spiceScale/(psgOccupancy[iTemp]+0.01f)
							+ psgSugar[iTemp]*sugarScale/(psgOccupancy[iTemp]+0.01f);

			// search a square neighborhood of dimension 2*vision+3 (from 3x3 to 9x9)
			float fTest = 0.0f;
			iTemp = 0;
			short sXTry = 0;
			short sYTry = 0;
			for (short i = sXMin; i <= sXMax; i++) {
				// wraparound
				sXTry = i;
				if (sXTry < 0) sXTry += GRID_SIZE;
				if (sXTry >= GRID_SIZE) sXTry -= GRID_SIZE;

				for (short j = sYMin; j <= sYMax; j++) {
					// wraparound
					sYTry = j;
					if (sYTry < 0) sYTry += GRID_SIZE;
					if (sYTry >= GRID_SIZE) sYTry -= GRID_SIZE;

					// weight target's sugar and spice by need, metabolism, and occupancy
					iTemp = sXTry*GRID_SIZE+sYTry;
					fTest = psgSpice[iTemp]*spiceScale/(psgOccupancy[iTemp]+1)
									+ psgSugar[iTemp]*sugarScale/(psgOccupancy[iTemp]+1);

					// choose new square if it's better
					if (fTest> fBest) {
						sXStore = sXTry;
						sYStore = sYTry;
						fBest = fTest;
					}
				}
			}
			if (sXStore != sXCenter || sYStore != sYCenter) {

				// if a move is warranted, lock old and new address - if either fails, defer

				// current agent's address in the grid
				int iOldAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];
				iLockedOld = atomicCAS(&(pigLocks[iOldAddy]), 0, 1);

				// agent's new address in the grid
				int iNewAddy = sXStore*GRID_SIZE+sYStore;
				iLockedNew = atomicCAS(&(pigLocks[iNewAddy]), 0, 1);
				// printf("old %d:%d new %d:%d\n",sXCenter,sYCenter,sXStore,sYStore);

				if (iLockedOld == 0 && iLockedNew == 0) {
					iFlag = atomicAdd(piLockSuccesses,1);

					// decrement occupancy at old address
					short sOldOcc = psgOccupancy[iOldAddy]--;
					if (sOldOcc >= 0) {

						// find match starting at end of list
						short k = sOldOcc;
						while (pigResidents[iOldAddy*MAX_OCCUPANCY+k] != iAgentID && k > 0) {k--;} //PROBLEM HERE!!!!

						// remove current id - if it is not at the end, replace it with the one from the end
						if (k != sOldOcc) atomicExch(&(pigResidents[iOldAddy*MAX_OCCUPANCY+k]),
								pigResidents[iOldAddy*MAX_OCCUPANCY+sOldOcc]);
					} else {

						// in case of bugs (i.e. old occupancy was already zero), report problem
						printf ("underflow at x:%d y:%d \n",sXStore,sYStore);
					}
					// increment occupancy at new address
					short sNewOcc = psgOccupancy[iNewAddy]++;

					// make sure new address is not already full before inserting new resident
					if (sNewOcc < MAX_OCCUPANCY) {
						iFlag = atomicExch(&(pigResidents[iNewAddy*MAX_OCCUPANCY+sNewOcc]),iAgentID);
						psaX[iAgentID] = sXStore;
						psaY[iAgentID] = sYStore;
					} else {
						// indicate an occupancy overflow
						printf ("overflow at x:%d y:%d \n",sXStore,sYStore);
						psgOccupancy[iNewAddy] = MAX_OCCUPANCY+1;
					}

				} else {

					// otherwise, add the agent to the deferred queue
					iTemp = atomicAdd(piDeferredQueueSize,1);
					piaDeferredQueue[iTemp]=iAgentID;
				}

				// unlock
				iFlag = atomicExch(&(pigLocks[iNewAddy]),0);
				iFlag = atomicExch(&(pigLocks[iOldAddy]),0);
			}
		}
	}
	return;
}

// this "failsafe" kernel has one thread, for persistent lock failures
// NOTE: NUM_AGENTS is an int, GRID_SIZE is a short
__global__ void best_move_by_traversal_fs(short* psaX, short* psaY, BitWiseType* pbaBits,
		float* pfaSugar, float* pfaSpice, short* psgSugar, short* psgSpice,
		short* psgOccupancy, int* pigResidents, int* piaActiveQueue, const int ciActiveQueueSize)
{
	int iAgentID;
	int iTemp = 0;

	// only the 1,1 block is active
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// iterate through the active queue
		for (int iOffset = 0; iOffset < ciActiveQueueSize; iOffset++) {

			// get the iAgentID from the active agent queue
			iAgentID = piaActiveQueue[iOffset];

			// work with live agents only
			if (psaX[iAgentID] > -1) {

				// begin finding best move, searching around current (center) square
				short sXCenter = psaX[iAgentID];
				short sYCenter = psaY[iAgentID];
				short sXStore = sXCenter;
				short sYStore = sYCenter;

				// scale values by agent's current sugar and spice levels,
				// converting to a duration using metabolic rates
				// add .01 to current values to avoid div by zero
				float sugarScale = 1.0f/(pfaSugar[iAgentID]+0.01f)/((&pbaBits[iAgentID])->metSugar+1);
				float spiceScale = 1.0f/(pfaSpice[iAgentID]+0.01f)/((&pbaBits[iAgentID])->metSpice+1);

				// search limits based on vision
				short sXMin = sXCenter-(&pbaBits[iAgentID])->vision-1;
				short sXMax = sXCenter+(&pbaBits[iAgentID])->vision+1;
				short sYMin = sYCenter-(&pbaBits[iAgentID])->vision-1;
				short sYMax = sYCenter+(&pbaBits[iAgentID])->vision+1;

				// calculate the value of the current square,
				// weighting its sugar and spice by need, metabolism, and occupancy
				int iTemp = sXCenter*GRID_SIZE+sYCenter;
				float fBest = psgSpice[iTemp]*spiceScale/(psgOccupancy[iTemp]+0.01f)
								+ psgSugar[iTemp]*sugarScale/(psgOccupancy[iTemp]+0.01f);

				// search a square neighborhood of dimension 2*vision+3 (from 3x3 to 9x9)
				float fTest = 0.0f;
				iTemp = 0;
				short sXTry = 0;
				short sYTry = 0;
				for (short i = sXMin; i <= sXMax; i++) {
					// wraparound
					sXTry = i;
					if (sXTry < 0) sXTry += GRID_SIZE;
					if (sXTry >= GRID_SIZE) sXTry -= GRID_SIZE;

					for (short j = sYMin; j <= sYMax; j++) {
						// wraparound
						sYTry = j;
						if (sYTry < 0) sYTry += GRID_SIZE;
						if (sYTry >= GRID_SIZE) sYTry -= GRID_SIZE;

						// weight target's sugar and spice by need, metabolism, and occupancy
						iTemp = sXTry*GRID_SIZE+sYTry;
						fTest = psgSpice[iTemp]*spiceScale/(psgOccupancy[iTemp]+1)
										+ psgSugar[iTemp]*sugarScale/(psgOccupancy[iTemp]+1);

						// choose new square if it's better
						if (fTest> fBest) {
							sXStore = sXTry;
							sYStore = sYTry;
							fBest = fTest;
						}
					}
				}
				if (sXStore != sXCenter || sYStore != sYCenter) {

					// if a move is warranted, go, no need to lock

					// get old and new addresses in the grid
					int iOldAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];
					int iNewAddy = sXStore*GRID_SIZE+sYStore;

					// decrement occupancy at old address
					short sOldOcc = psgOccupancy[iOldAddy]--;
					if (sOldOcc >= 0) {

						// find match starting at end of list
						short k = sOldOcc;
						while (pigResidents[iOldAddy*MAX_OCCUPANCY+k] != iAgentID && k > 0) {k--;} //PROBLEM HERE?

						// remove current id - if it is not at the end, replace it with the one from the end
						if (k != sOldOcc) iTemp = atomicExch(&(pigResidents[iOldAddy*MAX_OCCUPANCY+k]),
								pigResidents[iOldAddy*MAX_OCCUPANCY+sOldOcc]);
					} else {

						// in case of bugs (i.e. old occupancy was already zero), report problem
						printf ("underflow at x:%d y:%d \n",sXStore,sYStore);
					}
					// increment occupancy at new address
					short sNewOcc = psgOccupancy[iNewAddy]++;

					// make sure new address is not already full before inserting new resident
					if (sNewOcc < MAX_OCCUPANCY) {
						pigResidents[iNewAddy*MAX_OCCUPANCY+sNewOcc] = iAgentID;
						psaX[iAgentID] = sXStore;
						psaY[iAgentID] = sYStore;
					} else {
						// indicate an occupancy overflow
						printf ("overflow at x:%d y:%d \n",sXStore,sYStore);
						psgOccupancy[iNewAddy] = MAX_OCCUPANCY+1;
					}

				}

			}
		}
	}
	return;
}

int move (short* psaX, short* psaY, BitWiseType* pbaBits, float* pfaSugar, float* pfaSpice, short* psgSugar, short* psgSpice, short* psgOccupancy, 
		int* pigResidents, int* pigLocks, int* piaQueueA, const int iQueueSize, int* piaQueueB, int* piDeferredQueueSize, int* piLockSuccesses)
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
	best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pbaBits,pfaSugar,pfaSpice,psgSugar,psgSpice,
			psgOccupancy,pigResidents,pigLocks,piaQueueA,iQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
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
			best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pbaBits,pfaSugar,pfaSpice,psgSugar,psgSpice,
					psgOccupancy,pigResidents,pigLocks,piaQueueB,ihActiveQueueSize,piaQueueA,piDeferredQueueSize,piLockSuccesses);

		} else {
			CUDA_CALL(cudaMemset(piaQueueB,0xFF,iQueueSize*sizeof(int)));
			best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,pbaBits,pfaSugar,pfaSpice,psgSugar,psgSpice,
					psgOccupancy,pigResidents,pigLocks,piaQueueA,ihActiveQueueSize,piaQueueB,piDeferredQueueSize,piLockSuccesses);
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
			best_move_by_traversal_fs<<<1,1>>>(psaX,psaY,pbaBits,pfaSugar,pfaSpice,psgSugar,psgSpice,
					psgOccupancy,pigResidents,piaQueueB,ihActiveQueueSize);

		} else {
			best_move_by_traversal_fs<<<1,1>>>(psaX,psaY,pbaBits,pfaSugar,pfaSpice,psgSugar,psgSpice,
					psgOccupancy,pigResidents,piaQueueA,ihActiveQueueSize);
		}
		cudaDeviceSynchronize();
	}

	// cleanup
	free(pihLockSuccesses);
	free(pihDeferredQueueSize);
	free(piahTemp);

	return status;
}
