#include <cuda.h>
#include <stdio.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "move.h"

// this kernel has one thread per agent, each traversing the local neighborhood prescribed by its vision
// NOTE: NUM_AGENTS is an int, GRID_SIZE is a short
__global__ void best_move_by_traversal(short* psaX, short* psaY, int* piaAgentBits, float* pfaSugar, 
	float* pfaSpice, int* pigGridBits, int* pigResidents, int* piaActiveQueue, const int ciActiveQueueSize, 
	int* piaDeferredQueue, int* piDeferredQueueSize, int* piLockSuccesses)
{
	int iFlag = 0;
	bool lockFailed = false;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		int iAgentID = piaActiveQueue[iOffset];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

			// begin finding best move, searching around current (center) square
			short sXCenter = psaX[iAgentID];
			short sYCenter = psaY[iAgentID];
			short2 s2Temp = find_best_move_by_traversal(sXCenter,sYCenter,piaAgentBits,pfaSugar,pfaSpice,pigGridBits,iAgentID);
			short sXStore = s2Temp.x;
			short sYStore = s2Temp.y;

			// if a move is warranted, lock old and new address - if either fails, defer
			if (sXStore != sXCenter || sYStore != sYCenter) {

				// agent's current address in the grid
				int iOldAddy = sXCenter*GRID_SIZE+sYCenter;
				// unpack grid bits
				GridBitWise gbwBits;
				gbwBits.asInt = pigGridBits[iOldAddy];
			
				// test if old square is locked
				if (gbwBits.asBits.isLocked != 0) {
					// if so, lock failed
					lockFailed = true;
				} else {
					// if not, make a copy, but indicating locked
					GridBitWise gbwBitsCopy = gbwBits;
					gbwBitsCopy.asBits.isLocked = 1;

					// now lock the current address if possible
					int iLockedOld = atomicCAS(&(pigGridBits[iOldAddy]),gbwBits.asInt,gbwBitsCopy.asInt);
					// test if the lock failed
					if (iLockedOld != gbwBits.asInt) {
						lockFailed = true;
					} else {
						// at this point, old square is locked and a valid copy of its bits are in gbwBitsCopy (because locked)
						// agent's new address in the grid
						int iNewAddy = sXStore*GRID_SIZE+sYStore;
						// unpack grid bits
						GridBitWise gbwNewBits;
						gbwNewBits.asInt = pigGridBits[iNewAddy];
			
						// test if new square is locked
						if (gbwNewBits.asBits.isLocked != 0) {
							// if so, lock failed
							lockFailed = true;
							// unlock old square by replacing the old (unlocked) bits
							iFlag = atomicExch(&(pigGridBits[iOldAddy]),gbwBits.asInt);

						} else {
							// if not, make a copy, but indicating locked
							GridBitWise gbwNewBitsCopy = gbwNewBits;
							gbwNewBitsCopy.asBits.isLocked = 1;

							// now lock the new address if possible
							int iLockedNew = atomicCAS(&(pigGridBits[iNewAddy]),gbwNewBits.asInt,gbwNewBitsCopy.asInt);

							// test if the lock failed
							if (iLockedNew != gbwNewBits.asInt) {
								lockFailed = true;
								// unlock old square by replacing the old (unlocked) bits
								iFlag = atomicExch(&(pigGridBits[iOldAddy]),gbwBits.asInt);
							} else {
								// at this point the squares are locked and valid copies are in the "copy" variables
								iFlag = atomicAdd(piLockSuccesses,1);

								// before inserting new resident, check for nonzero old occupancy (negatives forbidden by unsigned short declaration)
								// and make sure new address is not already full 
								if (gbwBitsCopy.asBits.occupancy <= 0 || 
									gbwNewBitsCopy.asBits.occupancy >= MAX_OCCUPANCY) {
									
									// unlock with no changes
									iFlag = atomicExch(&(pigGridBits[iNewAddy]),gbwNewBits.asInt);
									iFlag = atomicExch(&(pigGridBits[iOldAddy]),gbwBits.asInt);
									
									// indicate an error
									printf("over occ %d at x:%d y:%d or under occ %d at x:%d y:%d agent %d\n",
										gbwNewBitsCopy.asBits.occupancy,sXStore,sYStore,gbwBitsCopy.asBits.occupancy,sXCenter,sYCenter,iAgentID);
								} else {
									remove_resident(&(gbwBitsCopy.asInt),iOldAddy,pigResidents,iAgentID);
									insert_resident(&(gbwNewBitsCopy.asInt),iNewAddy,pigResidents,psaX,psaY,sXStore,sYStore,iAgentID);
								} 
								// unlock and update global occupancy values
								gbwNewBitsCopy.asBits.isLocked = 0;
								iFlag = atomicExch(&(pigGridBits[iNewAddy]),gbwNewBitsCopy.asInt);
								gbwBitsCopy.asBits.isLocked = 0;
								iFlag = atomicExch(&(pigGridBits[iOldAddy]),gbwBitsCopy.asInt);
							}
						}
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
// NOTE: NUM_AGENTS is an int, GRID_SIZE is a short
__global__ void best_move_by_traversal_fs(short* psaX, short* psaY, int* piaAgentBits, float* pfaSugar, 
	float* pfaSpice, int* pigGridBits, int* pigResidents, int* piaActiveQueue, const int ciActiveQueueSize)
{

	int iAgentID;

	// only the 1,1 block is active
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// iterate through the active queue
		for (int iOffset = 0; iOffset < ciActiveQueueSize; iOffset++) {

			// get the iAgentID from the active agent queue
			iAgentID = piaActiveQueue[iOffset];

			// if the agent is alive
			if (psaX[iAgentID] > -1) {

				// begin finding best move, searching around current (center) square
				short sXCenter = psaX[iAgentID];
				short sYCenter = psaY[iAgentID];
				short2 s2Temp = find_best_move_by_traversal(sXCenter,sYCenter,piaAgentBits,pfaSugar,pfaSpice,pigGridBits,iAgentID);
				short sXStore = s2Temp.x;
				short sYStore = s2Temp.y;
				if (sXStore != sXCenter || sYStore != sYCenter) {

					// if a move is warranted, go, no need to lock
					// get old and new addresses in the grid
					int iOldAddy = sXCenter*GRID_SIZE+sYCenter;
					int iNewAddy = sXStore*GRID_SIZE+sYStore;

					// unpack grid bits
					GridBitWise gbwBits;
					gbwBits.asInt = pigGridBits[iOldAddy];
					GridBitWise gbwNewBits;
					gbwNewBits.asInt = pigGridBits[iNewAddy];

					// before inserting new resident, check for nonzero old occupancy (negatives forbidden by unsigned short declaration)
					// and make sure new address is not already full 
					if (gbwBits.asBits.occupancy <= 0 || 
						gbwNewBits.asBits.occupancy >= MAX_OCCUPANCY) {
									
						// indicate an error
						printf("over occ %d at x:%d y:%d or under occ %d at x:%d y:%d agent %d\n",
							gbwNewBits.asBits.occupancy,sXStore,sYStore,gbwBits.asBits.occupancy,sXCenter,sYCenter,iAgentID);

					} else {			

						remove_resident(&(gbwBits.asInt),iOldAddy,pigResidents,iAgentID);
						insert_resident(&(gbwNewBits.asInt),iNewAddy,pigResidents,psaX,psaY,sXStore,sYStore,iAgentID);

						// update global occupancy values
						int iFlag = atomicExch(&(pigGridBits[iNewAddy]),gbwNewBits.asInt);
						iFlag = atomicExch(&(pigGridBits[iOldAddy]),gbwBits.asInt);
					}
				}
			}
		}
	}
	return;
}

int move (short* psaX, short* psaY, int* piaAgentBits, float* pfaSugar, float* pfaSpice, int* pigGridBits, int* pigResidents, 
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

	// find best move for agents
	int hiNumBlocks = (iQueueSize+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
	best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
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
	while (pihDeferredQueueSize[0] > 10 && pihDeferredQueueSize[0] < ihActiveQueueSize) {
		ihActiveQueueSize = pihDeferredQueueSize[0];
		hiNumBlocks = (ihActiveQueueSize+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
		CUDA_CALL(cudaMemset(piDeferredQueueSize,0,sizeof(int)));
		CUDA_CALL(cudaMemset(piLockSuccesses,0,sizeof(int)));
		if (hQueue) {
			CUDA_CALL(cudaMemset(piaQueueA,0xFF,iQueueSize*sizeof(int)));
			best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
				pigGridBits,pigResidents,piaQueueB,ihActiveQueueSize,piaQueueA,piDeferredQueueSize,piLockSuccesses);

		} else {
			CUDA_CALL(cudaMemset(piaQueueB,0xFF,iQueueSize*sizeof(int)));
			best_move_by_traversal<<<hiNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,
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
			best_move_by_traversal_fs<<<1,1>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pigGridBits,
				pigResidents,piaQueueB,ihActiveQueueSize);

		} else {
			best_move_by_traversal_fs<<<1,1>>>(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pigGridBits,
				pigResidents,piaQueueA,ihActiveQueueSize);
		}
		cudaDeviceSynchronize();
	}

	// cleanup
	free(pihLockSuccesses);
	free(pihDeferredQueueSize);
	free(piahTemp);

	return status;
}
