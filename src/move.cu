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
			short sXStore = sXCenter;
			short sYStore = sYCenter;
			
			// reinterpret agent bits 
			AgentBitWise abwBits;
			abwBits.asInt = piaAgentBits[iAgentID];
			
			// scale values by agent's current sugar and spice levels,
			// converting to a duration using metabolic rates
			// add .01 to current values to avoid div by zero
			float sugarScale = 1.0f/(pfaSugar[iAgentID]+0.01f)/(abwBits.asBits.metSugar+1);
			float spiceScale = 1.0f/(pfaSpice[iAgentID]+0.01f)/(abwBits.asBits.metSpice+1);

			// search limits based on vision
			short sXMin = sXCenter-abwBits.asBits.vision-1;
			short sXMax = sXCenter+abwBits.asBits.vision+1;
			short sYMin = sYCenter-abwBits.asBits.vision-1;
			short sYMax = sYCenter+abwBits.asBits.vision+1;

			// calculate the value of the current square,
			// weighting its sugar and spice by need, metabolism, and occupancy
			int iTemp = sXCenter*GRID_SIZE+sYCenter;
			// reinterpret grid bits
			GridBitWise gbwBits;
			gbwBits.asInt = pigGridBits[iTemp];
			float fBest = gbwBits.asBits.spice*spiceScale/(gbwBits.asBits.occupancy+0.01f)
				+ gbwBits.asBits.sugar*sugarScale/(gbwBits.asBits.occupancy+0.01f);

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
					gbwBits.asInt = pigGridBits[iTemp];
					fTest = gbwBits.asBits.spice*spiceScale/(gbwBits.asBits.occupancy+1)
						+ gbwBits.asBits.sugar*sugarScale/(gbwBits.asBits.occupancy+1);

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
				// agent's current address in the grid
				int iOldAddy = sXCenter*GRID_SIZE+sYCenter;
				// unpack grid bits
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
									// find match starting at end of list
									short k = --gbwBitsCopy.asBits.occupancy;

									// remove current id - if not at the end, replace it with the one from the end and store -1 at end
									if (pigResidents[iOldAddy*MAX_OCCUPANCY+k] == iAgentID) {
										pigResidents[iOldAddy*MAX_OCCUPANCY+k] = -1;
									} else {
										while (pigResidents[iOldAddy*MAX_OCCUPANCY+k] != iAgentID && k >= 0) {k--;}
										if (k != gbwBitsCopy.asBits.occupancy) {
											pigResidents[iOldAddy*MAX_OCCUPANCY+k] = 
												pigResidents[iOldAddy*MAX_OCCUPANCY+gbwBitsCopy.asBits.occupancy];
											pigResidents[iOldAddy*MAX_OCCUPANCY+gbwBitsCopy.asBits.occupancy] = -1;
										}
									}
									// make sure we are replacing an "empty" placemarker
									if (pigResidents[iNewAddy*MAX_OCCUPANCY+gbwNewBitsCopy.asBits.occupancy] == -1) {
										psaX[iAgentID] = sXStore;
										psaY[iAgentID] = sYStore;
										pigResidents[iNewAddy*MAX_OCCUPANCY+gbwNewBitsCopy.asBits.occupancy] = iAgentID;
										// increment occupancy at new address
										gbwNewBitsCopy.asBits.occupancy++;
									} else {
										//otherwise notify about the error
										printf ("agent replaced %d at x:%d y:%d \n",
											pigResidents[iNewAddy*MAX_OCCUPANCY+gbwNewBitsCopy.asBits.occupancy],
											sXStore,sYStore);
									}
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
				short sXStore = sXCenter;
				short sYStore = sYCenter;

				// reinterpret piaAgentBits[iAgentID] bitwise
				AgentBitWise abwBits;
				abwBits.asInt = piaAgentBits[iAgentID];

				// scale values by agent's current sugar and spice levels,
				// converting to a duration using metabolic rates
				// add .01 to current values to avoid div by zero
				float sugarScale = 1.0f/(pfaSugar[iAgentID]+0.01f)/(abwBits.asBits.metSugar+1);
				float spiceScale = 1.0f/(pfaSpice[iAgentID]+0.01f)/(abwBits.asBits.metSpice+1);

				// search limits based on vision
				short sXMin = sXCenter-abwBits.asBits.vision-1;
				short sXMax = sXCenter+abwBits.asBits.vision+1;
				short sYMin = sYCenter-abwBits.asBits.vision-1;
				short sYMax = sYCenter+abwBits.asBits.vision+1;

				// calculate the value of the current square,
				// weighting its sugar and spice by need, metabolism, and occupancy
				int iTemp = sXCenter*GRID_SIZE+sYCenter;
				GridBitWise gbwBits;
				gbwBits.asInt = pigGridBits[iTemp];
				float fBest = gbwBits.asBits.spice*spiceScale/(gbwBits.asBits.occupancy+0.01f)
					+ gbwBits.asBits.sugar*sugarScale/(gbwBits.asBits.occupancy+0.01f);

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
						GridBitWise gbwBits;
						gbwBits.asInt = pigGridBits[iTemp];
						fTest = gbwBits.asBits.spice*spiceScale/(gbwBits.asBits.occupancy+1)
							+ gbwBits.asBits.sugar*sugarScale/(gbwBits.asBits.occupancy+1);

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
					int iOldAddy = sXCenter*GRID_SIZE+sYCenter;
					int iNewAddy = sXStore*GRID_SIZE+sYStore;

					// unpack grid bits
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

						// find match starting at end of list
						short k = --gbwBits.asBits.occupancy;

						// remove current id - if not at the end, replace it with the one from the end and store -1 at end
						if (pigResidents[iOldAddy*MAX_OCCUPANCY+k] == iAgentID) {
							pigResidents[iOldAddy*MAX_OCCUPANCY+k] = -1;
						} else {
							while (pigResidents[iOldAddy*MAX_OCCUPANCY+k] != iAgentID && k >= 0) {k--;}
							if (k != gbwBits.asBits.occupancy) {
								pigResidents[iOldAddy*MAX_OCCUPANCY+k] = pigResidents[iOldAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy];
								pigResidents[iOldAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy] = -1;
							}
						}
						
						// make sure we are replacing an "empty" placemarker
						if (pigResidents[iNewAddy*MAX_OCCUPANCY+gbwNewBits.asBits.occupancy] == -1) {
							psaX[iAgentID] = sXStore;
							psaY[iAgentID] = sYStore;
							pigResidents[iNewAddy*MAX_OCCUPANCY+gbwNewBits.asBits.occupancy] = iAgentID;

							// increment occupancy at new address
							gbwNewBits.asBits.occupancy++;
						} else {
							//otherwise notify about the error
							printf ("agent replaced %d at x:%d y:%d \n",pigResidents[iNewAddy*MAX_OCCUPANCY+gbwNewBits.asBits.occupancy],sXStore,sYStore);
						} 
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
