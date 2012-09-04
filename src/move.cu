#include <cuda.h>
#include <stdio.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "move.h"


// this kernel has one thread per agent, each traversing the local neighborhood prescribed by its vision
// NOTE: NUM_AGENTS is an int, GRID_SIZE is a short
__global__ void find_best_move_by_traversal(short* psaX, short* psaY, short* psaXNew, short* psaYNew, int* piaAgentBits, 
	float* pfaSugar, float* pfaSpice, int* pigGridBits, int* piaActiveQueue, const int ciActiveQueueSize)
{
	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		int iAgentID = piaActiveQueue[iOffset];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

			// begin finding best move, searching around current (center) square
			short sXStore = psaX[iAgentID];
			short sYStore = psaY[iAgentID];
			
			// reinterpret agent bits 
			AgentBitWise abwBits;
			abwBits.asInt = piaAgentBits[iAgentID];
			
			// scale values by agent's current sugar and spice levels,
			// converting to a duration using metabolic rates
			// add .01 to current values to avoid div by zero
			float sugarScale = 1.0f/(pfaSugar[iAgentID]+0.01f)/(abwBits.asBits.metSugar+1);
			float spiceScale = 1.0f/(pfaSpice[iAgentID]+0.01f)/(abwBits.asBits.metSpice+1);

			// calculate the value of the current square,
			// weighting its sugar and spice by need, metabolism, and occupancy
			int iTemp = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];

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
			for (short i = psaX[iAgentID]-abwBits.asBits.vision-1; i <= psaX[iAgentID]+abwBits.asBits.vision+1; i++) {
				// wraparound
				sXTry = i;
				if (sXTry < 0) sXTry += GRID_SIZE;
				if (sXTry >= GRID_SIZE) sXTry -= GRID_SIZE;

				for (short j = psaY[iAgentID]-abwBits.asBits.vision-1; j <= psaY[iAgentID]+abwBits.asBits.vision+1; j++) {
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

			if (sXStore != psaX[iAgentID] || sYStore != psaY[iAgentID]) {
				psaXNew[iAgentID] = sXStore;
				psaYNew[iAgentID] = sYStore;
			}
		}
	}
}

__global__ void do_move(short* psaX, short* psaY, short* psaXNew, short* psaYNew, int* pigGridBits, int* pigResidents, 
	int* piaActiveQueue, const int ciActiveQueueSize, int* piaDeferredQueue, int* piDeferredQueueSize, int* piLockSuccesses)
{
	int iFlag = 0;
	bool lockFailed = false;
	GridBitWise gbwBits;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		int iAgentID = piaActiveQueue[iOffset];

		// if the agent is alive
		if (psaX[iAgentID] > -1 &&
			// and a move is warranted
			psaXNew[iAgentID] != psaX[iAgentID] || psaYNew[iAgentID] != psaY[iAgentID]) {

			// agent's current address in the grid
			int iOldAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];

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
				int iLocked = atomicCAS(&(pigGridBits[iOldAddy]),gbwBits.asInt,gbwBitsCopy.asInt);
				// test if the lock failed
				if (iLocked != gbwBits.asInt) {
					lockFailed = true;
				} else {
					// at this point, old square is locked and a valid copy of its bits are in gbwBitsCopy (because locked)
					// agent's new address in the grid
					int iNewAddy = psaXNew[iAgentID]*GRID_SIZE+psaYNew[iAgentID];
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
						iLocked = atomicCAS(&(pigGridBits[iNewAddy]),gbwNewBits.asInt,gbwNewBitsCopy.asInt);

						// test if the lock failed
						if (iLocked != gbwNewBits.asInt) {
							lockFailed = true;
							// unlock old square by replacing the old (unlocked) bits
							iFlag = atomicExch(&(pigGridBits[iOldAddy]),gbwBits.asInt);
						} else {
							// at this point the squares are locked and valid copies are in the "copy" variables
							iFlag = atomicAdd(piLockSuccesses,1);

							// before inserting new resident, check for nonzero old occupancy (negatives forbidden by unsigned short declaration)
							// and make sure new address is not already full 
							if (gbwBitsCopy.asBits.occupancy <= 0 || gbwNewBitsCopy.asBits.occupancy >= MAX_OCCUPANCY) {

								// unlock with no changes
								iFlag = atomicExch(&(pigGridBits[iNewAddy]),gbwNewBits.asInt);
								iFlag = atomicExch(&(pigGridBits[iOldAddy]),gbwBits.asInt);
									
								// indicate an error
								printf("over occ %d at x:%d y:%d or under occ %d at x:%d y:%d agent %d\n",
									gbwNewBitsCopy.asBits.occupancy,psaXNew[iAgentID],psaYNew[iAgentID],
									gbwBitsCopy.asBits.occupancy,psaX[iAgentID],psaY[iAgentID],iAgentID);
							} else {
								remove_resident(&(gbwBitsCopy.asInt),iOldAddy,pigResidents,iAgentID);
								insert_resident(&(gbwNewBitsCopy.asInt),iNewAddy,pigResidents,psaX,psaY,psaXNew[iAgentID],psaYNew[iAgentID],iAgentID);
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
	return;
}

// this "failsafe" kernel has one thread, for persistent lock failures
// NOTE: NUM_AGENTS is an int, GRID_SIZE is a short
__global__ void best_move_by_traversal_fs(short* psaX, short* psaY, int* piaAgentBits, float* pfaSugar, 
	float* pfaSpice, int* pigGridBits, int* pigResidents, int* piaActiveQueue, const int ciActiveQueueSize)
{
	GridBitWise gbwBits;
	int iAgentID;

	// only the 1,1 block is active
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// iterate through the active queue
		for (int iOffset = 0; iOffset < ciActiveQueueSize; iOffset++) {

			// get the iAgentID from the active agent queue
			iAgentID = piaActiveQueue[iOffset];

			// if the agent is alive
			if (psaX[iAgentID] > -1) {

#include "traversal_routine.cu"

				if (sXStore != psaX[iAgentID] || sYStore != psaY[iAgentID]) {

					// if a move is warranted, go, no need to lock
					// get old and new addresses in the grid
					int iOldAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];
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
							gbwNewBits.asBits.occupancy,sXStore,sYStore,gbwBits.asBits.occupancy,psaX[iAgentID],psaY[iAgentID],iAgentID);

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

