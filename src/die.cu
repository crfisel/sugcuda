#include <cuda.h>
#include <stdio.h>
#include <limits.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "move.h"
#include "die.h"

// this kernel has one thread per agent
__global__ void register_deaths(short* psaX, short* psaY, int* piaAgentBits, float* pfaSugar, float* pfaSpice, 
		int* pigGridBits, int* pigResidents, int* piaActiveQueue, const int ciActiveQueueSize, int* piaDeferredQueue, 
		int* piDeferredQueueSize, int* piLockSuccesses)
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

			if ((abwBits.asBits.age > 64+abwBits.asBits.deathAge) || (pfaSpice[iAgentID] < 0.0f) || (pfaSpice[iAgentID] < 0.0f)) {

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
							// find match starting at end of list
							short k = --gbwBitsCopy.asBits.occupancy;

							// remove current id - if not at the end, replace it with the one from the end and store -1 at end
							if (pigResidents[iAddy*MAX_OCCUPANCY+k] == iAgentID) {
								pigResidents[iAddy*MAX_OCCUPANCY+k] = -1;
							} else {
								while (pigResidents[iAddy*MAX_OCCUPANCY+k] != iAgentID && k >= 0) {k--;}
								if (k != gbwBitsCopy.asBits.occupancy) {
									pigResidents[iAddy*MAX_OCCUPANCY+k] = pigResidents[iAddy*MAX_OCCUPANCY+gbwBitsCopy.asBits.occupancy];
									pigResidents[iAddy*MAX_OCCUPANCY+gbwBitsCopy.asBits.occupancy] = -1;
								}
							}
							
							// TODO: INHERITANCE MUST BE HANDLED BEFORE X POSITION INFO IS ERASED
							// mark agent as dead
							psaX[iAgentID] = SHRT_MIN;
							
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
__global__ void register_deaths_fs(short* psaX, short* psaY, int* piaAgentBits, 
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
				if ((abwBits.asBits.age > 64+(abwBits.asBits.deathAge)) || (pfaSpice[iAgentID] < 0.0f) || (pfaSpice[iAgentID] < 0.0f)) {
					
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
						// find match starting at end of list
						short k = --gbwBits.asBits.occupancy;

						// remove current id - if not at the end, replace it with the one from the end and store -1 at end
						if (pigResidents[iAddy*MAX_OCCUPANCY+k] == iAgentID) {
							pigResidents[iAddy*MAX_OCCUPANCY+k] = -1;
						} else {
							while (pigResidents[iAddy*MAX_OCCUPANCY+k] != iAgentID && k >= 0) {k--;}
							if (k != gbwBits.asBits.occupancy) {
								pigResidents[iAddy*MAX_OCCUPANCY+k] = pigResidents[iAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy];
								pigResidents[iAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy] = -1;
							}
						}							

						// TODO: INHERITANCE MUST BE HANDLED BEFORE X POSITION INFO IS ERASED
						// mark agent as dead
						psaX[iAgentID] = SHRT_MIN;
								
						// update global occupancy values
						int iFlag = atomicExch(&(pigGridBits[iAddy]),gbwBits.asInt);
					}
				}
			}
		}
	}
	return;
}

