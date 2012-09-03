#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "count.h"

__global__ void count_occupancy(short* psaX, short* psaY, int* pigGridBits, int* pigResidents, int* piaActiveQueue, 
	const int ciActiveQueueSize, int* piaDeferredQueue, int* piDeferredQueueSize, int* piLockSuccesses)
{
	int iAgentID;
	int iAddy;
	int iTemp;
	bool lockFailed = false;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		iAgentID = piaActiveQueue[iOffset];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

			// current agent's address in the grid
			iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];

			// unpack grid bits
			GridBitWise gbwBits;
			gbwBits.asInt = pigGridBits[iAddy];
			
			// test if square is locked
			if (gbwBits.asBits.isLocked != 0) {

				// if so, lock failed
				lockFailed = true;
			} else {

				// if so, make a copy, but indicating locked
				GridBitWise gbwBitsCopy = gbwBits;
				gbwBitsCopy.asBits.isLocked = 1;

				// now lock the current address if possible
				iTemp = atomicCAS(&(pigGridBits[iAddy]), gbwBits.asInt, gbwBitsCopy.asInt);

				// test lock
				if (iTemp != gbwBits.asInt) {
					lockFailed = true;

				} else {

					// at this point the square is locked and a valid copy (indicating locked) is in gbwBitsCopy
					iTemp = atomicAdd(piLockSuccesses,1);
					
					// check for overflow
					if (gbwBitsCopy.asBits.occupancy < MAX_OCCUPANCY) {

						// insert the resident at the next position in the residents list
						pigResidents[iAddy*MAX_OCCUPANCY+gbwBitsCopy.asBits.occupancy] = iAgentID;

						// now increment square occupancy
						gbwBitsCopy.asBits.occupancy++;
			
						// unlock the square and apply the change to occupancy
						gbwBitsCopy.asBits.isLocked = 0;
						if (gbwBitsCopy.asBits.occupancy == 0) {
							printf("agent %d x %d y %d\n",iAgentID,psaX[iAgentID],psaY[iAgentID]);
						}
						iTemp = atomicExch(&(pigGridBits[iAddy]),gbwBitsCopy.asInt);

					} else {
						
						// unlock with no changes					
						iTemp = atomicExch(&(pigGridBits[iAddy]),gbwBits.asInt);

						// indicate an occupancy overflow
						printf ("overflow occupancy %d at x:%d y:%d \n",gbwBitsCopy.asBits.occupancy,psaX[iAgentID], psaY[iAgentID]);
					}
				}
			}
			if (lockFailed) {
				// if lock failed or agent was already locked, add the agent to the deferred queue
				iTemp = atomicAdd(piDeferredQueueSize,1);
				piaDeferredQueue[iTemp]=iAgentID;
 			}
		}
	}
	return;
}

// this "failsafe" kernel has one thread, for persistent lock failures
__global__ void count_occupancy_fs(short* psaX, short* psaY, int* pigGridBits, int* pigResidents,
		int* piaActiveQueue, const int ciActiveQueueSize)
{
	int iAgentID;
	int iAddy;

	// only the 1,1 block is active
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// iterate through the active queue
		for (int iOffset = 0; iOffset < ciActiveQueueSize; iOffset++) {
			// get agent id
			iAgentID = piaActiveQueue[iOffset];

			// if the agent is alive
			if (psaX[iAgentID] > -1) {
				// current agent's address in the grid
				iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];

				// unpack grid bits
				GridBitWise gbwBits;
				gbwBits.asInt = pigGridBits[iAddy];

				// no locks necessary, so increment square occupancy
				
				// check for overflow
				if (gbwBits.asBits.occupancy < MAX_OCCUPANCY) {
				
					// insert the resident at the next position in the residents list
					pigResidents[iAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy] = iAgentID;
					
					// update occupancy in pigGridBits - no need for locks or atomics
					gbwBits.asBits.occupancy++;
					if (gbwBits.asBits.occupancy == 0) {
							printf("agent %d x %d y %d\n",iAgentID,psaX[iAgentID],psaY[iAgentID]);
						}
					pigGridBits[iAddy] = gbwBits.asInt;
					
				} else {
					// indicate an occupancy overflow, but otherwise do nothing
					printf ("overflow occupancy %d at x:%d y:%d \n",gbwBits.asBits.occupancy, psaX[iAgentID], psaY[iAgentID]);
				}
			}
		}
	}
	return;
}

