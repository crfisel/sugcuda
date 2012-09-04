#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "constants.h"
#include "count.h"
#include "utilities.h"

__global__ void count_occupancy(short* psaX, short* psaY, int* pigBits, int* pigResidents, int* piaActiveQueue,
		int* piaDeferredQueue, const int ciActiveQueueSize, int* piDeferredQueueSize, int* piLockSuccesses)
{
	int iAgentID;
	int iAddy;
	int iTemp;
	bool lockSuccess = false;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		iAgentID = piaActiveQueue[iOffset];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

			// current agent's address in the grid
			iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];

			lockSuccess = lock_location(iAddy,pigBits);
			if (lockSuccess) {
				// at this point the square is locked and a valid copy (indicating locked) is in gbwBits
				iTemp = atomicAdd(piLockSuccesses,1);

				// check for occupancy overflow
				iTemp = pigBits[iAddy];
				if (((iTemp&occupancyMask)>>occupancyShift) < MAX_OCCUPANCY) {
					add_resident(iAgentID,iAddy,pigBits,pigResidents,psaX,psaY);
				} else {
					// indicate an error
					printf("over occupancy %d at addy %d, agent %d\n",
							((iTemp&occupancyMask)>>occupancyShift),iAddy,iAgentID);
				}
				// unlock square
				unlock_location(iAddy,pigBits);
			} else {
				// if lock failed, add the agent to the deferred queue
				iTemp = atomicAdd(piDeferredQueueSize,1);
				piaDeferredQueue[iTemp]=iAgentID;
			}
		}
	}
	return;
}

// this "failsafe" kernel has one thread, for persistent lock failures
__global__ void count_occupancy_fs(short* psaX, short* psaY, int* pigBits, int* pigResidents,
		int* piaActiveQueue, const int ciActiveQueueSize)
{
	int iAgentID;
	int iAddy;
	int iTemp;

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

				iTemp = pigBits[iAddy];

				// no locks necessary, so increment square occupancy
				// but check for occupancy overflow first
				if (((iTemp&occupancyMask)>>occupancyShift) < MAX_OCCUPANCY) {
					add_resident(iAgentID,iAddy,pigBits,pigResidents,psaX,psaY);
				} else {
					// indicate an error
					printf("over occupancy %d at addy %d, agent %d\n",((iTemp&occupancyMask)>>occupancyShift),iAddy,iAgentID);
				}
			}
		}
	}
	return;
}

