#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "count.h"
#include "move.h"

__global__ void count_occupancy(short* psaX, short* psaY, int* pigGridBits, int* pigResidents, int* piaActiveQueue, 
		const int ciActiveQueueSize, int* piaDeferredQueue, int* piDeferredQueueSize, int* piLockSuccesses)
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

			// unpack grid bits
			GridBitWise gbwBits;
			lockSuccess = lock(iAddy,&gbwBits,pigGridBits);
			if (lockSuccess) {
				// at this point the square is locked and a valid copy (indicating locked) is in gbwBits
				iTemp = atomicAdd(piLockSuccesses,1);

				// check for occupancy overflow
				if (gbwBits.asBits.occupancy < MAX_OCCUPANCY) {

					insert_resident(&(gbwBits.asInt),iAddy,pigResidents,psaX,psaY,psaX[iAgentID],psaY[iAgentID],iAgentID);
				} else {
					// indicate an error
					printf("occupancy overflow %d to x:%d y:%d agent %d\n",
							gbwBits.asBits.occupancy,psaX[iAgentID],psaY[iAgentID],iAgentID);
				}
				// unlock and update global occupancy values
				gbwBits.asBits.isLocked = 0;
				iTemp = atomicExch(&(pigGridBits[iAddy]),gbwBits.asInt);
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
				// check for occupancy overflow
				if (gbwBits.asBits.occupancy < MAX_OCCUPANCY) {

					insert_resident(&(gbwBits.asInt),iAddy,pigResidents,psaX,psaY,psaX[iAgentID],psaY[iAgentID],iAgentID);
				} else {
					// indicate an error
					printf("occupancy overflow %d to x:%d y:%d agent %d\n",
							gbwBits.asBits.occupancy,psaX[iAgentID],psaY[iAgentID],iAgentID);
				}
					pigGridBits[iAddy] = gbwBits.asInt;
			}
		}
	}
	return;
}

