#include <cuda.h>
#include <stdio.h>
#include <limits.h>
#include "constants.h"
#include "utilities.h"
#include "die.h"

// this kernel has one thread per agent
__global__ void register_deaths(short* psaX, short* psaY, int* piaBits, int* pigBits, int* pigResidents, float* pfaSugar, float* pfaSpice,
		int* piaActiveQueue, int* piaDeferredQueue, const int ciActiveQueueSize, int* piDeferredQueueSize, int* piLockSuccesses)
{
	bool lockSuccess = false;
	int iFlag = 0;
	int iTempAgent;
	int iTempGrid;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		int iAgentID = piaActiveQueue[iOffset];

		// if agent is alive
		if (psaX[iAgentID] > -1) {

			// if the agent is over his death age or has starved to death, register the death
			iTempAgent = piaBits[iAgentID];
			if (((iTempAgent&ageMask)>>ageShift > 64+((iTempAgent&deathAgeMask)>>deathAgeShift)) ||
					(pfaSpice[iAgentID] < 0.0f) || (pfaSpice[iAgentID] < 0.0f)) {

				// current agent's address in the grid
				int iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];

				// lock address to register death - if lock fails, defer
				lockSuccess = lock_location(iAddy,pigBits);
				if (lockSuccess) {
					// note that lock succeeded
					iFlag = atomicAdd(piLockSuccesses,1);

					// before removing the resident, check that old occupancy was positive
					iTempGrid = pigBits[iAddy];
					if (((iTempGrid&occupancyMask)>>occupancyShift) > 0) {
						remove_resident(iAgentID,iAddy,pigBits,pigResidents);
						// TODO: INHERITANCE MUST BE HANDLED BEFORE X POSITION INFO IS ERASED
						// mark agent as dead
						psaX[iAgentID] = SHRT_MIN;
					} else {
						// indicate an error
						printf("under occupancy %d at addy %d agent %d\n",((iTempGrid&occupancyMask)>>occupancyShift),iAddy,iAgentID);
					}
					// unlock square
					unlock_location(iAddy,pigBits);
				} else {
					// if a death occurred, but lock failures prevented registering it, defer
					iFlag = atomicAdd(piDeferredQueueSize,1);
					piaDeferredQueue[iFlag]=iAgentID;
				}
			}
		}
	}
	return;
}

// this "fail-safe" kernel has one thread, for persistent lock failures
__global__ void register_deaths_fs(short* psaX, short* psaY, int* piaBits,
		int* pigBits, int* pigResidents, float* pfaSugar, float* pfaSpice,
		int* piaActiveQueue, const int ciActiveQueueSize)
{
	int iTempAgent;
	int iTempGrid;
	int iAgentID;

	// only the 1,1 block is active
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// iterate through the active queue
		for (int iOffset = 0; iOffset < ciActiveQueueSize; iOffset++) {

			// get the iAgentID from the active agent queue
			iAgentID = piaActiveQueue[iOffset];

			// if agent is alive
			if (psaX[iAgentID] > -1) {

				// if the agent is over his death age or has starved to death, register the death
				iTempAgent = piaBits[iAgentID];
				if (((iTempAgent&ageMask)>>ageShift > 64+((iTempAgent&deathAgeMask)>>deathAgeShift)) ||
						(pfaSpice[iAgentID] < 0.0f) || (pfaSpice[iAgentID] < 0.0f)) {

					// current agent's address in the grid
					int iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];

					// before removing the resident, check that old occupancy was positive
					iTempGrid = pigBits[iAddy];
					if (((iTempGrid&occupancyMask)>>occupancyShift) > 0) {
						remove_resident(iAgentID,iAddy,pigBits,pigResidents);
						// TODO: INHERITANCE MUST BE HANDLED BEFORE X POSITION INFO IS ERASED
						// mark agent as dead
						psaX[iAgentID] = SHRT_MIN;
					} else {
						// indicate an error
						printf("under occupancy %d at addy %d agent %d\n",((iTempGrid&occupancyMask)>>occupancyShift),iAddy,iAgentID);
					}
				}
			}
		}
	}
	return;
}

