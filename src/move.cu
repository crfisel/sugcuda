#include <cuda.h>
#include <stdio.h>
#include "constants.h"
#include "utilities.h"
#include "move.h"

// this kernel has one thread per agent, each traversing the local neighborhood prescribed by its vision
// NOTE: NUM_AGENTS is an int, GRID_SIZE is a short
__global__ void best_move_by_traversal(short* psaX, short* psaY, int* piaBits, int* pigBits, int* pigResidents, float* pfaSugar, float* pfaSpice,
		int* piaActiveQueue, int* piaDeferredQueue, const int ciActiveQueueSize, int* piDeferredQueueSize, int* piLockSuccesses, int* piStaticAgents)
{
	int iFlag = 0;
	bool lockSuccess = false;
	int iOldAddy;
	int iNewAddy;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		int iAgentID = piaActiveQueue[iOffset];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

#include "traversal_routine.cu"

			if (sXStore == sXCenter && sYStore == sYCenter) {
				iFlag = atomicAdd(piStaticAgents,1);
			} else {
				// if a move is warranted, lock old and new address
				iOldAddy = sXCenter*GRID_SIZE+sYCenter;
				// lock agent's current address in the grid
				lockSuccess = lock_location(iOldAddy,pigBits);
				if (lockSuccess) {
					// if lock succeeded, lock agent's new address in the grid
					iNewAddy = sXStore*GRID_SIZE+sYStore;
					lockSuccess = lockSuccess && lock_location(iNewAddy,pigBits);
					if (lockSuccess) {
						// note that both locks succeeded
						iFlag = atomicAdd(piLockSuccesses,1);

						// before moving the resident, check that old occupancy was positive
						// and that new address is not already full 
						int iTempOld = pigBits[iOldAddy];
						int iTempNew = pigBits[iNewAddy];
						if (((iTempOld&occupancyMask)>>occupancyShift) > 0 && ((iTempNew&occupancyMask)>>occupancyShift) < MAX_OCCUPANCY) {
							remove_resident(iAgentID,iOldAddy,pigBits,pigResidents);
							add_resident(iAgentID,iNewAddy,pigBits,pigResidents,psaX,psaY);
						} else {
							// indicate an error
							printf("over occ %d to addy %d or under occ %d from addy %d agent %d\n",
									((iTempNew&occupancyMask)>>occupancyShift),iNewAddy,((iTempOld&occupancyMask)>>occupancyShift),iOldAddy,iAgentID);
						} 
						// unlock and update global occupancy values
						unlock_location(iNewAddy,pigBits);
					}
					// unlock old address
					unlock_location(iOldAddy,pigBits);
				}
				// if a move was warranted, but lock failures prevented it, defer
				if (!lockSuccess) {
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
__global__ void best_move_by_traversal_fs(short* psaX, short* psaY, int* piaBits, int* pigBits, int* pigResidents,
		float* pfaSugar, float* pfaSpice, int* piaActiveQueue, const int ciActiveQueueSize, int* piStaticAgents)
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

#include "traversal_routine.cu"

				if (sXStore == sXCenter || sYStore == sYCenter) {
					piStaticAgents[0]++;
				} else {
					// if a move is warranted, go, no need to lock
					// get old and new addresses in the grid
					int iOldAddy = sXCenter*GRID_SIZE+sYCenter;
					int iNewAddy = sXStore*GRID_SIZE+sYStore;

					// before moving the resident, check that old occupancy was positive
					// and that new address is not already full
					int iTempOld = pigBits[iOldAddy];
					int iTempNew = pigBits[iNewAddy];
					if (((iTempOld&occupancyMask)>>occupancyShift) > 0 && ((iTempNew&occupancyMask)>>occupancyShift) < MAX_OCCUPANCY) {
						remove_resident(iAgentID,iOldAddy,pigBits,pigResidents);
						add_resident(iAgentID,iNewAddy,pigBits,pigResidents,psaX,psaY);
					} else {
						// indicate an error
						printf("over occ %d to addy %d or under occ %d from addy %d agent %d\n",
								((iTempNew&occupancyMask)>>occupancyShift),iNewAddy,((iTempOld&occupancyMask)>>occupancyShift),iOldAddy,iAgentID);
					}
				}
			}
		}
	}
	return;
}

