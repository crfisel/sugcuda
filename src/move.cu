#include <cuda.h>
#include <stdio.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "move.h"

// this kernel has one thread per agent, each traversing the local neighborhood prescribed by its vision
// NOTE: NUM_AGENTS is an int, GRID_SIZE is a short
__global__ void best_move_by_traversal(short* psaX, short* psaY, int* piaAgentBits, float* pfaSugar, 
	float* pfaSpice, int* pigGridBits, int* pigResidents, int* piaActiveQueue, const int ciActiveQueueSize, 
	int* piaDeferredQueue, int* piDeferredQueueSize, int* piLockSuccesses, int* piStaticAgents)
{
	GridBitWise gbwBits;
	GridBitWise gbwNewBits;
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
				lockSuccess = lock(iOldAddy,&gbwBits,pigGridBits);
				if (lockSuccess) {
					// if lock succeeded, lock agent's new address in the grid
					iNewAddy = sXStore*GRID_SIZE+sYStore;
					lockSuccess = lockSuccess && lock(iNewAddy,&gbwNewBits,pigGridBits);
					if (lockSuccess) {
						// note that both locks succeeded
						iFlag = atomicAdd(piLockSuccesses,1);

						// valid copies of the grid bits of both squares are in the "gbw" variables
						// before moving the resident, check that old occupancy was positive
						// and that new address is not already full 
						if (gbwBits.asBits.occupancy > 0 && gbwNewBits.asBits.occupancy < MAX_OCCUPANCY) {
							remove_resident(&(gbwBits.asInt),iOldAddy,pigResidents,iAgentID);
							insert_resident(&(gbwNewBits.asInt),iNewAddy,pigResidents,psaX,psaY,sXStore,sYStore,iAgentID);
						} else {
							// indicate an error
							printf("over occ %d to x:%d y:%d or under occ %d from x:%d y:%d agent %d\n",
								gbwNewBits.asBits.occupancy,sXStore,sYStore,gbwBits.asBits.occupancy,sXCenter,sYCenter,iAgentID);
						} 
						// unlock and update global occupancy values
						gbwNewBits.asBits.isLocked = 0;
						iFlag = atomicExch(&(pigGridBits[iNewAddy]),gbwNewBits.asInt);
					}
					// unlock and update global occupancy values
					gbwBits.asBits.isLocked = 0;
					iFlag = atomicExch(&(pigGridBits[iOldAddy]),gbwBits.asInt);	
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
__global__ void best_move_by_traversal_fs(short* psaX, short* psaY, int* piaAgentBits, float* pfaSugar, 
	float* pfaSpice, int* pigGridBits, int* pigResidents, int* piaActiveQueue, const int ciActiveQueueSize, int* piStaticAgents)
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

				if (sXStore == sXCenter || sYStore == sYCenter) {
					piStaticAgents[0]++;
				} else {
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

