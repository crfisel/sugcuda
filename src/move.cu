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

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		int iAgentID = piaActiveQueue[iOffset];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

#include "traversal_routine.cu"

			// if a move is warranted, lock old and new address - if either fails, defer
			if (sXStore != sXCenter || sYStore != sYCenter) {

				// agent's old and new addresses in the grid
				int iOldAddy = sXCenter*GRID_SIZE+sYCenter;
				int iNewAddy = sXStore*GRID_SIZE+sYStore;
				
				// setup grid bit unpacking
				GridBitWise gbwOldBits;
				GridBitWise gbwNewBits;
				
				// lock old and new addresses
				bool lockedOld = lock(iOldAddy,&gbwOldBits,pigGridBits);
				bool lockedNew = lock(iNewAddy,&gbwNewBits,pigGridBits);
				
				// test whether both locks were successful
				if (lockedOld && lockedNew) {
					// if so, the squares are locked and valid copies are in the gbw variables
					iFlag = atomicAdd(piLockSuccesses,1);

					// before inserting new resident, check for positive old occupancy
					// and make sure new address is not already full 
					if (gbwOldBits.asBits.occupancy > 0 || gbwNewBits.asBits.occupancy < MAX_OCCUPANCY) {
						// move resident to new address
						remove_resident(&gbwOldBits,iOldAddy,pigResidents,iAgentID);
						insert_resident(&gbwNewBits,iNewAddy,pigResidents,psaX,psaY,sXStore,sYStore,iAgentID);

						// and set grid bits to unlock
						gbwOldBits.asBits.isLocked = 0;
						gbwNewBits.asBits.isLocked = 0;
					} else {
						// indicate an error
						printf("over occ %d at x:%d y:%d or under occ %d at x:%d y:%d agent %d\n",
							gbwNewBits.asBits.occupancy,sXStore,sYStore,gbwOldBits.asBits.occupancy,sXCenter,sYCenter,iAgentID);
					}
				} else {
					// if either lock failed, add the agent to the deferred queue
					iFlag = atomicAdd(piDeferredQueueSize,1);
					piaDeferredQueue[iFlag]=iAgentID;
				}
				// update grid bit values
				iFlag = atomicExch(&(pigGridBits[iNewAddy]),gbwNewBits.asInt);
				iFlag = atomicExch(&(pigGridBits[iOldAddy]),gbwOldBits.asInt);
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

#include "traversal_routine.cu"

				if (sXStore != sXCenter || sYStore != sYCenter) {

					// if a move is warranted, go, no need to lock
					// get old and new addresses in the grid
					int iOldAddy = sXCenter*GRID_SIZE+sYCenter;
					int iNewAddy = sXStore*GRID_SIZE+sYStore;

					// unpack grid bits
					GridBitWise gbwOldBits;
					gbwOldBits.asInt = pigGridBits[iOldAddy];
					GridBitWise gbwNewBits;
					gbwNewBits.asInt = pigGridBits[iNewAddy];

					// before inserting new resident, check for nonzero old occupancy (negatives forbidden by unsigned short declaration)
					// and make sure new address is not already full 
					if (gbwOldBits.asBits.occupancy <= 0 || gbwNewBits.asBits.occupancy >= MAX_OCCUPANCY) {
									
						// indicate an error
						printf("over occ %d at x:%d y:%d or under occ %d at x:%d y:%d agent %d\n",
							gbwNewBits.asBits.occupancy,sXStore,sYStore,gbwOldBits.asBits.occupancy,sXCenter,sYCenter,iAgentID);

					} else {			

						remove_resident(&gbwOldBits,iOldAddy,pigResidents,iAgentID);
						insert_resident(&gbwNewBits,iNewAddy,pigResidents,psaX,psaY,sXStore,sYStore,iAgentID);

						// update global occupancy values
						int iFlag = atomicExch(&(pigGridBits[iNewAddy]),gbwNewBits.asInt);
						iFlag = atomicExch(&(pigGridBits[iOldAddy]),gbwOldBits.asInt);
					}
				}
			}
		}
	}
	return;
}

