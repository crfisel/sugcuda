#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwisetype.h"
#include "mate.h"

__global__ void mate(short* psaX, short* psaY, int* pbaAgentBits, float* pfaSugar, float* pfaSpice, int* pigGridBits, int* pigResidents, 
	int* piaActiveQueue, const int ciActiveQueueSize, int* piaDeferredQueue, int* piDeferredQueueSize, int* piLockSuccesses)
{
	int iAgentID;
	int iMateID;
	int iAddy;
	int iAddyTry;
	GridBitWise gbwBits;
	GridBitWise gbwBitsTry;
	bool pregnant = false;
	bool lockSuccess = false;
	short sOccTry;
	AgentBitWise abwAgentBits;
	AgentBitWise abwMateBits;
	
	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		iAgentID = piaActiveQueue[iOffset];

		// live, fertile, solvent female agents only
		if (is_fertile(iAgentID) &&
			&pbaAgentBits[iAgentID])->isFemale == 1 &&	
			(pfaSugar[iAgentID] > pfaInitialSugar[iAgentID]) &&
			(pfaSpice[iAgentID] > pfaInitialSpice[iAgentID])) {
			iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];
			gbwBits.asInt = pigGridBits[iAddy];
			while (gbwBits.asBits.occupancy < MAX_OCCUPANCY) {
				// get nearest neighbors
				//#pragma unroll 3
				for (short sXTry = psaX[iAgentID] - 1; sXTry < psaX[iAgentID] + 2; sXTry++) {
					//#pragma unroll 3
					for (short sYTry = psaY[iAgentID] - 1; sYTry < psaY[iAgentID] + 2; sYTry++) {
						iAddyTry = sXTry*GRID_SIZE+sYTry;
						gbwBitsTry.asInt = pigGridBits[iAddyTry];
							for (sOccTry = 0; sOccTry < gbwBitsTry.asBits.occupancy; sOccTry++) {
								while (!pregnant) {
									// get the potential mate's id
									int iMateID = pigResidents[iAddyTry*MAX_OCCUPANCY+sOccTry];
									if (is_acceptable_mate(iMateID,pbaAgentBits,psaX)) lockSuccess = lock_potential_mate(iMateID,psaX,pbaAgentBits,&abwMateBits);
									if (lockSuccess) {
										// now he's locked, check his solvency
										if	((pfaSugar[iMateID] > pfaInitialSugar[iMateID]) &&
											(pfaSpice[iMateID] > pfaInitialSpice[iMateID])) {
											// ok he's a keeper, make a baby...
											pregnant = true;
											//etc...
										} else {
											// unlock him
											iTemp = atomicExch(&(pbaAgentBits[iMateID]),abwMateBits);
										}
									} else {
										lockSuccess = false;
									}
								}
							}
					}
								}
							}
						}
					}
					// TODO: if lock failed anywhere, defer this agent
				}
			}
		}
	}



				// increment square occupancy and counter of successful locks
				sOldOcc = psgOccupancy[iAddy]++;
				iTemp = atomicAdd(piLockSuccesses,1);

				// check for overflow
				if (sOldOcc < MAX_OCCUPANCY) {

					// insert the resident at the next position in the pigResidents list
					pigResidents[iAddy*MAX_OCCUPANCY+sOldOcc] = iAgentID;

				} else {

					// indicate an occupancy overflow
					printf ("overflow at x:%d y:%d \n",psaX[iAgentID], psaY[iAgentID]);
					psgOccupancy[iAddy] = MAX_OCCUPANCY+1;
				}


			}
			else {

				// otherwise, add the agent to the deferred queue
				iTemp = atomicAdd(piDeferredQueueSize,1);
				piaDeferredQueue[iTemp]=iAgentID;
			}
		}
	}
	return;
}


