#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwisetype.h"
#include "mate.h"

__global__ void mate(short* psaX, short* psaY, short* pigGridBits, int* pigResidents, int* piaActiveQueue, 
	const int ciActiveQueueSize, int* piaDeferredQueue, int* piDeferredQueueSize, int* piLockSuccesses)
{
	int iAgentID;
	int iMateID;
	int iAddy;
	int iTemp;
	short sOldOcc;
	bool pregnant = false;
	bool lockFailed = false;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		iAgentID = piaActiveQueue[iOffset];

		// live, fertile, solvent female agents only
		if (((&pbaBits[iAgentID])->isFemale) == 1) {
			if (isFertile(iAgentID)) {
				if	((pfaSugar[iAgentID] > pfaInitialSugar[iAgentID]) &&
						(pfaSpice[iAgentID] > pfaInitialSpice[iAgentID])) {
					for (short sXTry = psaX[iAgentID] - 1; sXTry < psaX[iAgentID] + 2; sXTry++) {
						for (short sYTry = psaY[iAgentID] - 1; sYTry < psaY[iAgentID] + 2; sYTry++) {
							short sOccTry = 0;
							while (sOccTry < MAX_OCCUPANCY && !pregnant) {
								// get the potential mate's id
								int iMateID = pigResidents[(sXTry*GRID_SIZE+sYTry)*MAX_OCCUPANCY+sOccTry];
								// make sure this is not an "empty" placeholder
								if (iMateID > -1) {
									// make sure he's male, alive and fertile
									if ((&pbaBits[iAgentID])->isFemale == 0 && isFertile(iMateID)) {
										// if he's unlocked...
										short sHisAge = psaAge[iMateID];
										if (sHisAge > 0) {
											// lock him if possible by changing his age to negative
											int iTemp = atomicCAS(&(psaAge[iMateID]),sHisAge,-sHisAge);
											if (iTemp == sHisAge) {
												// now he's locked, check his solvency
												if	((pfaSugar[iMateID] > pfaInitialSugar[iMateID]) &&
														(pfaSpice[iMateID] > pfaInitialSpice[iMateID])) {
													// ok he's a keeper, make a baby...
													pregnant = true;
												} else {
													// unlock him
													iTemp = atomicExch(&(psaAge[iMateID]),abs(sHisAge));
												}
											} else {
												lockFailed = true;
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


