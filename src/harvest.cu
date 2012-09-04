/*
 * harvest.cu
 *
 *  Created on: Dec 3, 2011
 *      Author: C. Richard Fisel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "constants.h"
#include "harvest.h"

__global__ void harvest(short* psaX, int* pigBits, int* pigResidents,
		float* pfaSugar, float* pfaSpice, curandState* pgStates)
{
	int iAddy = blockIdx.x*blockDim.x+threadIdx.x;
	int iAgentID;

	// make local copy of grid bits
	int iTemp = pigBits[iAddy];
	short sOcc = (iTemp&occupancyMask)>>occupancyShift;
	// harvest based on occupancy
	switch (sOcc) {
	case 0:
		break;
	case 1:
		iAgentID = pigResidents[iAddy*MAX_OCCUPANCY];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

			pfaSugar[iAgentID] += (iTemp&sugarMask)>>sugarShift;
			pfaSpice[iAgentID] += (iTemp&spiceMask)>>spiceShift;
			iTemp &= ~sugarMask;
			iTemp &= ~spiceMask;
			pigBits[iAddy] = iTemp;
		}
//		if (iAgentID >= INIT_AGENTS) printf("Error - lone agent ID %d too big!\n",iAgentID);
		break;
	default:
		curandState localState = pgStates[iAddy];
		short sOffset = curand_uniform(&localState)*sOcc;
		iAgentID = pigResidents[iAddy*MAX_OCCUPANCY+sOffset];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

			pfaSugar[iAgentID] += (iTemp&sugarMask)>>sugarShift;
			pfaSpice[iAgentID] += (iTemp&spiceMask)>>spiceShift;
			iTemp &= ~sugarMask;
			iTemp &= ~spiceMask;
			pigBits[iAddy] = iTemp;
		}
		pgStates[iAddy] = localState;
//		if (iAgentID >= INIT_AGENTS) printf("Error - agent ID %d too big!\n",iAgentID);
		break;
	}
	return;
}
