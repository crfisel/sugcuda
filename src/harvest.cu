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
#include "symbolic_constants.h"
#include "randoms.h"
#include "harvest.h"

__global__ void harvest(curandStateXORWOW_t* devGridStates, short* psaX, float* pfaSugar, float* pfaSpice,
		short* psgSugar, short* psgSpice, short* psgOccupancy, int* pigResidents)
{
	short sX = blockIdx.x;
	short sY = threadIdx.x;
	int iAddy = sX*blockDim.x+sY;
	int iAgentID;
	short iOffset;

	switch (psgOccupancy[iAddy]) {
	case 0:
		break;
	case 1:
		iAgentID = pigResidents[iAddy*MAX_OCCUPANCY];

		// work with live agents only
		if (psaX[iAgentID] > -1) {

			pfaSugar[iAgentID] += psgSugar[iAddy];
			pfaSpice[iAgentID] += psgSpice[iAddy];
		}
		break;
	default:
		curandStateXORWOW_t localState = devGridStates[iAddy];
		iOffset = curand_uniform(&localState)*psgOccupancy[iAddy];
		devGridStates[iAddy] = localState;
		iAgentID = pigResidents[iAddy*MAX_OCCUPANCY+iOffset];

		// work with live agents only
		if (psaX[iAgentID] > -1) {

			pfaSugar[iAgentID] += psgSugar[iAddy];
			pfaSpice[iAgentID] += psgSpice[iAddy];
		}
		break;
	}
	return;
}
