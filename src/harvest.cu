/*
 * harvest.cu
 *
 *  Created on: Dec 3, 2011
 *      Author: C. Richard Fisel
 */

#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "harvest.h"

__global__ void harvest(curandState* pgStates, short* psaX, float* pfaSugar, float* pfaSpice,
		int* pigGridBits, int* pigResidents)
{
	int iAddy = blockIdx.x*blockDim.x+threadIdx.x;
	int iAgentID;
	short iOffset;
	GridBitWise gbwBits;

	gbwBits.asInt = pigGridBits[iAddy];
	switch (gbwBits.asBits.occupancy) {
	case 0:
		break;
	case 1:
		iAgentID = pigResidents[iAddy*MAX_OCCUPANCY];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

			pfaSugar[iAgentID] += gbwBits.asBits.sugar;
			pfaSpice[iAgentID] += gbwBits.asBits.spice;
			gbwBits.asBits.sugar = 0;
			gbwBits.asBits.spice = 0;
			pigGridBits[iAddy] = gbwBits.asInt;
		}
		break;
	default:
		curandState localState = pgStates[iAddy];
		iOffset = curand_uniform(&localState)*(gbwBits.asBits.occupancy);
		iAgentID = pigResidents[iAddy*MAX_OCCUPANCY+iOffset];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

			pfaSugar[iAgentID] += gbwBits.asBits.sugar;
			pfaSpice[iAgentID] += gbwBits.asBits.spice;
			gbwBits.asBits.sugar = 0;
			gbwBits.asBits.spice = 0;
			pigGridBits[iAddy] = gbwBits.asInt;
		}
		pgStates[iAddy] = localState;
		break;
	}
	return;
}
