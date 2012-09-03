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
#include "bitwise.h"
#include "randoms.h"
#include "harvest.h"

__global__ void harvest(unsigned int* theRandoms, short* psaX, float* pfaSugar, float* pfaSpice,
		int* pigGridBits, int* pigResidents)
{
	short sX = blockIdx.x;
	short sY = threadIdx.x;
	int iAddy = sX*blockDim.x+sY;
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
		}
		break;
	default:
		float fTemp = theRandoms[iAddy]*gbwBits.asBits.occupancy/UINT_MAX;
		iOffset = fTemp;
		iAgentID = pigResidents[iAddy*MAX_OCCUPANCY+iOffset];

		// if the agent is alive
		if (psaX[iAgentID] > -1) {

			pfaSugar[iAgentID] += gbwBits.asBits.sugar;
			pfaSpice[iAgentID] += gbwBits.asBits.spice;
		}
		break;
	}
	return;
}
