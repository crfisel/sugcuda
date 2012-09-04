/*
 * eat.cu
 *
 *  Created on: Dec 3, 2011
 *      Author: C. Richard Fisel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "constants.h"
#include "eat.h"

__global__ void eat(short* psaX, int* piaBits, float* pfaSugar, float* pfaSpice)
{
	int iAgentID = blockIdx.x*blockDim.x+threadIdx.x;
	int iTemp;

	// if the agent is alive
	if (psaX[iAgentID] > -1) {
		iTemp = piaBits[iAgentID];
		pfaSugar[iAgentID] -= (iTemp&metSugarMask)>>metSugarShift;
		pfaSpice[iAgentID] -= (iTemp&metSpiceMask)>>metSpiceShift;
	}
	return;
}
