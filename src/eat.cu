/*
 * eat.cu
 *
 *  Created on: Dec 3, 2011
 *      Author: C. Richard Fisel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "eat.h"

__global__ void eat(short* psaX, int* piaAgentBits, float* pfaSugar, float* pfaSpice)
{
	int iAgentID = blockIdx.x*blockDim.x+threadIdx.x;

	// work with live agents only
	if (psaX[iAgentID] > -1) {
		AgentBitWise abwBits;
		abwBits.asInt = piaAgentBits[iAgentID];
		pfaSugar[iAgentID] -= (abwBits.asBits.metSugar+1);
		pfaSpice[iAgentID] -= (abwBits.asBits.metSpice+1);
	}
	return;
}
