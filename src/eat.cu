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

__global__ void eat(short* psaX, int* piaBits, float* pfaSugar, float* pfaSpice)
{
	int iAgentID = blockIdx.x*blockDim.x+threadIdx.x;

	// work with live agents only
	if (psaX[iAgentID] > -1) {
		BitWise	bwLocalBits;
		bwLocalBits.asInt = piaBits[iAgentID];
		pfaSugar[iAgentID] -= (bwLocalBits.asBits.metSugar+1);
		pfaSpice[iAgentID] -= (bwLocalBits.asBits.metSpice+1);
	}
	return;
}
