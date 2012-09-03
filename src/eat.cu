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
#include "bitwisetype.h"
#include "eat.h"

__global__ void eat(short* psaX, BitWiseType* pbaBits, float* pfaSugar, float* pfaSpice)
{
	int iAgentID = blockIdx.x*blockDim.x+threadIdx.x;

	// work with live agents only
	if (psaX[iAgentID] > -1) {
		pfaSugar[iAgentID] -= ((&pbaBits[iAgentID])->metSugar+1);
		pfaSpice[iAgentID] -= ((&pbaBits[iAgentID])->metSpice+1);
	}
	return;
}
