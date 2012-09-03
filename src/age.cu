/*
 * age.cu
 *
 *  Created on: Dec 3, 2011
 *      Author: C. Richard Fisel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "age.h"

__global__ void age(int* piaAgentBits)
{
	int iAgentID = blockIdx.x*blockDim.x+threadIdx.x;
	AgentBitWise abwBits;
	abwBits.asInt = piaAgentBits[iAgentID];
	abwBits.asBits.age++;
	piaAgentBits[iAgentID]=abwBits.asInt;
	return;
}
