/*
 * grow_back1.cu
 *
 *  Created on: Dec 20, 2011
 *      Author: C. Richard Fisel
 */

#include <stdlib.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "grow_back1.h"

__global__ void grow_back1(int* pigGridBits)
{
	int iAddy = threadIdx.x + blockIdx.x*blockDim.x;
	GridBitWise gbwBits;

	gbwBits.asInt = pigGridBits[iAddy];
	if (gbwBits.asBits.sugar < gbwBits.asBits.maxSugar) gbwBits.asBits.sugar++;
	if (gbwBits.asBits.spice < gbwBits.asBits.maxSpice) gbwBits.asBits.spice++;
	pigGridBits[iAddy] = gbwBits.asInt;
	return;
}
