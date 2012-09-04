/*
 * grow_back1.cu
 *
 *  Created on: Dec 20, 2011
 *      Author: C. Richard Fisel
 */

#include <stdlib.h>
#include <cuda.h>
#include "constants.h"
#include "grow_back1.h"

__global__ void grow_back1(int* pigBits)
{
	int iAddy = threadIdx.x + blockIdx.x*blockDim.x;
	int iTemp = pigBits[iAddy];
	if ((iTemp&sugarMask)>>sugarShift < (iTemp&maxSugarMask)>>maxSugarShift)
		iTemp += sugarIncrement;
	if ((iTemp&spiceMask)>>spiceShift < (iTemp&maxSpiceMask)>>maxSpiceShift)
		iTemp += spiceIncrement;
	pigBits[iAddy] = iTemp;
	return;
}
