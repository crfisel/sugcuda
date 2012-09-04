/*
 * age.cu
 *
 *  Created on: Dec 3, 2011
 *      Author: C. Richard Fisel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "constants.h"
#include "age.h"

__global__ void age(int* piaBits)
{
	int iAgentID = blockIdx.x*blockDim.x+threadIdx.x;
	piaBits[iAgentID] += ageIncrement;
	return;
}
