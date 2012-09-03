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
#include "age.h"

__global__ void age(short* psaAge)
{
	int iAgentID = blockIdx.x*blockDim.x+threadIdx.x;
	psaAge[iAgentID]++;
	return;
}
