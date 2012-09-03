#include <stdio.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "randoms.h"

__global__ void initialize_food(unsigned int* theRandoms, float* target, float range)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	target[iAgentID] = theRandoms[iAgentID]*range/UINT_MAX;
//	printf("%f\n",target[iAgentID]);
}
__global__ void initialize_agentbits(unsigned int* theRandoms, int* target)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	AgentBitWise abwTemp;
	abwTemp.asInt = theRandoms[iAgentID]-INT_MIN;
	
	//TODO: fix unpacking of bits
	
	target[iAgentID] = abwTemp.asInt;
}
__global__ void fill_positions(unsigned int* theRandoms, short* psaX, short* psaY)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int adjusted = theRandoms[iAgentID] % (GRID_SIZE*GRID_SIZE);
	psaX[iAgentID] = adjusted / GRID_SIZE;
	psaY[iAgentID] = adjusted % GRID_SIZE;
//	printf("%d:%d\n",psaX[iAgentID],psaY[iAgentID]);
}

__global__ void initialize_gridbits(unsigned int* theRandoms, int* target)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	// pack bit fields separately
	GridBitWise gbwBits;
	gbwBits.asInt = theRandoms[iAgentID]-INT_MIN;
	//TODO: fix unpacking of bits

	gbwBits.asBits.isLocked = 0;
	gbwBits.asBits.occupancy = 0;

	target[iAgentID] = gbwBits.asInt;
}

