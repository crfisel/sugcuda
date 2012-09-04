#include <stdio.h>
#include <cuda.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "randoms.h"

__global__ void initialize_food(unsigned int* piaRandoms, float* target, float range)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	target[iAgentID] = piaRandoms[iAgentID]*range/UINT_MAX;
//	printf("%f\n",target[iAgentID]);
}
__global__ void initialize_agentbits(unsigned int* piaRandoms, int* target)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	BitUnpacker buTemp;
	AgentBitWise abwTemp;
	buTemp.asUInt = piaRandoms[iAgentID];
	
	abwTemp.asBits.isFemale = buTemp.asBits.b1;
	abwTemp.asBits.vision = buTemp.asBits.b2+2*buTemp.asBits.b3;
	abwTemp.asBits.metSugar = buTemp.asBits.b4+2*buTemp.asBits.b5;
	abwTemp.asBits.metSpice = buTemp.asBits.b6+2*buTemp.asBits.b7;
	abwTemp.asBits.startFertilityAge = buTemp.asBits.b8+2*buTemp.asBits.b9;
	abwTemp.asBits.endFertilityAge = buTemp.asBits.b10+2*buTemp.asBits.b11+4*buTemp.asBits.b12+8*buTemp.asBits.b13;
	abwTemp.asBits.deathAge = buTemp.asBits.b14+2*buTemp.asBits.b15+4*buTemp.asBits.b16+8*buTemp.asBits.b17+16*buTemp.asBits.b18;
	abwTemp.asBits.pad = 0;
	abwTemp.asBits.isLocked = 0;
	float fTemp = buTemp.asBits.b8+2*buTemp.asBits.b9+4*buTemp.asBits.b10+8*buTemp.asBits.b11+16*buTemp.asBits.b12+32*buTemp.asBits.b13+64*buTemp.asBits.b4;
	abwTemp.asBits.age = fTemp*100.0f/127.0f;
	
	target[iAgentID] = abwTemp.asInt;
}
__global__ void fill_positions(unsigned int* piaRandoms, short* psaX, short* psaY)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int adjusted = piaRandoms[iAgentID]; // % (GRID_SIZE*GRID_SIZE);
	psaX[iAgentID] = adjusted / GRID_SIZE;
	psaY[iAgentID] = adjusted % GRID_SIZE;
//	printf("%d:%d\n",psaX[iAgentID],psaY[iAgentID]);
}

__global__ void initialize_gridbits(unsigned int* pigRandoms, int* target)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	BitUnpacker buTemp;
	buTemp.asUInt = pigRandoms[iAgentID];

	// pack bit fields separately
	GridBitWise gbwBits;
	gbwBits.asBits.isLocked = 0;
	gbwBits.asBits.occupancy = 0;
	float fTemp = buTemp.asBits.b1+2*buTemp.asBits.b2+4*buTemp.asBits.b3+8*buTemp.asBits.b4;
	gbwBits.asBits.sugar = fTemp*10.0f/15.0f;
	gbwBits.asBits.maxSugar = gbwBits.asBits.sugar;
	fTemp = buTemp.asBits.b5+2*buTemp.asBits.b6+4*buTemp.asBits.b7+8*buTemp.asBits.b8;
	gbwBits.asBits.spice = fTemp*10.0f/15.0f;
	gbwBits.asBits.maxSpice = gbwBits.asBits.spice;
	gbwBits.asBits.pad = 0;

	target[iAgentID] = gbwBits.asInt;
}

