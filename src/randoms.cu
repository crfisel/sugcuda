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
	abwTemp.asBits.pad = 0x3F;
	abwTemp.asBits.isLocked = 0;
	float fTemp = buTemp.asBits.b19+2*buTemp.asBits.b20+4*buTemp.asBits.b21+8*buTemp.asBits.b22+16*buTemp.asBits.b23+32*buTemp.asBits.b24+64*buTemp.asBits.b25;
	abwTemp.asBits.age = fTemp*60.0f/127.0f;
	
	target[iAgentID] = abwTemp.asInt;
}
__global__ void fill_positions(unsigned int* piaRandoms, short* psaX, short* psaY)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int adjusted = piaRandoms[iAgentID]; // % (GRID_SIZE*GRID_SIZE);
	psaX[iAgentID] = adjusted/GRID_SIZE;
	psaY[iAgentID] = adjusted&(GRID_SIZE-1); // same as % GRID_SIZE
//	printf("%d:%d\n",psaX[iAgentID],psaY[iAgentID]);
}

__global__ void initialize_gridbits(unsigned int* pigRandoms, int* target, grid_layout gridCode)
{
	int iLoc = threadIdx.x + blockIdx.x*blockDim.x;
	BitUnpacker buTemp;
	float fTemp;
	buTemp.asUInt = pigRandoms[iLoc];
	short sXRel;
	short sYRel;
	short sTileX;
	short sTileY;

	// pack bit fields separately
	GridBitWise gbwBits;
	gbwBits.asBits.isLocked = 0;
	gbwBits.asBits.occupancy = 0;
	switch (gridCode) {
	case TILED:
		// position relative to tile boundaries (same as %16)
		sXRel = blockIdx.x&15;
		sYRel = threadIdx.x&15;

		// tile position (same as /16)
		sTileX = blockIdx.x>>4;
		sTileY = threadIdx.x>>4;

		// for even-even or odd-odd, it's spice, otherwise sugar
		if (sTileX&1 == sTileY&1) {
			gbwBits.asBits.sugar = 0.0f;
			gbwBits.asBits.spice = tile_value(sXRel,sYRel);
		} else {
			gbwBits.asBits.spice = 0.0f;
			gbwBits.asBits.sugar = tile_value(sXRel,sYRel);
		}
	case STRETCHED:
		// position relative to tile boundaries (same as %(GRID_SIZE/2))
		sXRel = blockIdx.x&((GRID_SIZE>>1)-1);
		sYRel = threadIdx.x&((GRID_SIZE>>1)-1);

		// tile position (same as /2)
		sTileX = blockIdx.x>>1;
		sTileY = threadIdx.x>>1;

		// for even-even or odd-odd, it's spice, otherwise sugar
		if (sTileX&1 == sTileY&1) {
			gbwBits.asBits.sugar = 0.0f;
			gbwBits.asBits.spice = stretched_value(sXRel,sYRel);
		} else {
			gbwBits.asBits.spice = 0.0f;
			gbwBits.asBits.sugar = stretched_value(sXRel,sYRel);
		}
	case RANDOM:
	default:
		fTemp = buTemp.asBits.b1+2*buTemp.asBits.b2+4*buTemp.asBits.b3+8*buTemp.asBits.b4;
		gbwBits.asBits.sugar = fTemp*10.0f/15.0f;
		fTemp = buTemp.asBits.b5+2*buTemp.asBits.b6+4*buTemp.asBits.b7+8*buTemp.asBits.b8;
		gbwBits.asBits.spice = fTemp*10.0f/15.0f;
	}
	gbwBits.asBits.maxSugar = gbwBits.asBits.sugar;
	gbwBits.asBits.maxSpice = gbwBits.asBits.spice;
	gbwBits.asBits.pad = 0;

	target[iLoc] = gbwBits.asInt;
}

