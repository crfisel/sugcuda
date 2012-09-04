#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "randoms.h"

__global__ void setup_kernel(curandState* state)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	/* Each thread gets same seed, a different sequence number, no offset */
	curand_init (1234,id,0,&state[id]);
}

__global__ void initialize_food(float* pfaSugar, float* pfaSpice, curandState* paStates, float range)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	curandState localState = paStates[iAgentID];

	pfaSugar[iAgentID] = curand_uniform(&localState)*range;
	pfaSpice[iAgentID] = curand_uniform(&localState)*range;

	paStates[iAgentID] = localState;

//	printf("%f, %f\n",pfaSugar[iAgentID],pfaSpice[iAgentID]);
}
__global__ void initialize_agentbits(curandState* paStates, int* target)
{
	AgentBitWise abwTemp;
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	curandState localState = paStates[iAgentID];

	abwTemp.asBits.isFemale = (curand_uniform(&localState) > 0.5) ? 0: 1;
	// printf("%d\n",abwTemp.asBits.isFemale);
	abwTemp.asBits.vision = curand_uniform(&localState)*3.999f;
	abwTemp.asBits.metSugar = curand_uniform(&localState)*3.999f;
	abwTemp.asBits.metSpice = curand_uniform(&localState)*3.999f;
	abwTemp.asBits.startFertilityAge = curand_uniform(&localState)*3.999f;
	abwTemp.asBits.endFertilityAge = curand_uniform(&localState)*15.999f;
	abwTemp.asBits.deathAge = curand_uniform(&localState)*31.999f;
	abwTemp.asBits.pad = 0;
	abwTemp.asBits.isLocked = 0;
	abwTemp.asBits.age = curand_uniform(&localState)*59.999f;
	
	target[iAgentID] = abwTemp.asInt;
	paStates[iAgentID] = localState;
}
__global__ void fill_positions(curandState* paStates, short* psaX, short* psaY, short range)
{
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	curandState localState = paStates[iAgentID];

	psaX[iAgentID] = (curand_uniform(&localState))*(range-0.01f);
	psaY[iAgentID] = (curand_uniform(&localState))*(range-0.01f);

	paStates[iAgentID] = localState;

//	printf("%d,%d\n",psaX[iAgentID],psaY[iAgentID]);
}

__global__ void initialize_gridbits(curandState* pgStates, int* target, grid_layout gridCode)
{
	int iLoc = threadIdx.x + blockIdx.x*blockDim.x;
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
		curandState localState = pgStates[iLoc];

		gbwBits.asBits.sugar = curand_uniform(&localState)*10.0f;
		gbwBits.asBits.spice = curand_uniform(&localState)*10.0f;

		pgStates[iLoc] = localState;
	}
	gbwBits.asBits.maxSugar = gbwBits.asBits.sugar;
	gbwBits.asBits.maxSpice = gbwBits.asBits.spice;
	gbwBits.asBits.pad = 0;

	target[iLoc] = gbwBits.asInt;
}
