#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "constants.h"
#include "randoms.h"

__global__ void setup_kernel(curandState* pStates)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	/* Each thread gets same seed, a different sequence number, no offset */
	curand_init (1234,id,0,&pStates[id]);
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
	int iAgentID = threadIdx.x + blockIdx.x*blockDim.x;
	curandState localState = paStates[iAgentID];

	// NOTE: this first setting leaves pad and grid lock fields = 0
	int iTemp = visionIncrement*curand_uniform(&localState)*3.999f;
	iTemp += metSugarIncrement*curand_uniform(&localState)*3.999f;
	iTemp += metSpiceIncrement*curand_uniform(&localState)*3.999f;
	iTemp += startFertilityAgeIncrement*curand_uniform(&localState)*3.999f;
	iTemp += endFertilityAgeIncrement*curand_uniform(&localState)*15.999f;
	iTemp += deathAgeIncrement*curand_uniform(&localState)*31.999f;
	iTemp += ageIncrement*curand_uniform(&localState)*59.999f;
	iTemp += isFemaleIncrement*((curand_uniform(&localState) > 0.5) ? 0: 1);

	// copy back to globals
	target[iAgentID] = iTemp;
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
	int iTemp = 0;

	// pack bit fields separately
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
			iTemp += spiceIncrement*tile_value(sXRel,sYRel);
		} else {
			iTemp += sugarIncrement*tile_value(sXRel,sYRel);
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
			iTemp += spiceIncrement*stretched_value(sXRel,sYRel);
		} else {
			iTemp += sugarIncrement*stretched_value(sXRel,sYRel);
		}
	case RANDOM:
	default:
		curandState localState = pgStates[iLoc];

		iTemp =+ sugarIncrement*curand_uniform(&localState)*10;
		iTemp += spiceIncrement*curand_uniform(&localState)*10;

		pgStates[iLoc] = localState;
	}
	iTemp += maxSugarIncrement*((iTemp&sugarMask)>>sugarShift);
	iTemp += maxSpiceIncrement*((iTemp&spiceMask)>>spiceShift);

	target[iLoc] = iTemp;
}
