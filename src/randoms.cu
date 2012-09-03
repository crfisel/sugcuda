#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "symbolic_constants.h"
#include "bitwise.h"
#include "randoms.h"

__global__ void setup_kernel(curandStateXORWOW_t* state)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	/* Each thread gets same seed, unique sequence number, no offset */
	curand_init(1234567,id,0,&state[id]);
}
__global__ void generate_floats(curandStateXORWOW_t* state, float* target, float range)
{
	/* Copy to local memory for efficiency */
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curandStateXORWOW_t localState = state[id];
	float fTemp = curand_uniform(&localState)*range;
	
	/* Generate pseudo-random floats*/
	target[id] = fTemp;

	/* Copy state back to global memory */
	state[id] = localState;
}
__global__ void generate_ints(curandStateXORWOW_t* state, int* target, int range)
{
	/* Copy to local memory for efficiency */
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curandStateXORWOW_t localState = state[id];

	/* Generate pseudo-random ints*/
	target[id] = curand(&localState);

	/* Copy state back to global memory */
	state[id] = localState;
}
__global__ void generate_shorts(curandStateXORWOW_t* state, short* target, short range)
{
	/* Copy to local memory for efficiency */
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curandStateXORWOW_t localState = state[id];
	float fTemp = curand_uniform(&localState) * range;
	
	/* Generate pseudo-random shorts */
	target[id] = fTemp;
//	printf("%d\n", target[id]);

	/* Copy state back to global memory */
	state[id] = localState;
}

__global__ void initialize_gridbits(curandStateXORWOW_t* state, int* target)
{
	/* Copy to local memory for efficiency */
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curandStateXORWOW_t localState = state[id];

	// pack bit fields separately
	GridBitWise gbwBits;
	gbwBits.asBits.isLocked = 0;
	gbwBits.asBits.occupancy = 0;

	float fTemp = curand_uniform(&localState)*1.1f;
	gbwBits.asBits.sugar = fTemp;
	gbwBits.asBits.maxSugar = gbwBits.asBits.sugar;

	fTemp = curand_uniform(&localState)*1.1f;
	gbwBits.asBits.spice = fTemp;
	gbwBits.asBits.maxSpice = gbwBits.asBits.spice;
	
	//store to target
	target[id] = gbwBits.asInt;

	/* Copy state back to global memory */
	state[id] = localState;
}

