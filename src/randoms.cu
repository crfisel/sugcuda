#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "symbolic_constants.h"
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
	float fTemp = curand_uniform(&localState)*range;

	/* Generate pseudo-random ints*/
	target[id] = fTemp;

	/* Copy state back to global memory */
	state[id] = localState;
}
__global__ void generate_shorts(curandStateXORWOW_t* state, short* target, short range)
{
	/* Copy to local memory for efficiency */
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curandStateXORWOW_t localState = state[id];
	float fTemp = curand_uniform(&localState)*range;

	/* Generate pseudo-random shorts */
	target[id] = fTemp;

	/* Copy state back to global memory */
	state[id] = localState;
}

