#include <limits.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "symbolic_constants.h"
#include "bitwisetype.h"
#include "randoms.h"
#include "rngs.h"


__global__ void setup_kernel(curandState* state)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	/* Each thread gets a different seed, a different sequence number, no offset */
	curand_init(((1237*id)%LONG_MAX),id,0,&state[id]);
}
__global__ void generate_floats(curandState* state, float* target, float range)
{
	/* Copy to local memory for efficiency */
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curandState localState = state[id];

	/* Generate pseudo-random floats*/
	target[id] = curand_uniform(&localState)*range;

	/* Copy state back to global memory */
	state[id] = localState;
}
__global__ void generate_ints(curandState* state, int* target, int range)
{
	/* Copy to local memory for efficiency */
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curandState localState = state[id];

	/* Generate pseudo-random ints*/
	target[id] = curand_uniform(&localState)*range;

	/* Copy state back to global memory */
	state[id] = localState;
}
__global__ void generate_shorts(curandState* state, short* target, short range)
{
	/* Copy to local memory for efficiency */
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curandState localState = state[id];

	/* Generate pseudo-random shorts */
	target[id] = curand_uniform(&localState)*range;

	/* Copy state back to global memory */
	state[id] = localState;
}

__global__ void generate_bits(curandState* state, BitWiseType* target)
{
	/* Copy to local memory for efficiency */
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curandState localState = state[id];
	BitWiseType* temp;
	short ts;

	/* Generate pseudo-random shorts */
	ts = (curand_uniform(&localState)*32768);
	temp = reinterpret_cast <BitWiseType*> (&ts);
	target[id] = *temp;

	/* Copy state back to global memory */
	state[id] = localState;
}


