/*
 * randoms.h
 *
 *  Created on: Nov 14, 2011
 *      Author: C. Richard Fisel
 */

#ifndef RANDOMS_H_
#define RANDOMS_H_

enum grid_layout {RANDOM, TILED, STRETCHED};

__forceinline__ __device__ float tile_value(short x, short y)
{
	float fRadius = (x-7.5f)*(x-7.5f)+(y-7.5f)*(y-7.5f);
	short sTemp = 0;
	if (fRadius < 85.0f) sTemp++;
	if (fRadius < 61.0f) sTemp++;
	if (fRadius < 41.0f) sTemp++;
	if (fRadius < 25.0f) sTemp++;
	if (fRadius < 13.0f) sTemp++;
	if (fRadius < 5.0f) sTemp++;
	if (fRadius < 1.0f) sTemp++;
	return (float) sTemp;
}
__forceinline__ __device__ float stretched_value(short x, short y)
{
	const float fCenter = (GRID_SIZE-1)/2.0f;
	float fRadius = ((x-fCenter)*(x-fCenter)+(y-fCenter)*(y-fCenter))/GRID_SIZE/GRID_SIZE;
	short sTemp = 0;
	if (fRadius < 0.42f) sTemp++;
	if (fRadius < 0.36f) sTemp++;
	if (fRadius < 0.31f) sTemp++;
	if (fRadius < 0.26f) sTemp++;
	if (fRadius < 0.22f) sTemp++;
	if (fRadius < 0.18f) sTemp++;
	if (fRadius < 0.15f) sTemp++;
	if (fRadius < 0.11f) sTemp++;
	if (fRadius < 0.08f) sTemp++;
	if (fRadius < 0.05f) sTemp++;
	return (float) sTemp;
}

__global__ void setup_kernel(curandState* pStates);

__global__ void initialize_food(float* pfaSugar, float* pfaSpice, curandState* paStates, float range);

__global__ void initialize_agentbits(curandState* paStates, int* target);

__global__ void fill_positions(curandState* paStates, short* psaX, short* psaY, short range);

__global__ void initialize_gridbits(curandState* pgStates, int* target, grid_layout gridCode);

#endif //RANDOMS_H
