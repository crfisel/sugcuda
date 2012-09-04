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
	float fRadius = (x-7.5)*(x-7.5)+(y-7.5)*(y-7.5);
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
__global__ void initialize_food(unsigned int* , float* , float );

__global__ void initialize_agentbits(unsigned int* , int* );

__global__ void fill_positions(unsigned int* , short* , short* );

__global__ void initialize_gridbits(unsigned int* , int* , grid_layout );

#endif //RANDOMS_H
