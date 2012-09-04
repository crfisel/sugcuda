/*
 * harvest.h
 *
 *  Created on: Dec 3, 2011
 *      Author: C. Richard Fisel
 */

#ifndef HARVEST_H_
#define HARVEST_H_

__global__ void harvest(short* psaX, int* pigBits, int* pigResidents,
		float* pfaSugar, float* pfaSpice, curandState* pgStates);

#endif /* HARVEST_H_ */
