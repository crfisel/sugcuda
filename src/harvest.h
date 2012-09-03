/*
 * harvest.h
 *
 *  Created on: Dec 3, 2011
 *      Author: C. Richard Fisel
 */

#ifndef HARVEST_H_
#define HARVEST_H_

__global__ void harvest(curandStateXORWOW_t* , short* , float* , float* , int* , int* );

#endif /* HARVEST_H_ */
