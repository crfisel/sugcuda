/*
 * move.h
 *
 *  Created on: Nov 24, 2011
 *      Author: C. Richard Fisel
 */

#ifndef MOVE_H_
#define MOVE_H_

#include <cuda.h>
#include "bitwise.h"

__forceinline__ 
__device__ bool lock(int iAddy, GridBitWise* gbwBits, int* pigGridBits)
{
	bool status = false;
	
	// unpack grid bits
	gbwBits->asInt = pigGridBits[iAddy];
	
	// test if square is locked
	if (gbwBits->asBits.isLocked == 0) {
	
		// if so, make a copy, but indicating locked
		GridBitWise gbwBitsCopy;
		gbwBitsCopy.asInt = gbwBits->asInt;
		gbwBitsCopy.asBits.isLocked = 1;
		
		// now lock the current address if possible
		int iLocked = atomicCAS(&(pigGridBits[iAddy]),gbwBits->asInt,gbwBitsCopy.asInt);
		
		// test if the lock worked
		if (iLocked == gbwBits->asInt) {
			status = true;
			gbwBits->asInt = gbwBitsCopy.asInt;
		}
	}
	return status;
}

__forceinline__ 
__device__ void remove_resident(GridBitWise* gbwTemp, int iAddy, int* pigResidents, int iAgentID)
{
	// find match starting at end of list
	short k = --gbwTemp->asBits.occupancy;

	// remove current id - if not at the end, replace it with the one from the end and store -1 at end
	if (pigResidents[iAddy*MAX_OCCUPANCY+k] == iAgentID) {
		pigResidents[iAddy*MAX_OCCUPANCY+k] = -1;
	} 
	else {
		while (pigResidents[iAddy*MAX_OCCUPANCY+k] != iAgentID && k >= 0) {k--;}
		if (k != gbwTemp->asBits.occupancy) {
			pigResidents[iAddy*MAX_OCCUPANCY+k] = pigResidents[iAddy*MAX_OCCUPANCY+gbwTemp->asBits.occupancy];
			pigResidents[iAddy*MAX_OCCUPANCY+gbwTemp->asBits.occupancy] = -1;
		}
	}
}

__forceinline__ 
__device__ void insert_resident(GridBitWise* gbwTemp, int iAddy, int* pigResidents, short* psaX, short* psaY, short sXStore, short sYStore, int iAgentID)
{
	// make sure we are replacing an "empty" placemarker
	if (pigResidents[iAddy*MAX_OCCUPANCY+gbwTemp->asBits.occupancy] == -1) {
		psaX[iAgentID] = sXStore;
		psaY[iAgentID] = sYStore;
		pigResidents[iAddy*MAX_OCCUPANCY+gbwTemp->asBits.occupancy] = iAgentID;

		// increment occupancy at new address
		gbwTemp->asBits.occupancy++;
	} else {
		//otherwise notify about the error
		printf ("agent replaced %d at x:%d y:%d \n",
			pigResidents[iAddy*MAX_OCCUPANCY+gbwTemp->asBits.occupancy],sXStore,sYStore);
	}
}

__global__ void best_move_by_traversal(short* , short* , int* ,	float* , float* , 
	int* , int* , int* , const int , int* , int* , int* );

__global__ void best_move_by_traversal_fs(short* , short* , int* , float* , float* , 
	int* , int* , int* , const int );

#endif /* MOVE_H_ */

