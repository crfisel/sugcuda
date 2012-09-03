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

__forceinline__ __device__ short wraparound(short i) {

		short sTry = i;
		if (sTry < 0) sTry += GRID_SIZE;
		if (sTry >= GRID_SIZE) sTry -= GRID_SIZE;
		return sTry;
}

inline
__device__ void remove_resident(int* piBits, int iAddy, int* pigResidents, int iAgentID)
{
	// convert to bitwise
	GridBitWise gbwBits;
	gbwBits.asInt = *piBits;
	
	// find match starting at end of list
	short k = --gbwBits.asBits.occupancy;

	// remove current id - if not at the end, replace it with the one from the end and store -1 at end
	if (pigResidents[iAddy*MAX_OCCUPANCY+k] == iAgentID) {
		pigResidents[iAddy*MAX_OCCUPANCY+k] = -1;
	} else {
		while (pigResidents[iAddy*MAX_OCCUPANCY+k] != iAgentID && k >= 0) {k--;}
		if (k != gbwBits.asBits.occupancy) {
			pigResidents[iAddy*MAX_OCCUPANCY+k] = pigResidents[iAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy];
			pigResidents[iAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy] = -1;
		}
	}
	*piBits = gbwBits.asInt;
}

inline
__device__ void insert_resident(int* piBits, int iAddy, int* pigResidents, short* psaX, short* psaY, short sXStore, short sYStore, int iAgentID)
{
	// convert to bitwise
	GridBitWise gbwBits;
	gbwBits.asInt = *piBits;

	// make sure we are replacing an "empty" placemarker
	if (pigResidents[iAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy] == -1) {
		psaX[iAgentID] = sXStore;
		psaY[iAgentID] = sYStore;
		pigResidents[iAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy] = iAgentID;

		// increment occupancy at new address
		gbwBits.asBits.occupancy++;
	} else {

		//otherwise notify about the error
		printf ("agent replaced %d at x:%d y:%d \n",
			pigResidents[iAddy*MAX_OCCUPANCY+gbwBits.asBits.occupancy],
			sXStore,sYStore);
	}
	*piBits = gbwBits.asInt;
}

inline
__device__ short2 find_best_move_by_traversal(short sXCenter, short sYCenter, int* piaAgentBits, float* pfaSugar, 
	float* pfaSpice, int* pigGridBits, int iAgentID)
{
	short sXStore = sXCenter;
	short sYStore = sYCenter;
			
	// reinterpret agent bits 
	AgentBitWise abwBits;
	abwBits.asInt = piaAgentBits[iAgentID];
			
	// scale values by agent's current sugar and spice levels,
	// converting to a duration using metabolic rates
	// add .01 to current values to avoid div by zero
	float sugarScale = 1.0f/(pfaSugar[iAgentID]+0.01f)/(abwBits.asBits.metSugar+1);
	float spiceScale = 1.0f/(pfaSpice[iAgentID]+0.01f)/(abwBits.asBits.metSpice+1);

	// search limits based on vision
	short sXMin = sXCenter-abwBits.asBits.vision-1;
	short sXMax = sXCenter+abwBits.asBits.vision+1;
	short sYMin = sYCenter-abwBits.asBits.vision-1;
	short sYMax = sYCenter+abwBits.asBits.vision+1;

	// calculate the value of the current square,
	// weighting its sugar and spice by need, metabolism, and occupancy
	int iTemp = sXCenter*GRID_SIZE+sYCenter;
	// reinterpret grid bits
	GridBitWise gbwBits;
	gbwBits.asInt = pigGridBits[iTemp];
	float fBest = gbwBits.asBits.spice*spiceScale/(gbwBits.asBits.occupancy+0.01f)
		+ gbwBits.asBits.sugar*sugarScale/(gbwBits.asBits.occupancy+0.01f);

	// search a square neighborhood of dimension 2*vision+3 (from 3x3 to 9x9)
	float fTest = 0.0f;
	iTemp = 0;
	short sXTry = 0;
	short sYTry = 0;
	for (short i = sXMin; i <= sXMax; i++) {
		sXTry = wraparound(i);
		for (short j = sYMin; j <= sYMax; j++) {
			sYTry = wraparound(j);

			// weight target's sugar and spice by need, metabolism, and occupancy
			iTemp = sXTry*GRID_SIZE+sYTry;
			gbwBits.asInt = pigGridBits[iTemp];
			fTest = gbwBits.asBits.spice*spiceScale/(gbwBits.asBits.occupancy+1)
				+ gbwBits.asBits.sugar*sugarScale/(gbwBits.asBits.occupancy+1);

			// choose new square if it's better
			if (fTest> fBest) {
				sXStore = sXTry;
				sYStore = sYTry;
				fBest = fTest;
			}
		}
	}
	return make_short2(sXStore,sYStore);
}

__global__ void best_move_by_traversal(short* , short* , int* ,	float* , float* , 
	int* , int* , int* , const int , int* , int* , int* );

__global__ void best_move_by_traversal_fs(short* , short* , int* , float* , float* , 
	int* , int* , int* , const int );

int move (short* , short* , int* , float* , float* , int* , int* , int* , const int , 
	int* , int* , int* );

#endif /* MOVE_H_ */

