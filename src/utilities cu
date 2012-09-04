/*
 * utilities.cu
 *
 *  Created on: Jan 21, 2012
 *      Author: C. Richard Fisel
 */

#include <cuda.h>
#include "constants.h"
#include "utilities.h"

//__forceinline__
__device__ bool lock_agent(int iAgentID, int* piaBits, const int iPopulation)
{
	bool lockSuccess = false;
	if (iAgentID > -1 && iAgentID < iPopulation) {
		int iTemp = piaBits[iAgentID];

		// if agent is unlocked...
		if (iTemp&agentLockMask == 0) {

			// lock if possible
			int iLocked = atomicCAS(&(piaBits[iAgentID]),iTemp,iTemp|agentLockMask);

			// test if the lock worked
			if (iLocked == iTemp) {
				lockSuccess = true;
			}

		}
	} else {
		printf("Error: bad agent id %d!\n",iAgentID);
	}
	return lockSuccess;
}

//__forceinline__
__device__ void unlock_agent(int iAgentID, int* piaBits, const int iPopulation)
{
	if (iAgentID > -1 && iAgentID < iPopulation) {

		// make a copy of the agent's bits, but indicating unlocked
		int iTemp = piaBits[iAgentID];
		iTemp &= ~agentLockMask;

		// switch unlocked copy for global value
		int iUnlocked = atomicExch(&(piaBits[iAgentID]),iTemp);
	} else {
		printf("Error: bad agent id %d!\n",iAgentID);
	}
	return;
}

//__forceinline__
__device__ bool lock_location(int iAddy, int* pigBits)
{
	bool lockSuccess = false;
	if (iAddy < GRID_SIZE*GRID_SIZE && iAddy > -1) {
		// unpack grid bits
		int iTemp = pigBits[iAddy];

		// test if square is unlocked
		if ((iTemp&gridLockMask) == 0) {

			// if so, make a copy, but indicating locked
			int iTempLocked = iTemp|gridLockMask;

			// now lock the address if possible
			int iLocked = atomicCAS(&(pigBits[iAddy]),iTemp,iTempLocked);

			// test if the lock worked
			if (iLocked == iTemp) lockSuccess = true;
		}
	} else {
		printf("Error: bad address %d!\n",iAddy);
	}
	return lockSuccess;
}

//__forceinline__
__device__ void unlock_location(int iAddy, int* pigBits)
{
	if (iAddy < GRID_SIZE*GRID_SIZE && iAddy > -1) {
		// unpack grid bits
		int iTemp = pigBits[iAddy];

		// set to unlocked
		iTemp &= ~gridLockMask;
		printf("itemp %x pigBits[iAddy] %x\n",iTemp,pigBits[iAddy]);
		// switch with the global value
		int iLocked = atomicExch(&(pigBits[iAddy]),iTemp);

	} else {

		printf("Error: bad address %d!\n",iAddy);
	}
	return;
}

/*
 * NOTE: use only when occupancy is previously verified to be positive
 * and thread contention for global memory has been suppressed (e.g. by locking)
 */
//__forceinline__
__device__ void remove_resident(int iAgentID, int iAddy, int* pigBits, int* pigResidents)
{
	if (iAgentID > -1) {
		// copy to local memory
		unsigned int iTemp = pigBits[iAddy];

		// decrement occupancy by one
		iTemp -= occupancyIncrement;

		// find match starting at end of list
		unsigned short sOcc = (iTemp&occupancyMask)>>occupancyShift;
		short k = sOcc;
		// remove current id - if not at the end, replace it with the one from the end and store -1 at end
		if (pigResidents[iAddy*MAX_OCCUPANCY+k] == iAgentID) {
			pigResidents[iAddy*MAX_OCCUPANCY+k] = -1;
		} else {
			while (pigResidents[iAddy*MAX_OCCUPANCY+k] != iAgentID && k >= 0) {k--;}
			if (k != sOcc) {
				pigResidents[iAddy*MAX_OCCUPANCY+k] = pigResidents[iAddy*MAX_OCCUPANCY+sOcc];
				pigResidents[iAddy*MAX_OCCUPANCY+sOcc] = -1;
			}
		}
		pigBits[iAddy] = iTemp;
	} else {
		printf("Error: agent id %d at addy %d is negative!\n",iAgentID,iAddy);
	}
	return;
}

/*
 * NOTE: use only when occupancy is previously verified to be less than MAX_OCCUPANCY
 * and thread contention for global memory has been suppressed (e.g. by locking).
 * NOTE also that GRID_SIZE is a power of 2.
 */
//__forceinline__
__device__ void add_resident(int iAgentID, int iAddy, int* pigBits, int* pigResidents, short* psaX, short* psaY)
{
	if (iAgentID > -1) {
		// copy to local memory
		unsigned int iTemp = pigBits[iAddy];

		// make sure we are replacing an "empty" placemarker
		if (pigResidents[iAddy*MAX_OCCUPANCY+(iTemp&occupancyMask)>>occupancyShift] == -1) {
			psaX[iAgentID] = iAddy>>log2GRID_SIZE;
			psaY[iAgentID] = iAddy&(GRID_SIZE-1);
			pigResidents[iAddy*MAX_OCCUPANCY+(iTemp&occupancyMask)>>occupancyShift] = iAgentID;

			// increment occupancy by one
			iTemp += occupancyIncrement;

		} else {

			//otherwise notify about the error
			printf ("agent %d replaced %d at addy %d \n",
					iAgentID,pigResidents[iAddy*MAX_OCCUPANCY+(iTemp&occupancyMask)>>occupancyShift],iAddy);
		}
		pigBits[iAddy] = iTemp;
	} else {
		printf("Error: agent id %d at addy %d is negative!\n",iAgentID,iAddy);
	}
	return;
}

