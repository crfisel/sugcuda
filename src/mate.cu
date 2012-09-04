/*
 * mate.cu
 *
 *  Created on: Dec 4, 2011
 *      Author: C. Richard Fisel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "constants.h"
#include "utilities.h"
#include "mate.h"

__noinline__ __device__ bool is_fertile(int iAgentID, int* piaBits, short* psaX)
{
	int iTemp = piaBits[iAgentID];
	bool ydResult = (psaX[iAgentID] > -1) &&
			(((iTemp&ageMask)>>ageShift) > ((iTemp&startFertilityAgeMask)>>startFertilityAgeShift) + 12) &&
			(((iTemp&ageMask)>>ageShift) < 2*((iTemp&endFertilityAgeMask)>>endFertilityAgeShift) + 40);
	//	if (ydResult) printf("agent %d is fertile\n",iAgentID);
	return ydResult;
}

__noinline__ __device__ bool is_acceptable_mate(int iMateID, int* piaBits, short* psaX)
{
	bool acceptable = false;
	int iTemp = piaBits[iMateID];

	// make sure this is not an "empty" placeholder
	if (iMateID > -1) {
		// make sure he's male
		acceptable = (((iTemp&isFemaleMask)>>isFemaleShift) == 0) &&
				// and fertile
				is_fertile(iMateID,piaBits,psaX);
	}
	return acceptable;
}
/*
__noinline__ __device__ bool is_fertile(int iAgentID, AgentBitWise* abwBits, short* psaX)
{
	bool ydResult = (psaX[iAgentID] > -1) &&
			(abwBits->asBits.age > (abwBits->asBits.startFertilityAge + 12)) &&
			(abwBits->asBits.age < (2*abwBits->asBits.endFertilityAge + 40));
	return ydResult;
}

__noinline__ __device__ bool is_acceptable_mate(int iMateID, AgentBitWise* abwBits, short* psaX)
{
	bool acceptable = false;

	// make sure this is not an "empty" placeholder
	acceptable = (iMateID > -1 &&
		// make sure he's male
		abwBits->asBits.isFemale == 0 &&
		// and fertile
		is_fertile(iMateID,abwBits,psaX));
	return acceptable;
}

__noinline__ __device__ bool lock_agent(int iOtherAgentID, short* psaX, int* piaBits, AgentBitWise* abwBits)
{
	bool lockSuccess = false;

	// if he's unlocked...
	if (abwBits->asBits.isLocked == 0) {
		// make a copy, but indicating locked
		AgentBitWise abwBitsCopy;
		abwBitsCopy.asInt = abwBits->asInt;
		abwBitsCopy.asBits.isLocked = 1;

		// now lock him if possible
		int iLocked = atomicCAS(&(piaBits[iOtherAgentID]),abwBits->asInt,abwBitsCopy.asInt);

		// test if the lock worked
		if (iLocked == abwBits->asInt) {
			lockSuccess = true;
			abwBits->asInt = abwBitsCopy.asInt;
		}
	}
	return lockSuccess;
}

__global__ void mate_once(short* psaX, short* psaY, int* piaBits, int* pigBits, int* pigResidents,
		float* pfaSugar, float* pfaSpice, float* pfaInitialSugar, float* pfaInitialSpice,
		int* piaActiveQueue, int* piaDeferredQueue, const int ciActiveQueueSize, int* piPopulation,
		curandState* paStates, int* piDeferredQueueSize, int* piLockSuccesses)
{
	int iAgentID;
	int iMateID;
	int iAddy;
	int iAddyTry;
	GridBitWise gbwBits;
	GridBitWise gbwBitsTry;
	bool mated = false;
	bool isLocLocked = false;
	bool isMateLocked = false;
	short sOccTry;
	AgentBitWise abwAgentBits;
	AgentBitWise abwMateBits;
	BitUnpacker buRandoms;
	float fTemp = 0;
	int iTemp = 0;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		iAgentID = piaActiveQueue[iOffset];
	//	buRandoms.asUInt = piaRandoms[iAgentID];
		abwAgentBits.asInt = piaBits[iAgentID];

		// live, fertile, solvent female agents only
		if (is_fertile(iAgentID,&abwAgentBits,psaX) && abwAgentBits.asBits.isFemale == 1 &&
				(pfaSugar[iAgentID] > pfaInitialSugar[iAgentID]) && (pfaSpice[iAgentID] > pfaInitialSpice[iAgentID])) {
			iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];
			// need to have room on the grid for the kid
			isLocLocked = lock_location(iAddy,&gbwBits,pigBits);
			if (isLocLocked) {
				if (gbwBits.asBits.occupancy < MAX_OCCUPANCY) {
					printf("%d\n",iOffset);
					// get nearest neighbors
					for (short i = -1; i<= 1; i++) {
						for (short j = -1; j <= 1; j++) {
							iAddyTry = (psaX[iAgentID]+i)*GRID_SIZE+psaY[iAgentID]+j;
							gbwBitsTry.asInt = pigBits[iAddyTry];
							for (sOccTry = 0; sOccTry < gbwBitsTry.asBits.occupancy; sOccTry++) {
								// note that "mated" terminates the search for mates
								if (!mated) {
									// get the potential mate's id
									iMateID = pigResidents[iAddyTry*MAX_OCCUPANCY+sOccTry];

									// vet his internal properties
									abwMateBits.asInt = piaBits[iMateID];
									if (is_acceptable_mate(iMateID,&abwMateBits,psaX)) {
										// if acceptable, try to lock him
										isMateLocked = lock_agent(iMateID,psaX,piaBits,&abwMateBits);

										// to get to this point isLocLocked must be true, so this is a logical AND
										if (isMateLocked) {
											// now he's locked, check his solvency
											if	((pfaSugar[iMateID] > pfaInitialSugar[iMateID]) && (pfaSpice[iMateID] > pfaInitialSpice[iMateID])) {
												// ok, he's a keeper
												// mark "mated" only when mate is fully vetted, that way if lock fails, other potential mates are still screened
												mated = true;
												// note that both locks succeeded
												iTemp = atomicAdd(piLockSuccesses,1);

												// get baby's id
												int iChildID = atomicAdd(&(piPopulation[0]),1);

												// insert baby in the grid
												insert_resident(&(gbwBits.asInt),iAddy,pigResidents,psaX,psaY,psaX[iAgentID],psaY[iAgentID],iChildID);

												// set internal properties of baby
												AgentBitWise abwBaby;
												abwBaby.asBits.age = 0;
												abwBaby.asBits.pad = 0;
												abwBaby.asBits.isLocked = 0;

/*												// baby's sex is random
												abwBaby.asBits.isFemale = buRandoms.asBits.b16;

												// baby's vision and metabolism are inherited from one parent or the other, at random
												if (buRandoms.asBits.b2 == 0) {
													abwBaby.asBits.vision = abwAgentBits.asBits.vision;
												} else {
													abwBaby.asBits.vision = abwMateBits.asBits.vision;
												}
												if (buRandoms.asBits.b3 == 0) {
													abwBaby.asBits.metSugar = abwAgentBits.asBits.metSugar;
												} else {
													abwBaby.asBits.metSugar = abwMateBits.asBits.metSugar;
												}
												if (buRandoms.asBits.b4 == 0) {
													abwBaby.asBits.metSpice = abwAgentBits.asBits.metSpice;
												} else {
													abwBaby.asBits.metSpice = abwMateBits.asBits.metSpice;
												}

												// baby's fertility ages and life expectancy are random (for now)
												abwBaby.asBits.startFertilityAge = buRandoms.asBits.b5+2*buRandoms.asBits.b6;
												abwBaby.asBits.endFertilityAge = buRandoms.asBits.b7+2*buRandoms.asBits.b8+
														4*buRandoms.asBits.b9+8*buRandoms.asBits.b10;
												abwBaby.asBits.deathAge = buRandoms.asBits.b11+2*buRandoms.asBits.b12+
														4*buRandoms.asBits.b13+8*buRandoms.asBits.b14+16*buRandoms.asBits.b15;
												iTemp = atomicExch(&(piaBits[iChildID]),abwBaby.asInt);

												// baby gets all assets each parent has, up to 5 units of each
												fTemp = min(5.0f,pfaSugar[iAgentID]);
												pfaSugar[iChildID] = fTemp;
												pfaSugar[iAgentID] -= fTemp;
												fTemp = min(5.0f,pfaSugar[iMateID]);
												pfaSugar[iChildID] += fTemp;
												pfaSugar[iMateID] -= fTemp;
												fTemp = min(5.0f,pfaSpice[iAgentID]);
												pfaSpice[iChildID] = fTemp;
												pfaSpice[iAgentID] -= fTemp;
												fTemp = min(5.0f,pfaSpice[iMateID]);
												pfaSpice[iChildID] += fTemp;
												pfaSpice[iMateID] -= fTemp;
												pfaInitialSugar[iChildID] = pfaSugar[iChildID];
												pfaInitialSpice[iChildID] = pfaSpice[iChildID];
												// TODO: give both parents memory of child's id for future inheritance
											}
											// unlock mate
											iTemp = atomicExch(&(piaBits[iMateID]),abwMateBits.asInt);
										}
									}
 			}
							}
						}
					}
				} else {
					// if square is already full, indicate an error
					printf("over occupancy %d to x:%d y:%d\n",gbwBits.asBits.occupancy,psaX[iAgentID],psaY[iAgentID]);
				}
				// unlock square and update global occupancy values
				gbwBits.asBits.isLocked = 0;
				iTemp = atomicExch(&(pigBits[iAddy]),gbwBits.asInt);
			}
			// if either lock failed, add the agent to the deferred queue
			if (!isLocLocked || !isMateLocked) {
				iTemp = atomicAdd(piDeferredQueueSize,1);
				piaDeferredQueue[iTemp]=iAgentID;
			}
		}
	}
	return;
}
 */
__global__ void mate_masked(short* psaX, short* psaY, int* piaBits, int* pigBits, int* pigResidents,
		float* pfaSugar, float* pfaSpice, float* pfaInitialSugar, float* pfaInitialSpice,
		int* piaActiveQueue, int* piaDeferredQueue, const int ciActiveQueueSize, int* piPopulation,
		curandState* paStates, int* piDeferredQueueSize, int* piLockSuccesses)
{
	int iAgentID;
	int iMateID;
	int iAddy;
	int iAddyTry;
	int iTemp;
	int iTry;
	bool mated = false;
	bool isLocLocked = false;
	bool isMateLocked = false;
	short sOccTry;
	float fTemp = 0.0f;

	// get the iAgentID from the active agent queue
	int iOffset = threadIdx.x + blockIdx.x*blockDim.x;
	if (iOffset < ciActiveQueueSize) {
		iAgentID = piaActiveQueue[iOffset];

		// live, fertile, solvent female agents only
		if (is_fertile(iAgentID,piaBits,psaX) &&
				(((piaBits[iAgentID])&isFemaleMask)>>isFemaleShift) == 1 &&
				(pfaSugar[iAgentID] > pfaInitialSugar[iAgentID]) &&
				(pfaSpice[iAgentID] > pfaInitialSpice[iAgentID]) ) {
			iAddy = psaX[iAgentID]*GRID_SIZE+psaY[iAgentID];
			// need to have room on the grid for the kid
			isLocLocked = lock_location(iAddy,pigBits);
			if (isLocLocked) {
				iTemp = pigBits[iAddy];
				if ((iTemp&occupancyMask)>>occupancyShift < MAX_OCCUPANCY) {
					// get nearest neighbors
					for (short i = -1; i<= 1; i++) {
						for (short j = -1; j <= 1; j++) {
							iAddyTry = (psaX[iAgentID]+i)*GRID_SIZE+psaY[iAgentID]+j;
							iTry = pigBits[iAddyTry];
							for (sOccTry = 0; sOccTry < (iTry&occupancyMask)>>occupancyShift; sOccTry++) {
								// note that "mated" terminates the search for mates
								if (!mated) {
									// get the potential mate's id
									iMateID = pigResidents[iAddyTry*MAX_OCCUPANCY+sOccTry];
									if (iMateID > INIT_AGENTS) printf ("mate id %d is too large\n",iMateID);
									// vet his internal properties
									/*		if (is_acceptable_mate_masked(iMateID,piaBits,psaX)) {
										// if acceptable, try to lock him
										isMateLocked = lock_agent_masked(iMateID,piaBits);

										// to get to this point isLocLocked must be true, so this is a logical AND
										if (isMateLocked) {
											// now he's locked, check his solvency
											if	((pfaSugar[iMateID] > pfaInitialSugar[iMateID]) && (pfaSpice[iMateID] > pfaInitialSpice[iMateID])) {
												// ok, he's a keeper
												// mark "mated" only when mate is fully vetted, that way if lock fails, other potential mates are still screened
												mated = true;
												// note that both locks succeeded
												iTemp = atomicAdd(piLockSuccesses,1);

												// get baby's id
										/*		int iChildID = atomicAdd(&(piPopulation[0]),1);

												// insert baby in the grid
												insert_resident(&(gbwBits.asInt),iAddy,pigResidents,psaX,psaY,psaX[iAgentID],psaY[iAgentID],iChildID);

												// set internal properties of baby
												AgentBitWise abwBaby;
												abwBaby.asBits.age = 0;
												abwBaby.asBits.pad = 0;
												abwBaby.asBits.isLocked = 0;
												curandState localState = paStates[iOffset];

												// baby's sex is random
												abwBaby.asBits.isFemale = buRandoms.asBits.b16;

												// baby's vision and metabolism are inherited from one parent or the other, at random
												if (buRandoms.asBits.b2 == 0) {
													abwBaby.asBits.vision = abwAgentBits.asBits.vision;
												} else {
													abwBaby.asBits.vision = abwMateBits.asBits.vision;
												}
												if (buRandoms.asBits.b3 == 0) {
													abwBaby.asBits.metSugar = abwAgentBits.asBits.metSugar;
												} else {
													abwBaby.asBits.metSugar = abwMateBits.asBits.metSugar;
												}
												if (buRandoms.asBits.b4 == 0) {
													abwBaby.asBits.metSpice = abwAgentBits.asBits.metSpice;
												} else {
													abwBaby.asBits.metSpice = abwMateBits.asBits.metSpice;
												}

												// baby's fertility ages and life expectancy are random (for now)
												abwBaby.asBits.startFertilityAge = buRandoms.asBits.b5+2*buRandoms.asBits.b6;
												abwBaby.asBits.endFertilityAge = buRandoms.asBits.b7+2*buRandoms.asBits.b8+
														4*buRandoms.asBits.b9+8*buRandoms.asBits.b10;
												abwBaby.asBits.deathAge = buRandoms.asBits.b11+2*buRandoms.asBits.b12+
														4*buRandoms.asBits.b13+8*buRandoms.asBits.b14+16*buRandoms.asBits.b15;
												iTemp = atomicExch(&(piaBits[iChildID]),abwBaby.asInt);

												// baby gets all assets each parent has, up to 5 units of each
												fTemp = min(5.0f,pfaSugar[iAgentID]);
												pfaSugar[iChildID] = fTemp;
												pfaSugar[iAgentID] -= fTemp;
												fTemp = min(5.0f,pfaSugar[iMateID]);
												pfaSugar[iChildID] += fTemp;
												pfaSugar[iMateID] -= fTemp;
												fTemp = min(5.0f,pfaSpice[iAgentID]);
												pfaSpice[iChildID] = fTemp;
												pfaSpice[iAgentID] -= fTemp;
												fTemp = min(5.0f,pfaSpice[iMateID]);
												pfaSpice[iChildID] += fTemp;
												pfaSpice[iMateID] -= fTemp;
												pfaInitialSugar[iChildID] = pfaSugar[iChildID];
												pfaInitialSpice[iChildID] = pfaSpice[iChildID];
												// TODO: give both parents memory of child's id for future inheritance
											}
											// unlock mate
											isMateLocked = isMateLocked && unlock_agent_masked(iMateID,piaBits);
										}
									}
									 */
								}
							}
						}
					}
				} else {
					// if square is already full, indicate an error
					printf("over occupancy %d to x:%d y:%d\n",(iTemp&occupancyMask)>>occupancyShift,psaX[iAgentID],psaY[iAgentID]);
				}
				// unlock square
				unlock_location(iAddy,pigBits);
			}
			// if either lock failed, add the agent to the deferred queue
			if (!isLocLocked || !isMateLocked) {
				iTemp = atomicAdd(piDeferredQueueSize,1);
				piaDeferredQueue[iTemp]=iAgentID;
			}
		}
	}
	return;
}
