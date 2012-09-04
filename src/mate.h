/*
 * mate.h
 *
 *  Created on: Dec 4, 2011
 *      Author: C. Richard Fisel
 */

#ifndef MATE_H_
#define MATE_H_

__forceinline__ __device__ bool is_fertile(int iAgentID, int* pbaBits, short* psaX)
{
	AgentBitWise abwBits;
	abwBits.asInt = pbaBits[iAgentID];
	
	bool ydResult = (psaX[iAgentID] > -1) &&
			(abwBits.asBits.age > (abwBits.asBits.startFertilityAge + 12)) &&
			(abwBits.asBits.age < (2*abwBits.asBits.endFertilityAge + 40));
	return ydResult;
}

__forceinline__ __device__ bool is_acceptable_mate(int iMateID, short* psaX, int* pbaBits)
{
	bool acceptable = false;

	// unpack agent bits
	AgentBitWise abwBits;
	abwBits.asInt = pbaBits[iMateID];
	
	// make sure this is not an "empty" placeholder
	acceptable = (iMateID > -1 &&
		// make sure he's male
		abwBits.isFemale == 0 &&
		// and alive
		psaX[iMateID] > -1 &&
		// and fertile
		&& isFertile(iMateID))
	return acceptable;
}
		
__forceinline__ __device__ bool lock_potential_mate(int iMateID, short* psaX, int* pbaBits, AgentBitWise* abwBits)
{
	bool lockSuccess = false;
	
	// unpack agent bits
	abwBits->asInt = pbaBits[iMateID];
		
	// if he's unlocked...
	if (abwBits->asBits.isLocked == 0) {
		// make a copy, but indicating locked
		AgentBitWise abwBitsCopy;
		abwBitsCopy.asInt = abwBits->asInt;
		abwBitsCopy.asBits.isLocked = 1;

		// now lock him if possible
		int iLocked = atomicCAS(&(pbaBits[iMateID]),abwBits->asInt,abwBitsCopy.asInt);

		// test if the lock worked
		if (iLocked == abwBits->asInt) {
			lockSuccess = true;
			abwBits->asInt = abwBitsCopy.asInt;
		}
	}
	return lockSuccess;
}

#endif /* MATE_H_ */
