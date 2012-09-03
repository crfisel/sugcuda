/*
 * mate.h
 *
 *  Created on: Dec 4, 2011
 *      Author: C. Richard Fisel
 */

#ifndef MATE_H_
#define MATE_H_

__forceinline__ __device__ bool isFertile(int iAgentID, BitWiseType* pbaBits, short* psaAge)
{
	bool ydResult = (psaX[iAgentID] > -1) &&
			(psaAge[iAgentID] > ((&pbaBits[iAgentID])->startFertilityAge + 12)) &&
			(psaAge[iAgentID] < (2*((&pbaBits[iAgentID])->endFertilityAge) + 40));
	return ydResult;
}


#endif /* MATE_H_ */
