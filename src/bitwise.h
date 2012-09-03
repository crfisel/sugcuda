/*
 * bitwise.h
 *
 *  Created on: Dec 4, 2011
 *      Author: C. Richard Fisel
 */

#ifndef BITWISE_H_
#define BITWISE_H_

union AgentBitWise
{
	struct AgentBitWiseType
	{
		unsigned short isFemale : 1;
		unsigned short vision : 2;
		unsigned short metSugar : 2;
		unsigned short metSpice : 2;
		unsigned short startFertilityAge : 2;
		unsigned short endFertilityAge : 4;
		unsigned short deathAge : 5;
		unsigned short pad : 13;
		unsigned short isLocked : 1;
	} asBits;
    	int asInt;
};

union GridBitWise
{
	struct GridBitWiseType
	{
		unsigned short occupancy : 4;
		unsigned short sugar : 4;
		unsigned short spice : 4;
		unsigned short maxSugar : 4;
		unsigned short maxSpice : 4;
		unsigned short pad : 10;
		unsigned short isLocked : 1;
	} asBits;
    	int asInt;
};

#endif /* BITWISE_H_ */
