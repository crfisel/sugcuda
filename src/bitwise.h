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
		unsigned short isLocked : 1;
		unsigned short isFemale : 1;
		unsigned short vision : 2;
		unsigned short metSugar : 2;
		unsigned short metSpice : 2;
		unsigned short age : 7;
		unsigned short startFertilityAge : 2;
		unsigned short endFertilityAge : 4;
		unsigned short deathAge : 5;
		unsigned short pad : 6;
	} asBits;
    	int asInt;
};

union GridBitWise
{
	struct GridBitWiseType
	{
		unsigned short isLocked : 1;
		unsigned short occupancy : 4;
		unsigned short sugar : 4;
		unsigned short spice : 4;
		unsigned short maxSugar : 4;
		unsigned short maxSpice : 4;
		unsigned short pad : 11;
	} asBits;
    	int asInt;
};

union BitUnpacker
{
	struct BitMask
	{
		unsigned short b1 : 1;
		unsigned short b2 : 1;
		unsigned short b3 : 1;
		unsigned short b4 : 1;
		unsigned short b5 : 1;
		unsigned short b6 : 1;
		unsigned short b7 : 1;
		unsigned short b8 : 1;
		unsigned short b9 : 1;
		unsigned short b10 : 1;
		unsigned short b11 : 1;
		unsigned short b12 : 1;
		unsigned short b13 : 1;
		unsigned short b14 : 1;
		unsigned short b15 : 1;
		unsigned short b16 : 1;
		unsigned short b17 : 1;
		unsigned short b18 : 1;
		unsigned short b19 : 1;
		unsigned short b20 : 1;
		unsigned short b21 : 1;
		unsigned short b22 : 1;
		unsigned short b23 : 1;
		unsigned short b24 : 1;
		unsigned short b25 : 1;
		unsigned short b26 : 1;
		unsigned short b27 : 1;
		unsigned short b28 : 1;
		unsigned short b29 : 1;
		unsigned short b30 : 1;
		unsigned short b31 : 1;
		unsigned short b32 : 1;
	} asBits;
    unsigned int asUInt;
};

#endif /* BITWISE_H_ */
