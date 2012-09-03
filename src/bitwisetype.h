/*
 * bitwisetype.h
 *
 *  Created on: Jul 27, 2011
 *      Author: C. Richard Fisel
 */

// unpacking structure for small data sets

#ifndef BITWISETYPE_H_
#define BITWISETYPE_H_
	
struct BitWiseType
{
	short isFemale : 1;
	short vision : 2;
	short metSugar : 2;
	short metSpice : 2;
	short startFertilityAge : 2;
	short endFertilityAge : 3;
	short deathAge : 4;
};

#endif /* BITWISETYPE_H_ */
