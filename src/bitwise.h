union BitWise
{
	struct BitWiseType
	{
		unsigned short isLocked : 1;
		unsigned short isFemale : 1;
		unsigned short vision : 2;
		unsigned short metSugar : 2;
		unsigned short metSpice : 2;
		unsigned short startFertilityAge : 2;
		unsigned short endFertilityAge : 4;
		unsigned short deathAge : 5;
		unsigned short pad : 13;
	} asBits;
    	int asInt;
};
