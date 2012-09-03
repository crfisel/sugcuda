union AgentBitWise
{
	struct AgentBitWiseType
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

union GridBitWise
{
	struct GridBitWiseType
	{
		unsigned short isLocked : 1;
		unsigned short occupancy : 7;
		unsigned short sugar : 4;
		unsigned short spice : 4;
		unsigned short maxSugar : 4;
		unsigned short maxSpice : 4;
		unsigned short pad : 8;
	} asBits;
    	int asInt;
};
