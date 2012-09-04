			// begin finding best move, searching around current (center) square
			short sXCenter = psaX[iAgentID];
			short sYCenter = psaY[iAgentID];
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
			gbwBits.asInt = pigGridBits[iTemp];
			float fBest = gbwBits.asBits.spice*spiceScale/(gbwBits.asBits.occupancy+0.01f)
				+ gbwBits.asBits.sugar*sugarScale/(gbwBits.asBits.occupancy+0.01f);

			// search a square neighborhood of dimension 2*vision+3 (from 3x3 to 9x9)
			float fTest = 0.0f;
			iTemp = 0;
			short sXTry = 0;
			short sYTry = 0;

			for (short i = sXMin; i <= sXMax; i++) {
				// wraparound
				sXTry = i;
				if (sXTry < 0) sXTry += GRID_SIZE;
				if (sXTry >= GRID_SIZE) sXTry -= GRID_SIZE;

				for (short j = sYMin; j <= sYMax; j++) {
					// wraparound
					sYTry = j;
					if (sYTry < 0) sYTry += GRID_SIZE;
					if (sYTry >= GRID_SIZE) sYTry -= GRID_SIZE;

					// weight target's sugar and spice by need, metabolism, and occupancy
					iTemp = sXTry*GRID_SIZE + sYTry;
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
