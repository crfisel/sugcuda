/*
 * constants.h
 *
 *  Created on: Aug 4, 2011
 *      Author: C. Richard Fisel
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

// dimensions
#define NUM_THREADS_PER_BLOCK 1024
#define GRID_SIZE 1024
#define log2GRID_SIZE 10
#define INIT_AGENTS 1048576
#define MAX_AGENTS 2097152
#define MAX_OCCUPANCY 15 // MAX 15!!

// error conditions
#define EXIT_SUCCESS 0
#define OCCUPANCY_OVERFLOW 1
#define OCCUPANCY_UNDERFLOW 2
#define EXIT_OPCODE_OVERFLOW 3

// bit masks - agent
const int agentLockMask = 				0x80000000;
const int isFemaleMask = 				0x40000000;
const int visionMask = 					0x30000000;
const int metSugarMask =				0x0C000000;
const int metSpiceMask =				0x03000000;
const int ageMask = 					0x00FE0000;
const int startFertilityAgeMask = 		0x00018000;
const int endFertilityAgeMask = 		0x00007800;
const int deathAgeMask =				0x000007C0;

const int isFemaleIncrement = 			0x40000000;
const int visionIncrement =				0x10000000;
const int metSugarIncrement = 			0x04000000;
const int metSpiceIncrement = 			0x01000000;
const int ageIncrement = 				0x00020000;
const int startFertilityAgeIncrement =	0x00008000;
const int endFertilityAgeIncrement =	0x00000800;
const int deathAgeIncrement =			0x00000040;

// bit shifts - agent
const int agentLockShift =				31;
const int isFemaleShift = 				30;
const int visionShift = 				28;
const int metSugarShift =				26;
const int metSpiceShift =				24;
const int ageShift = 					17;
const int startFertilityAgeShift = 		15;
const int endFertilityAgeShift = 		11;
const int deathAgeShift =		 		6;

// bit masks - grid
const int gridLockMask = 				0x80000000;
const int occupancyMask = 				0x78000000;
const int sugarMask = 					0x07800000;
const int spiceMask = 					0x00780000;
const int maxSugarMask =				0x00078000;
const int maxSpiceMask =				0x00007800;

const int occupancyIncrement = 			0x08000000;
const int sugarIncrement = 				0x00800000;
const int spiceIncrement =				0x00080000;
const int maxSugarIncrement = 			0x00008000;
const int maxSpiceIncrement =			0x00000800;

// bit shifts - grid
const int gridLockShift = 				31;
const int occupancyShift = 				27;
const int sugarShift = 					23;
const int spiceShift = 					19;
const int maxSugarShift =				15;
const int maxSpiceShift =				11;

#define CUDA_CALL(x) do { if((x) != cudaSuccess ) { \
		printf ("Error at %s:%d \n",__FILE__,__LINE__); \
		return EXIT_FAILURE;}} while (0)

#endif /* CONSTANTS_H_ */
