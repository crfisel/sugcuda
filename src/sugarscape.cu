/* 
 *  sugarscape.cu
 *
 * *  Created on: Dec 3, 2011
 *      Author: C. Richard Fisel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <limits.h>
#include <curand_kernel.h>
#include "symbolic_constants.h"
#include "rngs.h"
#include "randoms.h"
#include "harvest.h"
#include "eat.h"
#include "age.h"
#include "exercise_locks.h"

int main (int argc , char* argv [])
{
	curandStateXORWOW_t* devAgentStates = 0;
	curandStateXORWOW_t* devGridStates = 0;

	// use the GTX470
	cudaSetDevice(0);

	// Allocate and set up prng states on device
	CUDA_CALL(cudaMalloc((void**)&devAgentStates,NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK*sizeof(curandStateXORWOW_t)));
	CUDA_CALL(cudaMalloc((void**)&devGridStates,GRID_SIZE*GRID_SIZE*sizeof(curandStateXORWOW_t)));
	setup_kernel<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(devAgentStates);
	setup_kernel<<<GRID_SIZE,GRID_SIZE>>>(devGridStates);

	// create agent arrays on device
	// bit storage
	int* piaAgentBits;
	CUDA_CALL(cudaMalloc((void**)&piaAgentBits,MAX_AGENTS*sizeof(int)));

	// sugar holdings
	float* pfaSugar;
	CUDA_CALL(cudaMalloc((void**)&pfaSugar,MAX_AGENTS*sizeof(float)));

	// spice holdings
	float* pfaSpice;
	CUDA_CALL(cudaMalloc((void**)&pfaSpice,MAX_AGENTS*sizeof(float)));

	// current age
	short* psaAge;
	CUDA_CALL(cudaMalloc((void**)&psaAge,MAX_AGENTS*sizeof(short)));

	// initialize agent properties
	// setup dimensions
	int hNumTiles = (INIT_AGENTS+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK/NUM_THREADS_PER_BLOCK;
	int hNumBlocks = (INIT_AGENTS+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;

	// for large numbers of agents, tile the prngs
	if (INIT_AGENTS > NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK) {
		for (int i = 0; i < hNumTiles; i++) {
/*			generate_shorts<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(psaX[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),(GRID_SIZE));
			generate_shorts<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(psaY[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),(GRID_SIZE));
*/			generate_ints<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(piaAgentBits[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]));
			generate_floats<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(pfaSugar[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),30.0f);
			generate_floats<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(pfaSpice[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),30.0f);
			generate_shorts<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(psaAge[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),100);
		}
	} else {
/*		generate_shorts<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,psaX,GRID_SIZE);
		generate_shorts<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,psaY,GRID_SIZE);
*/		generate_ints<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,piaAgentBits);
		generate_floats<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,pfaSugar,30.0f);
		generate_floats<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,pfaSpice,30.0f);
		generate_shorts<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,psaAge,100);
	}

	// initial sugar holdings
	float* pfaInitialSugar;
	CUDA_CALL(cudaMalloc((void**)&pfaInitialSugar,MAX_AGENTS*sizeof(float)));
	CUDA_CALL(cudaMemcpy(pfaInitialSugar,pfaSugar,INIT_AGENTS*sizeof(float),cudaMemcpyDeviceToDevice));

	// initial spice holdings
	float* pfaInitialSpice;
	CUDA_CALL(cudaMalloc((void**)&pfaInitialSpice,MAX_AGENTS*sizeof(float)));
	CUDA_CALL(cudaMemcpy(pfaInitialSpice,pfaSpice,INIT_AGENTS*sizeof(float),cudaMemcpyDeviceToDevice));

	// create grid on device
	// sugar in square
	int* pigGridBits;
	CUDA_CALL(cudaMalloc((void**)&pigGridBits,GRID_SIZE*GRID_SIZE*sizeof(int)));

	initialize_gridbits<<<GRID_SIZE,GRID_SIZE>>>(devGridStates,pigGridBits);

	// current residents in square - initialized to -1's, aka empty
	int* pigResidents;
	CUDA_CALL(cudaMalloc((void**)&pigResidents,GRID_SIZE*GRID_SIZE*MAX_OCCUPANCY*sizeof(int)));
	CUDA_CALL(cudaMemset(pigResidents,0xFFFF,GRID_SIZE*GRID_SIZE*MAX_OCCUPANCY*sizeof(int)));

	// the agent queue
	int* piaQueueA;
	CUDA_CALL(cudaMalloc((void**)&piaQueueA,MAX_AGENTS*sizeof(int)));

	// the deferred queue
	int* piaQueueB;
	CUDA_CALL(cudaMalloc((void**)&piaQueueB,MAX_AGENTS*sizeof(int)));

	// the deferred queue size
	int* piDeferredQueueSize;
	CUDA_CALL(cudaMalloc((void**)&piDeferredQueueSize,sizeof(int)));

	// the successful locks counter
	int* piLockSuccesses;
	CUDA_CALL(cudaMalloc((void**)&piLockSuccesses,sizeof(int)));
	
	// the (dynamic) population counter
	int* piPopulation;
	CUDA_CALL(cudaMalloc((void**)&piPopulation,sizeof(int)));
	int* pihPopulation = (int*) malloc(sizeof(int)); 
	pihPopulation[0] = INIT_AGENTS;
	CUDA_CALL(cudaMemcpy(piPopulation,pihPopulation,sizeof(int),cudaMemcpyHostToDevice));
		
	// position in x
	short* psaX;
	CUDA_CALL(cudaMalloc((void**)&psaX,MAX_AGENTS*sizeof(short)));
	short* psahTemp = (short*) malloc(MAX_AGENTS*sizeof(short));
	for (int i = 0; i < INIT_AGENTS; i++) {
		psahTemp[i] = Random()*(GRID_SIZE-1);
	}
	CUDA_CALL(cudaMemcpy(psaX,psahTemp,INIT_AGENTS*sizeof(short),cudaMemcpyHostToDevice));

	//position in y
	short* psaY;
	CUDA_CALL(cudaMalloc((void**)&psaY,MAX_AGENTS*sizeof(short)));
	// fill iTemp arrays with random numbers and copy to device
	for (int i = 0; i < INIT_AGENTS; i++) {
		psahTemp[i] = Random()*(GRID_SIZE-1);
	}
	CUDA_CALL(cudaMemcpy(psaY,psahTemp,INIT_AGENTS*sizeof(short),cudaMemcpyHostToDevice));

	free(psahTemp);

	cudaDeviceSynchronize();

	// timing
	cudaEvent_t start;
	cudaEvent_t end;
	float elapsed_time;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start,0);

	// count occupancy and store residents

	int status = exercise_locks(COUNT,psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,psaAge,pigGridBits,
		pigResidents,piaQueueA,piPopulation,pihPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses);

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);
	
	printf ("Counting %d agents takes %f milliseconds\n",(int) pihPopulation[0], (float) elapsed_time);

	// time movement
	cudaEventRecord(start,0);

	// do movement
	status = exercise_locks(MOVE,psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,psaAge,pigGridBits, 
		pigResidents,piaQueueA,piPopulation,pihPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses);
	cudaDeviceSynchronize();

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);

	printf ("Moving %d agents takes %f milliseconds\n",(int) pihPopulation[0], (float) elapsed_time);

	// time harvest, meal and aging
	cudaEventRecord(start,0);

	harvest<<<GRID_SIZE,GRID_SIZE>>>(devGridStates,psaX,pfaSugar,pfaSpice,pigGridBits,pigResidents);

	eat<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,piaAgentBits,pfaSugar,pfaSpice);

	age<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaAge);

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);

	printf ("Harvesting %d squares and feeding and aging %d agents takes %f milliseconds\n",(int) GRID_SIZE*GRID_SIZE, (int) pihPopulation[0], (float) elapsed_time);

	// time dying
	cudaEventRecord(start,0);

	status = exercise_locks(DIE,psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,psaAge,pigGridBits, 
		pigResidents,piaQueueA,piPopulation,pihPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses);

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);

	CUDA_CALL(cudaMemcpy(pihPopulation,piPopulation,sizeof(int),cudaMemcpyDeviceToHost));
	printf ("Registering deaths among %d agents takes %f milliseconds\n",(int) pihPopulation[0], (float) elapsed_time);

	// Cleanup 
	CUDA_CALL(cudaFree(psaX));
	CUDA_CALL(cudaFree(psaY));
	CUDA_CALL(cudaFree(piaAgentBits));
	CUDA_CALL(cudaFree(pfaSugar));
	CUDA_CALL(cudaFree(pfaSpice));
	CUDA_CALL(cudaFree(psaAge));
	CUDA_CALL(cudaFree(pfaInitialSugar));
	CUDA_CALL(cudaFree(pfaInitialSpice));
	CUDA_CALL(cudaFree(pigGridBits));
	CUDA_CALL(cudaFree(pigResidents));
	CUDA_CALL(cudaFree(piaQueueA));
	CUDA_CALL(cudaFree(piaQueueB));
	CUDA_CALL(cudaFree(piDeferredQueueSize));
	CUDA_CALL(cudaFree(piLockSuccesses));
	CUDA_CALL(cudaFree(devAgentStates));
	CUDA_CALL(cudaFree(devGridStates));

	return EXIT_SUCCESS;
}
