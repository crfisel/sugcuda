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
#include "count.h"
#include "randoms.h"
#include "move.h"
#include "harvest.h"
#include "eat.h"
#include "age.h"
#include "die.h"

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
	int hPopulation = NUM_AGENTS;

	// position in x
	short* psaX;
	CUDA_CALL(cudaMalloc((void**)&psaX,hPopulation*sizeof(short)));

	//position in y
	short* psaY;
	CUDA_CALL(cudaMalloc((void**)&psaY,hPopulation*sizeof(short)));

	// bit storage
	int* piaAgentBits;
	CUDA_CALL(cudaMalloc((void**)&piaAgentBits,hPopulation*sizeof(int)));

	// sugar holdings
	float* pfaSugar;
	CUDA_CALL(cudaMalloc((void**)&pfaSugar,hPopulation*sizeof(float)));

	// spice holdings
	float* pfaSpice;
	CUDA_CALL(cudaMalloc((void**)&pfaSpice,hPopulation*sizeof(float)));

	// current age
	short* psaAge;
	CUDA_CALL(cudaMalloc((void**)&psaAge,hPopulation*sizeof(short)));

	// initialize agent properties
	// setup dimensions
	int hNumTiles = (hPopulation+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK/NUM_THREADS_PER_BLOCK;
	int hNumBlocks = (hPopulation+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;

	// for large numbers of agents, tile the prngs
	if (hPopulation > NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK) {
		for (int i = 0; i < hNumTiles; i++) {
			generate_shorts<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(psaX[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),(GRID_SIZE-1));
			generate_shorts<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(psaY[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),(GRID_SIZE-1));
			generate_ints<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(piaAgentBits[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),INT_MAX);
			generate_floats<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(pfaSugar[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),30.0f);
			generate_floats<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(pfaSpice[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),30.0f);
			generate_shorts<<<NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(
					devAgentStates,&(psaAge[i*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK]),100);
		}
	} else {
		generate_shorts<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,psaX,GRID_SIZE-1);
		generate_shorts<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,psaY,GRID_SIZE-1);
		generate_ints<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,piaAgentBits,INT_MAX);
		generate_floats<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,pfaSugar,30.0f);
		generate_floats<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,pfaSpice,30.0f);
		generate_shorts<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(devAgentStates,psaAge,100);
	}

	// initial sugar holdings
	float* pfaInitialSugar;
	CUDA_CALL(cudaMalloc((void**)&pfaInitialSugar,hPopulation*sizeof(float)));
	CUDA_CALL(cudaMemcpy(pfaInitialSugar,pfaSugar,hPopulation*sizeof(float),cudaMemcpyDeviceToDevice));

	// initial spice holdings
	float* pfaInitialSpice;
	CUDA_CALL(cudaMalloc((void**)&pfaInitialSpice,hPopulation*sizeof(float)));
	CUDA_CALL(cudaMemcpy(pfaInitialSpice,pfaSpice,hPopulation*sizeof(float),cudaMemcpyDeviceToDevice));

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
	CUDA_CALL(cudaMalloc((void**)&piaQueueA,hPopulation*sizeof(int)));

	// the deferred queue
	int* piaQueueB;
	CUDA_CALL(cudaMalloc((void**)&piaQueueB,hPopulation*sizeof(int)));

	// the deferred queue size
	int* piDeferredQueueSize;
	CUDA_CALL(cudaMalloc((void**)&piDeferredQueueSize,sizeof(int)));

	// the successful locks counter
	int* piLockSuccesses;
	CUDA_CALL(cudaMalloc((void**)&piLockSuccesses,sizeof(int)));
	
	cudaDeviceSynchronize();

	// timing
	cudaEvent_t start;
	cudaEvent_t end;
	float elapsed_time;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start,0);

	// count occupancy and store residents
	int status = count(psaX,psaY,pigGridBits,pigResidents,piaQueueA,hPopulation,
		piaQueueB,piDeferredQueueSize,piLockSuccesses);

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);
	printf ("Counting %d agents takes %d milliseconds\n",(int) hPopulation, (int) elapsed_time);

	// time movement
	cudaEventRecord(start,0);

	// do movement
	move(psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pigGridBits,pigResidents,
		piaQueueA,hPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses);
	cudaDeviceSynchronize();

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);
	printf ("Moving %d agents takes %d milliseconds\n",(int) hPopulation, (int) elapsed_time);

	// time harvest
	cudaEventRecord(start,0);

	harvest<<<GRID_SIZE,GRID_SIZE>>>(devGridStates,psaX,pfaSugar,pfaSpice,pigGridBits,pigResidents);

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);
	printf ("Harvesting %d grid squares takes %d milliseconds\n",(int) GRID_SIZE*GRID_SIZE, (int) elapsed_time);

	// time meal
	cudaEventRecord(start,0);

	eat<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,piaAgentBits,pfaSugar,pfaSpice);

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);
	printf ("Feeding %d agents takes %d milliseconds\n",(int) hPopulation, (int) elapsed_time);

	// time aging
	cudaEventRecord(start,0);

	age<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaAge);

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);

	printf ("Aging %d agents takes %d milliseconds\n",(int) hPopulation, (int) elapsed_time);

	// time dying
	cudaEventRecord(start,0);

	die(psaX,psaY,piaAgentBits,psaAge,pfaSugar,pfaSpice,pigGridBits,pigResidents,
		piaQueueA,hPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses);

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);
	printf ("Registering deaths among %d agents takes %d milliseconds\n",(int) hPopulation, (int) elapsed_time);

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

	return EXIT_SUCCESS ;
}
