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
#include "symbolic_constants.h"
#include "common_config.h"
#include "cudpp.h"
#include "randoms.h"
#include "harvest.h"
#include "eat.h"
#include "age.h"
#include "exercise_locks.h"
#include "grow_back1.h"
#include "aggregate.h"

enum ReduceType
{
    REDUCE_INT,
    REDUCE_FLOAT,
    REDUCE_DOUBLE
};

#define MAX_BLOCK_DIM_SIZE 65535

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

int main (int argc , char* argv [])
{
	// use the GTX470
	cudaSetDevice(0);

    unsigned int seed = 9999;   //constant seed
    unsigned int* piaRandoms;
    unsigned int* pigRandoms;

    //initialize the CUDPP config
    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_UINT;
    config.algorithm = CUDPP_RAND_MD5;
    config.options = 0;

    CUDPPHandle randPlan = 0;
    CUDPPResult result;

    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
	if(result != CUDPP_SUCCESS)
	{
		printf("Error initializing CUDPP Library.\n");
		return -1;
	}

	CUDA_CALL(cudaMalloc((void**)&piaRandoms,MAX_AGENTS*sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void**)&pigRandoms,MAX_AGENTS*sizeof(unsigned int)));
	result = cudppPlan(theCudpp,&randPlan,config,MAX_AGENTS,1,0);

	if (CUDPP_SUCCESS != result)
        {
            printf("Error creating CUDPPPlan\n");
            exit(-1);
        }

	cudppRandSeed(randPlan, seed);
	cudppRand(randPlan,piaRandoms,INIT_AGENTS);

// initialize agent properties
	// setup dimensions
	int hNumBlocks = (INIT_AGENTS+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;

	// position in x and y
	short* psaX;
	CUDA_CALL(cudaMalloc((void**)&psaX,MAX_AGENTS*sizeof(short)));
	short* psaY;
	CUDA_CALL(cudaMalloc((void**)&psaY,MAX_AGENTS*sizeof(short)));
	fill_positions<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(piaRandoms,psaX,psaY);	

	// create agent arrays on device
	// bit storage
	int* piaAgentBits;
	CUDA_CALL(cudaMalloc((void**)&piaAgentBits,MAX_AGENTS*sizeof(int)));
	cudppRand(randPlan,piaRandoms,INIT_AGENTS);
	initialize_agentbits<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(piaRandoms,piaAgentBits);
	
	// sugar holdings
	float* pfaSugar;
	CUDA_CALL(cudaMalloc((void**)&pfaSugar,MAX_AGENTS*sizeof(float)));
	cudppRand(randPlan,piaRandoms,INIT_AGENTS);
	initialize_food<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(piaRandoms,pfaSugar,30.0f);	
	float* pfaInitialSugar;
	CUDA_CALL(cudaMalloc((void**)&pfaInitialSugar,MAX_AGENTS*sizeof(float)));
	CUDA_CALL(cudaMemcpy(pfaInitialSugar,pfaSugar,INIT_AGENTS*sizeof(float),cudaMemcpyDeviceToDevice));

	// spice holdings
	float* pfaSpice;
	CUDA_CALL(cudaMalloc((void**)&pfaSpice,MAX_AGENTS*sizeof(float)));
	cudppRand(randPlan,piaRandoms,INIT_AGENTS);
	initialize_food<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(piaRandoms,pfaSpice,30.0f);	
	float* pfaInitialSpice;
	CUDA_CALL(cudaMalloc((void**)&pfaInitialSpice,MAX_AGENTS*sizeof(float)));
	CUDA_CALL(cudaMemcpy(pfaInitialSpice,pfaSpice,INIT_AGENTS*sizeof(float),cudaMemcpyDeviceToDevice));

	// create grid on device
	// sugar in square
	int* pigGridBits;
	CUDA_CALL(cudaMalloc((void**)&pigGridBits,GRID_SIZE*GRID_SIZE*sizeof(int)));
	cudppRand(randPlan,piaRandoms,GRID_SIZE*GRID_SIZE);
	initialize_gridbits<<<GRID_SIZE,GRID_SIZE>>>(pigRandoms,pigGridBits);

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
	
	// and its host-side copy
	int* pihDeferredQueueSize = (int*) malloc(sizeof(int));

	// the successful locks counter
	int* piLockSuccesses;
	CUDA_CALL(cudaMalloc((void**)&piLockSuccesses,sizeof(int)));
	
	// and its host-side copy
	int* pihLockSuccesses = (int*) malloc(sizeof(int));
	
	// the (dynamic) population counter
	int* piPopulation;
	CUDA_CALL(cudaMalloc((void**)&piPopulation,sizeof(int)));
	int* pihPopulation = (int*) malloc(sizeof(int)); 
	pihPopulation[0] = INIT_AGENTS;
	CUDA_CALL(cudaMemcpy(piPopulation,pihPopulation,sizeof(int),cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();

	// timing
	cudaEvent_t start;
	cudaEvent_t end;
	float elapsed_time;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start,0);

	// count occupancy and store residents

	int status = exercise_locks(COUNT,psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pigGridBits,pigResidents,piaQueueA,
		piPopulation,pihPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses,pihDeferredQueueSize,pihLockSuccesses);

	//   end timing
	cudaThreadSynchronize();
	cudaEventSynchronize(end);
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);
	
	printf ("Counting %d agents takes %f milliseconds\n",(int) pihPopulation[0], (float) elapsed_time);

	// main loop
		while(pihPopulation[0] > 10) {
		// time movement
		cudaEventRecord(start,0);

		// do movement
		status = exercise_locks(MOVE,psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pigGridBits,pigResidents,piaQueueA,
			piPopulation,pihPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses,pihDeferredQueueSize,pihLockSuccesses);
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

		cudppRand(randPlan,pigRandoms,GRID_SIZE*GRID_SIZE);
		harvest<<<GRID_SIZE,GRID_SIZE>>>(pigRandoms,psaX,pfaSugar,pfaSpice,pigGridBits,pigResidents);
	
		eat<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,piaAgentBits,pfaSugar,pfaSpice);

		age<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(piaAgentBits);

		//   end timing
		cudaThreadSynchronize();
		cudaEventSynchronize(end);
		cudaEventRecord(end,0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&elapsed_time, start, end);

		printf ("Harvesting %d squares and feeding and aging %d agents takes %f milliseconds\n",(int) GRID_SIZE*GRID_SIZE, (int) pihPopulation[0], (float) elapsed_time);

		// time dying
		cudaEventRecord(start,0);

		status = exercise_locks(DIE,psaX,psaY,piaAgentBits,pfaSugar,pfaSpice,pigGridBits,pigResidents,piaQueueA,
			piPopulation,pihPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses,pihDeferredQueueSize,pihLockSuccesses);

		//   end timing
		cudaThreadSynchronize();
		cudaEventSynchronize(end);
		cudaEventRecord(end,0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&elapsed_time, start, end);

		CUDA_CALL(cudaMemcpy(pihPopulation,piPopulation,sizeof(int),cudaMemcpyDeviceToHost));
		printf ("Registering deaths among %d agents takes %f milliseconds\n",(int) pihPopulation[0], (float) elapsed_time);

		// time regrowth
		cudaEventRecord(start,0);

		grow_back1<<<GRID_SIZE,GRID_SIZE>>>(pigGridBits);

		//   end timing
		cudaThreadSynchronize();
		cudaEventSynchronize(end);
		cudaEventRecord(end,0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&elapsed_time, start, end);
		cudaDeviceSynchronize();
		printf ("Growing sugar and spice on %d squares takes %f milliseconds\n",(int) GRID_SIZE*GRID_SIZE, (float) elapsed_time);
	}

	// Cleanup 
	CUDA_CALL(cudaFree(psaX));
	CUDA_CALL(cudaFree(psaY));
	CUDA_CALL(cudaFree(piaAgentBits));
	CUDA_CALL(cudaFree(pfaSugar));
	CUDA_CALL(cudaFree(pfaSpice));
	CUDA_CALL(cudaFree(pfaInitialSugar));
	CUDA_CALL(cudaFree(pfaInitialSpice));
	CUDA_CALL(cudaFree(pigGridBits));
	CUDA_CALL(cudaFree(pigResidents));
	CUDA_CALL(cudaFree(piaQueueA));
	CUDA_CALL(cudaFree(piaQueueB));
	CUDA_CALL(cudaFree(piDeferredQueueSize));
	CUDA_CALL(cudaFree(piLockSuccesses));
	free(pihLockSuccesses);
	free(pihDeferredQueueSize);

	CUDA_CALL(cudaFree(piaRandoms));
	CUDA_CALL(cudaFree(pigRandoms));

	result = cudppDestroyPlan(randPlan);
	if (CUDPP_SUCCESS != result) {
			printf("Error destroying CUDPPPlan\n");
			exit(-1);
	}
	
	result = cudppDestroy(theCudpp);
	if (CUDPP_SUCCESS != result) {
		printf("Error shutting down CUDPP Library.\n");
		exit(-1);
	}

	return EXIT_SUCCESS;
}
