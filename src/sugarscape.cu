/* 
 *  sugarscape.cu
 *
 * *  Created on: Dec 3, 2011
 *      Author: C. Richard Fisel
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <limits.h>
#include "constants.h"
#include "randoms.h"
#include "harvest.h"
#include "eat.h"
#include "age.h"
#include "exercise_locks.h"
#include "grow_back1.h"

int main (int argc , char* argv [])
{
	// use the GTX470
	CUDA_CALL(cudaSetDevice(0));

	int ihMaxBlocks = (MAX_AGENTS+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
	int ihInitBlocks = (INIT_AGENTS+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;

	// set up grid-oriented states
	curandState* pgStates = 0;
	CUDA_CALL(cudaMalloc((void **)&pgStates,GRID_SIZE*GRID_SIZE*sizeof(curandState)));
	setup_kernel<<<GRID_SIZE,GRID_SIZE>>>(pgStates);

	// set up agent-oriented states
	curandState* paStates = 0;
	CUDA_CALL(cudaMalloc((void **)&paStates,2*MAX_AGENTS*sizeof(curandState)));
	setup_kernel<<<ihMaxBlocks,NUM_THREADS_PER_BLOCK>>>(paStates);

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
	int* pihDeferredQueueSize;
	CUDA_CALL(cudaHostAlloc((void**)&pihDeferredQueueSize,sizeof(int),cudaHostAllocDefault));

	// the successful locks counter
	int* piLockSuccesses;
	CUDA_CALL(cudaMalloc((void**)&piLockSuccesses,sizeof(int)));

	// and its host-side copy
	int* pihLockSuccesses;
	CUDA_CALL(cudaHostAlloc((void**)&pihLockSuccesses,sizeof(int),cudaHostAllocDefault));

	// the (dynamic) population counter
	int* piPopulation;
	CUDA_CALL(cudaMalloc((void**)&piPopulation,sizeof(int)));

	// and its host-side copy
	int* pihPopulation;
	CUDA_CALL(cudaHostAlloc((void**)&pihPopulation,sizeof(int),cudaHostAllocDefault));

	// the static agents counter
	int* piStaticAgents;
	CUDA_CALL(cudaMalloc((void**)&piStaticAgents,sizeof(int)));

	// and its host-side copy
	int* pihStaticAgents;
	CUDA_CALL(cudaHostAlloc((void**)&pihStaticAgents,sizeof(int),cudaHostAllocDefault));

	// initialize agent properties
	// position in x and y
	short* psaX;
	CUDA_CALL(cudaMalloc((void**)&psaX,MAX_AGENTS*sizeof(short)));
	short* psaY;
	CUDA_CALL(cudaMalloc((void**)&psaY,MAX_AGENTS*sizeof(short)));

	// bit storage
	int* piaBits;
	CUDA_CALL(cudaMalloc((void**)&piaBits,MAX_AGENTS*sizeof(int)));

	// sugar and spice holdings
	float* pfaSugar;
	CUDA_CALL(cudaMalloc((void**)&pfaSugar,MAX_AGENTS*sizeof(float)));
	float* pfaSpice;
	CUDA_CALL(cudaMalloc((void**)&pfaSpice,MAX_AGENTS*sizeof(float)));

	// initial sugar and spice holdings
	float* pfaInitialSugar;
	CUDA_CALL(cudaMalloc((void**)&pfaInitialSugar,MAX_AGENTS*sizeof(float)));
	float* pfaInitialSpice;
	CUDA_CALL(cudaMalloc((void**)&pfaInitialSpice,MAX_AGENTS*sizeof(float)));

	// initialize grid properties
	// sugar and spice in square
	int* pigBits;
	CUDA_CALL(cudaMalloc((void**)&pigBits,GRID_SIZE*GRID_SIZE*sizeof(int)));

	pihPopulation[0] = INIT_AGENTS;
	CUDA_CALL(cudaMemcpy(piPopulation,pihPopulation,sizeof(int),cudaMemcpyHostToDevice));

	CUDA_CALL(cudaDeviceSynchronize());

	fill_positions<<<ihInitBlocks,NUM_THREADS_PER_BLOCK>>>(paStates,psaX,psaY,GRID_SIZE);

	initialize_agentbits<<<ihInitBlocks,NUM_THREADS_PER_BLOCK>>>(paStates,piaBits);

	initialize_food<<<ihInitBlocks,NUM_THREADS_PER_BLOCK>>>(pfaSugar,pfaSpice,paStates,30.0f);
	CUDA_CALL(cudaMemcpy(pfaInitialSugar,pfaSugar,INIT_AGENTS*sizeof(float),cudaMemcpyDeviceToDevice));
	CUDA_CALL(cudaMemcpy(pfaInitialSpice,pfaSpice,INIT_AGENTS*sizeof(float),cudaMemcpyDeviceToDevice));

	initialize_gridbits<<<GRID_SIZE,GRID_SIZE>>>(pgStates,pigBits,TILED);

	CUDA_CALL(cudaDeviceSynchronize());

	// timing
	cudaEvent_t start;
	cudaEvent_t end;
	float elapsed_time;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&end));
	CUDA_CALL(cudaEventRecord(start,0));

	// count occupancy and store residents

	int status = exercise_locks(COUNT,psaX,psaY,piaBits,pigBits,pigResidents,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,piaQueueA,piaQueueB,
			piPopulation,pihPopulation,piDeferredQueueSize,pihDeferredQueueSize,paStates,piLockSuccesses,pihLockSuccesses,piStaticAgents,pihStaticAgents);

	//   end timing
	CUDA_CALL(cudaThreadSynchronize());
	CUDA_CALL(cudaEventSynchronize(end));
	CUDA_CALL(cudaEventRecord(end,0));
	CUDA_CALL(cudaEventSynchronize(end));
	CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));

	printf ("Counting %d agents takes %f milliseconds\n",(int) pihPopulation[0], (float) elapsed_time);

	// main loop
	while(pihPopulation[0] > 10) {

		int ihCurrBlocks = (pihPopulation[0]+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;

		// time movement
		CUDA_CALL(cudaEventRecord(start,0));

		// do movement
		status = exercise_locks(MOVE,psaX,psaY,piaBits,pigBits,pigResidents,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,piaQueueA,piaQueueB,
				piPopulation,pihPopulation,piDeferredQueueSize,pihDeferredQueueSize,paStates,piLockSuccesses,pihLockSuccesses,piStaticAgents,pihStaticAgents);
		CUDA_CALL(cudaDeviceSynchronize());

		//   end timing
		CUDA_CALL(cudaThreadSynchronize());
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventRecord(end,0));
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));

		printf ("Moving %d agents takes %f milliseconds\n",(int) pihPopulation[0], (float) elapsed_time);

		// time harvest, meal and aging
		CUDA_CALL(cudaEventRecord(start,0));

		harvest<<<GRID_SIZE,GRID_SIZE>>>(psaX,pigBits,pigResidents,pfaSugar,pfaSpice,pgStates);

		CUDA_CALL(cudaDeviceSynchronize());

		eat<<<ihCurrBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,piaBits,pfaSugar,pfaSpice);

		age<<<ihCurrBlocks,NUM_THREADS_PER_BLOCK>>>(piaBits);

		//   end timing
		CUDA_CALL(cudaThreadSynchronize());
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventRecord(end,0));
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));

		printf ("Harvesting %d squares and feeding and aging %d agents takes %f milliseconds\n",(int) GRID_SIZE*GRID_SIZE, (int) pihPopulation[0], (float) elapsed_time);

/*		// time mating
		CUDA_CALL(cudaEventRecord(start,0));

		status = exercise_locks(MATE,psaX,psaY,piaBits,pigBits,pigResidents,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,piaQueueA,piaQueueB,
				piPopulation,pihPopulation,piDeferredQueueSize,pihDeferredQueueSize,paStates,piLockSuccesses,pihLockSuccesses,piStaticAgents,pihStaticAgents);

		//   end timing
		CUDA_CALL(cudaThreadSynchronize());
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventRecord(end,0));
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));

		CUDA_CALL(cudaMemcpy(pihPopulation,piPopulation,sizeof(int),cudaMemcpyDeviceToHost));
		printf ("Mating %d agents takes %f ms\n",(int) pihPopulation[0], (float) elapsed_time);
*/

		// time dying
		CUDA_CALL(cudaEventRecord(start,0));

		status = exercise_locks(DIE,psaX,psaY,piaBits,pigBits,pigResidents,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,piaQueueA,piaQueueB,
				piPopulation,pihPopulation,piDeferredQueueSize,pihDeferredQueueSize,paStates,piLockSuccesses,pihLockSuccesses,piStaticAgents,pihStaticAgents);

		//   end timing
		CUDA_CALL(cudaThreadSynchronize());
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventRecord(end,0));
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));

		CUDA_CALL(cudaMemcpy(pihPopulation,piPopulation,sizeof(int),cudaMemcpyDeviceToHost));
		printf ("Registering deaths among %d agents takes %f milliseconds\n",(int) pihPopulation[0], (float) elapsed_time);

		// time regrowth
		CUDA_CALL(cudaEventRecord(start,0));

		grow_back1<<<GRID_SIZE,GRID_SIZE>>>(pigBits);

		//   end timing
		CUDA_CALL(cudaThreadSynchronize());
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventRecord(end,0));
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));
		CUDA_CALL(cudaDeviceSynchronize());
		printf ("Growing sugar and spice on %d squares takes %f milliseconds\n",(int) GRID_SIZE*GRID_SIZE, (float) elapsed_time);
	}

	// Cleanup 
	CUDA_CALL(cudaFree(psaX));
	CUDA_CALL(cudaFree(psaY));
	CUDA_CALL(cudaFree(piaBits));
	CUDA_CALL(cudaFree(pfaSugar));
	CUDA_CALL(cudaFree(pfaSpice));
	CUDA_CALL(cudaFree(pfaInitialSugar));
	CUDA_CALL(cudaFree(pfaInitialSpice));
	CUDA_CALL(cudaFree(pigBits));
	CUDA_CALL(cudaFree(pigResidents));
	CUDA_CALL(cudaFree(piaQueueA));
	CUDA_CALL(cudaFree(piaQueueB));
	CUDA_CALL(cudaFree(piDeferredQueueSize));
	CUDA_CALL(cudaFree(piLockSuccesses));
	CUDA_CALL(cudaFreeHost(pihLockSuccesses));
	CUDA_CALL(cudaFreeHost(pihDeferredQueueSize));
	CUDA_CALL(cudaFreeHost(pihStaticAgents));
	CUDA_CALL(cudaFree(paStates));
	CUDA_CALL(cudaFree(pgStates));

	return EXIT_SUCCESS;
}
