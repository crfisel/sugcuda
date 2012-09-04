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
#include "randoms.h"
#include "harvest.h"
#include "eat.h"
#include "age.h"
#include "exercise_locks.h"
#include "grow_back1.h"
#include "rngs.h"

int main (int argc , char* argv [])
{
	// use the GTX470
	cudaSetDevice(0);

    unsigned int* piaRandoms;
    unsigned int* pigRandoms;
    unsigned int* piahTemp;
    float* pfahTemp;
    unsigned int* pighTemp;

	// seed rngs
	SelectStream(0);
	PutSeed(1234567);

	// initialize agent properties
	int hNumBlocks = (INIT_AGENTS+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK;
	CUDA_CALL(cudaMalloc((void**)&piaRandoms,MAX_AGENTS*sizeof(int)));

	// position in x and y
	short* psaX;
	CUDA_CALL(cudaMalloc((void**)&psaX,MAX_AGENTS*sizeof(short)));
	short* psaY;
	CUDA_CALL(cudaMalloc((void**)&psaY,MAX_AGENTS*sizeof(short)));
	CUDA_CALL(cudaHostAlloc((void**)&piahTemp,MAX_AGENTS*sizeof(int),cudaHostAllocDefault));
	for (int i = 0; i < INIT_AGENTS; i++) {
		piahTemp[i] = Random()*GRID_SIZE*(GRID_SIZE-0.01);
	}
	CUDA_CALL(cudaMemcpy(piaRandoms,piahTemp,INIT_AGENTS*sizeof(int),cudaMemcpyHostToDevice));
	fill_positions<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(piaRandoms,psaX,psaY);	

	// bit storage
	int* piaAgentBits;
	CUDA_CALL(cudaMalloc((void**)&piaAgentBits,MAX_AGENTS*sizeof(int)));
	for (int i = 0; i < INIT_AGENTS; i++) {
		piahTemp[i] = Random_uint();
	}
	CUDA_CALL(cudaMemcpy(piaRandoms,piahTemp,INIT_AGENTS*sizeof(int),cudaMemcpyHostToDevice));
	initialize_agentbits<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(piaRandoms,piaAgentBits);
	
	// sugar holdings
	float* pfaSugar;
	CUDA_CALL(cudaMalloc((void**)&pfaSugar,MAX_AGENTS*sizeof(float)));
	CUDA_CALL(cudaHostAlloc((void**)&pfahTemp,MAX_AGENTS*sizeof(float),cudaHostAllocDefault));
		for (int i = 0; i < INIT_AGENTS; i++) {
		pfahTemp[i] = Random()*30.0f;
	}
	CUDA_CALL(cudaMemcpy(pfaSugar,pfahTemp,INIT_AGENTS*sizeof(int),cudaMemcpyHostToDevice));
	float* pfaInitialSugar;
	CUDA_CALL(cudaMalloc((void**)&pfaInitialSugar,MAX_AGENTS*sizeof(float)));
	CUDA_CALL(cudaMemcpy(pfaInitialSugar,pfaSugar,INIT_AGENTS*sizeof(float),cudaMemcpyDeviceToDevice));

	// spice holdings
	float* pfaSpice;
	CUDA_CALL(cudaMalloc((void**)&pfaSpice,MAX_AGENTS*sizeof(float)));
	for (int i = 0; i < INIT_AGENTS; i++) {
		pfahTemp[i] = Random()*30.0f;
	}
	CUDA_CALL(cudaMemcpy(pfaSpice,pfahTemp,INIT_AGENTS*sizeof(int),cudaMemcpyHostToDevice));
	float* pfaInitialSpice;
	CUDA_CALL(cudaMalloc((void**)&pfaInitialSpice,MAX_AGENTS*sizeof(float)));
	CUDA_CALL(cudaMemcpy(pfaInitialSpice,pfaSpice,INIT_AGENTS*sizeof(float),cudaMemcpyDeviceToDevice));

	// initialize grid properties
	CUDA_CALL(cudaMalloc((void**)&pigRandoms,GRID_SIZE*GRID_SIZE*sizeof(int)));

	// sugar in square
	int* pigGridBits;
	CUDA_CALL(cudaMalloc((void**)&pigGridBits,GRID_SIZE*GRID_SIZE*sizeof(int)));
	CUDA_CALL(cudaHostAlloc((void**)&pighTemp,GRID_SIZE*GRID_SIZE*sizeof(int),cudaHostAllocDefault));
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++) {
		pighTemp[i] = Random()*UINT_MAX;
	}
	CUDA_CALL(cudaMemcpy(pigRandoms,pighTemp,GRID_SIZE*GRID_SIZE*sizeof(int),cudaMemcpyHostToDevice));
	initialize_gridbits<<<GRID_SIZE,GRID_SIZE>>>(pigRandoms,pigGridBits,TILED);

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
	pihPopulation[0] = INIT_AGENTS;
	CUDA_CALL(cudaMemcpy(piPopulation,pihPopulation,sizeof(int),cudaMemcpyHostToDevice));

	// the static agents counter
	int* piStaticAgents;
	CUDA_CALL(cudaMalloc((void**)&piStaticAgents,sizeof(int)));
	
	// and its host-side copy
	int* pihStaticAgents;
	CUDA_CALL(cudaHostAlloc((void**)&pihStaticAgents,sizeof(int),cudaHostAllocDefault));
	
	CUDA_CALL(cudaDeviceSynchronize());

	// timing
	cudaEvent_t start;
	cudaEvent_t end;
	float elapsed_time;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&end));
	CUDA_CALL(cudaEventRecord(start,0));

	// count occupancy and store residents

	int status = exercise_locks(COUNT,psaX,psaY,piaAgentBits,piaRandoms,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,pigGridBits,pigResidents,piaQueueA,
		piPopulation,pihPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses,pihDeferredQueueSize,pihLockSuccesses,piStaticAgents,pihStaticAgents);

	//   end timing
	CUDA_CALL(cudaThreadSynchronize());
	CUDA_CALL(cudaEventSynchronize(end));
	CUDA_CALL(cudaEventRecord(end,0));
	CUDA_CALL(cudaEventSynchronize(end));
	CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));
	
	printf ("Counting %d agents takes %f ms\n",(int) pihPopulation[0], (float) elapsed_time);

	// main loop
		while(pihPopulation[0] > 10) {
		// time movement
		CUDA_CALL(cudaEventRecord(start,0));

		// do movement
		status = exercise_locks(MOVE,psaX,psaY,piaAgentBits,piaRandoms,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,pigGridBits,pigResidents,piaQueueA,
			piPopulation,pihPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses,pihDeferredQueueSize,pihLockSuccesses,piStaticAgents,pihStaticAgents);
		CUDA_CALL(cudaDeviceSynchronize());

		//   end timing
		CUDA_CALL(cudaThreadSynchronize());
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventRecord(end,0));
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));

		printf ("Moving %d agents takes %f ms\n",(int) pihPopulation[0], (float) elapsed_time);

		for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++) {
			pighTemp[i] = Random()*UINT_MAX;
		}
		CUDA_CALL(cudaMemcpy(pigRandoms,pighTemp,GRID_SIZE*GRID_SIZE*sizeof(int),cudaMemcpyHostToDevice));
		harvest<<<GRID_SIZE,GRID_SIZE>>>(pigRandoms,psaX,pfaSugar,pfaSpice,pigGridBits,pigResidents);
	
		eat<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(psaX,piaAgentBits,pfaSugar,pfaSpice);

		age<<<hNumBlocks,NUM_THREADS_PER_BLOCK>>>(piaAgentBits);

		// time mating
		CUDA_CALL(cudaEventRecord(start,0));

		for (int i = 0; i < pihPopulation[0]; i++) {
			piahTemp[i] = Random()*UINT_MAX;
		}
		CUDA_CALL(cudaMemcpy(piaRandoms,piahTemp,pihPopulation[0]*sizeof(int),cudaMemcpyHostToDevice));
		status = exercise_locks(MATE,psaX,psaY,piaAgentBits,piaRandoms,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,pigGridBits,pigResidents,piaQueueA,
			piPopulation,pihPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses,pihDeferredQueueSize,pihLockSuccesses,piStaticAgents,pihStaticAgents);

		//   end timing
		CUDA_CALL(cudaThreadSynchronize());
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventRecord(end,0));
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));

		CUDA_CALL(cudaMemcpy(pihPopulation,piPopulation,sizeof(int),cudaMemcpyDeviceToHost));
		printf ("Mating %d agents takes %f ms\n",(int) pihPopulation[0], (float) elapsed_time);

		// time dying
		CUDA_CALL(cudaEventRecord(start,0));

		status = exercise_locks(DIE,psaX,psaY,piaAgentBits,piaRandoms,pfaSugar,pfaSpice,pfaInitialSugar,pfaInitialSpice,pigGridBits,pigResidents,piaQueueA,
			piPopulation,pihPopulation,piaQueueB,piDeferredQueueSize,piLockSuccesses,pihDeferredQueueSize,pihLockSuccesses,piStaticAgents,pihStaticAgents);

		//   end timing
		CUDA_CALL(cudaThreadSynchronize());
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventRecord(end,0));
		CUDA_CALL(cudaEventSynchronize(end));
		CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));

		CUDA_CALL(cudaMemcpy(pihPopulation,piPopulation,sizeof(int),cudaMemcpyDeviceToHost));
		printf ("Registering deaths among %d agents takes %f ms\n",(int) pihPopulation[0], (float) elapsed_time);

		grow_back1<<<GRID_SIZE,GRID_SIZE>>>(pigGridBits);
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
	CUDA_CALL(cudaFree(piaRandoms));
	CUDA_CALL(cudaFree(pigRandoms));
	CUDA_CALL(cudaFreeHost(pihLockSuccesses));
	CUDA_CALL(cudaFreeHost(pihDeferredQueueSize));
	CUDA_CALL(cudaFreeHost(pihStaticAgents));
	CUDA_CALL(cudaFreeHost(piahTemp));
	CUDA_CALL(cudaFreeHost(pfahTemp));
	CUDA_CALL(cudaFreeHost(pighTemp));

	return EXIT_SUCCESS;
}
