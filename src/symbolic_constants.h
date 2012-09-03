/*
 * symbolic_constants.h
 *
 *  Created on: Aug 4, 2011
 *      Author: C. Richard Fisel
 */

#ifndef DIMENSIONS_H_
#define DIMENSIONS_H_

// dimensions
#define NUM_THREADS_PER_BLOCK 1024
#define GRID_SIZE 1024
#define INIT_AGENTS 524288 
#define MAX_AGENTS 2097152
#define MAX_OCCUPANCY 15 // MAX 15!!

// error conditions
#define EXIT_SUCCESS 0
#define OCCUPANCY_OVERFLOW 1
#define OCCUPANCY_UNDERFLOW 2

#define CUDA_CALL(x) do { if((x) != cudaSuccess ) { \
		printf ("Error at %s:%d \n",__FILE__,__LINE__); \
		return EXIT_FAILURE;}} while (0)

#endif /* DIMENSIONS_H_ */
