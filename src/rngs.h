/* ----------------------------------------------------------------------- 
 * Name            : rngs.h  (header file for the library file rngs.c) 
 * Author          : Steve Park & Dave Geyer
 * Language        : ANSI C
 * Latest Revision : 09-22-98
 * CUDA adaptation C. Richard Fisel Dec 6, 2011
 * ----------------------------------------------------------------------- 
 */

#if !defined( _RNGS_ )
#define _RNGS_

__device__ double Random(void);
__device__ void   PlantSeeds(long x);
__device__ void   GetSeed(long *x);
__device__ void   PutSeed(long x);
__device__ void   SelectStream(int index);
__global__ void   TestRandom(void);

#endif
