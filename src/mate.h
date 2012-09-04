/*
 * mate.h
 *
 *  Created on: Dec 4, 2011
 *      Author: C. Richard Fisel
 */

#ifndef MATE_H_
#define MATE_H_

#include <curand.h>
#include <curand_kernel.h>
//__noinline__
__device__ bool is_fertile_masked(int iAgentID, int* pbaAgentBits, short* psaX);
//__noinline__
__device__ bool is_acceptable_mate_masked(int iMateID, int* pbaAgentBits, short* psaX);
//__noinline__
__device__ bool lock_potential_mate_masked(int iMateID, int* pbaAgentBits);
//__noinline__ __device__ bool lock_potential_mate(int iMateID, short* psaX, int* pbaBits, AgentBitWise* abwBits);
//__global__ void mate_once(short* , short* , int* , unsigned int* ,
//		float* , float* , float* , float* , int* , int* , int* , const int , int* , int* , int* , int* );
__global__ void mate_masked(curandState* , short* , short* , int* ,
		float* , float* , float* , float* , int* , int* , int* , const int , int* , int* , int* , int* );



#endif /* MATE_H_ */
