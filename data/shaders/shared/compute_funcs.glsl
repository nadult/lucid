#ifndef _COMPUTE_FUNCS_GLSL_
#define _COMPUTE_FUNCS_GLSL_

#include "structures.glsl"

#extension GL_KHR_shader_subgroup_shuffle_relative : require

#define INCLUSIVE_ADD_STEP(step)                                                                   \
	if(WARP_SIZE > step) {                                                                         \
		temp = subgroupShuffleUp(accum, step);                                                     \
		accum += gl_SubgroupInvocationID >= step ? temp : 0;                                       \
	}

// These functions are faster than subgroupInclusiveAdd...
int inclusiveAdd(int accum) {
	int temp;
	INCLUSIVE_ADD_STEP(1);
	INCLUSIVE_ADD_STEP(2);
	INCLUSIVE_ADD_STEP(4);
	INCLUSIVE_ADD_STEP(8);
	INCLUSIVE_ADD_STEP(16);
	INCLUSIVE_ADD_STEP(32);
	INCLUSIVE_ADD_STEP(64);
	return accum;
}

uint inclusiveAdd(uint accum) {
	uint temp;
	INCLUSIVE_ADD_STEP(1);
	INCLUSIVE_ADD_STEP(2);
	INCLUSIVE_ADD_STEP(4);
	INCLUSIVE_ADD_STEP(8);
	INCLUSIVE_ADD_STEP(16);
	INCLUSIVE_ADD_STEP(32);
	INCLUSIVE_ADD_STEP(64);
	return accum;
}

#undef INCLUSIVE_ADD_STEP

#endif