#ifndef _COMPUTE_FUNCS_GLSL_
#define _COMPUTE_FUNCS_GLSL_

#include "structures.glsl"

#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#ifdef VENDOR_NVIDIA

#define INCLUSIVE_ADD_STEP(step)                                                                   \
	if(WARP_SIZE > step) {                                                                         \
		temp = subgroupShuffleUp(accum, step);                                                     \
		accum += gl_SubgroupInvocationID >= step ? temp : 0;                                       \
	}

int subgroupInclusiveAddFast(int accum) {
	int temp;
	INCLUSIVE_ADD_STEP(1);
	INCLUSIVE_ADD_STEP(2);
	INCLUSIVE_ADD_STEP(4);
	INCLUSIVE_ADD_STEP(8);
	INCLUSIVE_ADD_STEP(16);
	INCLUSIVE_ADD_STEP(32);
	return accum;
}

uint subgroupInclusiveAddFast(uint accum) {
	uint temp;
	INCLUSIVE_ADD_STEP(1);
	INCLUSIVE_ADD_STEP(2);
	INCLUSIVE_ADD_STEP(4);
	INCLUSIVE_ADD_STEP(8);
	INCLUSIVE_ADD_STEP(16);
	INCLUSIVE_ADD_STEP(32);
	return accum;
}

#undef INCLUSIVE_ADD_STEP

#else

#define subgroupInclusiveAddFast subgroupInclusiveAdd

#endif

uint subgroupMax_(uint value, int width) {
	if(width >= 2)
		value = max(value, subgroupShuffleXor(value, 1));
	if(width >= 4)
		value = max(value, subgroupShuffleXor(value, 2));
	if(width >= 8)
		value = max(value, subgroupShuffleXor(value, 4));
	if(width >= 16)
		value = max(value, subgroupShuffleXor(value, 8));
	if(width >= 32)
		value = max(value, subgroupShuffleXor(value, 16));
	if(width >= 64)
		value = max(value, subgroupShuffleXor(value, 32));
	return value;
}

#endif