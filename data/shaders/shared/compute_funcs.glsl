// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#ifndef _COMPUTE_FUNCS_GLSL_
#define _COMPUTE_FUNCS_GLSL_

#include "structures.glsl"

#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#ifdef VENDOR_NVIDIA

#define INCLUSIVE_ADD_STEP(step)                                                                   \
	if(SUBGROUP_SIZE > step) {                                                                     \
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

// TODO: ifdef for SUBGROUP_SIZE == 32?
uint subgroupInclusiveAddFast32(uint accum) {
	uint temp, invocation_id = LIX & 31;
#define INCLUSIVE_ADD_STEP(step)                                                                   \
	if(SUBGROUP_SIZE > step) {                                                                     \
		temp = subgroupShuffleUp(accum, step);                                                     \
		accum += invocation_id >= step ? temp : 0;                                                 \
	}
	INCLUSIVE_ADD_STEP(1);
	INCLUSIVE_ADD_STEP(2);
	INCLUSIVE_ADD_STEP(4);
	INCLUSIVE_ADD_STEP(8);
	INCLUSIVE_ADD_STEP(16);
#undef INCLUSIVE_ADD_STEP
	return accum;
}

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