#ifndef _COMPUTE_FUNCS_GLSL_
#define _COMPUTE_FUNCS_GLSL_

#include "structures.glsl"

#extension GL_KHR_shader_subgroup_shuffle_relative : require

int inclusiveAdd(int accum) {
	int temp;
	uint thread_id = gl_SubgroupInvocationID;
	temp = subgroupShuffleUp(accum, 1), accum += thread_id >= 1 ? temp : 0;
	temp = subgroupShuffleUp(accum, 2), accum += thread_id >= 2 ? temp : 0;
#if WARP_SIZE >= 8
	temp = subgroupShuffleUp(accum, 4), accum += thread_id >= 4 ? temp : 0;
#endif
#if WARP_SIZE >= 16
	temp = subgroupShuffleUp(accum, 8), accum += thread_id >= 8 ? temp : 0;
#endif
#if WARP_SIZE >= 32
	temp = subgroupShuffleUp(accum, 16), accum += thread_id >= 16 ? temp : 0;
#endif
#if WARP_SIZE >= 64
	temp = subgroupShuffleUp(accum, 32), accum += thread_id >= 32 ? temp : 0;
#endif
	return accum;
}

#endif