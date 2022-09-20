#ifndef _TIMERS_GLSL_
#define _TIMERS_GLSL_

#include "definitions.glsl"

#ifdef TIMERS_ENABLED

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_ARB_shader_clock : require
#extension GL_KHR_shader_subgroup_basic : require

shared uint s_timers[TIMERS_COUNT];
#define START_TIMER() uint64_t timer0_ = clockARB();
#define UPDATE_TIMER(idx)                                                                          \
	if((gl_LocalInvocationIndex & (gl_SubgroupSize - 1)) == 0) {                                   \
		uint64_t timer = clockARB();                                                               \
		atomicAdd(s_timers[idx], uint(timer - timer0_) >> 4);                                      \
		timer0_ = timer;                                                                           \
	}

#define INIT_TIMERS()                                                                              \
	{                                                                                              \
		if(LIX < TIMERS_COUNT)                                                                     \
			s_timers[LIX] = 0;                                                                     \
	}

#define COMMIT_TIMERS(out_timers)                                                                  \
	{                                                                                              \
		barrier();                                                                                 \
		if(LIX < TIMERS_COUNT)                                                                     \
			atomicAdd(out_timers[LIX], s_timers[LIX]);                                             \
	}

#else

#define START_TIMER()
#define UPDATE_TIMER(idx)

#define INIT_TIMERS()                                                                              \
	{}
#define COMMIT_TIMERS(out_timers)                                                                  \
	{}

#endif

#endif