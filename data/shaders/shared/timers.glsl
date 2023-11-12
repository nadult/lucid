// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#ifndef _TIMERS_GLSL_
#define _TIMERS_GLSL_

#include "definitions.glsl"

#ifdef TIMERS_ENABLED

#extension GL_ARB_shader_clock : require
#extension GL_KHR_shader_subgroup_basic : require

shared uint s_timers[TIMERS_COUNT];
#define START_TIMER() uvec2 timer0_ = clock2x32ARB();
// TODO: for now we're just ignoring high bits; Maybe we shouldn't do that?
#define UPDATE_TIMER(idx)                                                                          \
	if(gl_SubgroupInvocationID == 0) {                                                             \
		uvec2 timer = clock2x32ARB();                                                              \
		uint low = timer.x - timer0_.x;                                                            \
		uint high = timer.y - timer0_.y;                                                           \
		if(low > timer.x)                                                                          \
			high--;                                                                                \
		atomicAdd(s_timers[idx], uint(low) >> 4);                                                  \
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