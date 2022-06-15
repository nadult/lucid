// $$include structures

#ifdef ENABLE_TIMERS

shared uint s_timers[TIMERS_COUNT];
#define START_TIMER() uint64_t timer0_ = clockARB();
#define UPDATE_TIMER(idx)                                                                          \
	if((LIX & WARP_MASK) == 0) {                                                                   \
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
