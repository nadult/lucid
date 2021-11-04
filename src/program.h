#pragma once

#include "lucid_base.h"
#include <fwk/gfx/shader_combiner.h>

DEFINE_ENUM(ProgramOpt, debug);
using ProgramOpts = EnumFlags<ProgramOpt>;

// Wrapper around GlProgram which helps with handling shader pieces
class Program {
  public:
	Program() {}
	FWK_MOVABLE_CLASS(Program);

	using Opts = ProgramOpts;

	static Ex<Program> make(Str name, string defs, vector<string> locations, Opts = none);
	static Ex<Program> makeCompute(Str name, string defs, Opts = none);

	template <class Arg,
			  class Ret = decltype(DECLVAL(PProgram &)[std::forward<Arg>(DECLVAL(Arg &&))])>
	Ret operator[](Arg &&arg) const {
		return m_ref[std::forward<Arg>(arg)];
	}

	void use();

	explicit operator bool() const { return !!m_ref; }
	PProgram glProgram() const { return m_ref; }
	const ShaderPieceSet &pieces() const { return m_pieces; }
	Opts opts() const { return m_opts; }

	// ------------------------------------------------------------------------------------------
	// ---  Functions for different shader pieces; ----------------------------------------------
	// TODO: these functions shouldn't be here

	void setViewport(const Camera &, int2 viewport_size);

	void setRect(FRect rect_ndc);
	void setFullscreenRect();
	void setShadows(Matrix4, bool is_enabled);
	void setFrustum(const Camera &);

	CSpan<Pair<string, int>> sourceRanges() const { return m_source_ranges; }

  private:
	Program(PProgram, ShaderPieceSet, Opts, vector<pair<string, int>>);

	PProgram m_ref;
	ShaderPieceSet m_pieces;
	Opts m_opts;
	vector<pair<string, int>> m_source_ranges;
};
