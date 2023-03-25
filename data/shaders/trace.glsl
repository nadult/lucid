#define LSIZE 1024
#define LSHIFT 10

#include "shared/funcs.glsl"
#include "shared/structures.glsl"

#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_shuffle : require

coherent layout(std430, binding = 0) buffer info_ {
	PathTracerInfo g_info;
	int g_counts[];
};
layout(binding = 1) uniform config_ { PathTracerConfig u_config; };

layout(binding = 2, rgba8) uniform image2D g_raster_image;
layout(binding = 3) buffer buf03_ { uint g_bvh_nodes[]; };
layout(binding = 4) buffer buf04_ { float g_bvh_boxes[]; };
layout(binding = 5) buffer buf05_ { vec4 g_bvh_tris[]; };

layout(binding = 10) uniform sampler2D opaque_texture;
layout(binding = 11) uniform sampler2D transparent_texture;

#include "%shader_debug"
DEBUG_SETUP(1, 12)

shared ivec2 s_bin_pos;

void outputPixel(ivec2 pixel_pos, vec4 color) {
	imageStore(g_raster_image, s_bin_pos + pixel_pos, color);
}

layout(local_size_x = LSIZE) in;

void traceBin() {
	vec4 color = vec4(1.0);
	color.r = float(LIX & 31) * (1.0f / 31.0f);
	color.g = float(LIX >> 5) * (1.0f / 31.0f);
	color.b = 0.0f;
	color.a = 1.0f;

	ivec2 pixel_pos = ivec2(LIX & 31, LIX >> 5);
	outputPixel(pixel_pos, color);
}

void main() {
	if(LIX == 0) {
		s_bin_pos = ivec2(gl_WorkGroupID.xy) * 32;
	}
	barrier();
	traceBin();
}