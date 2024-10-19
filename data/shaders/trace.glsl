// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#version 460

#define LSIZE 1024
#define LSHIFT 10

#include "shared/funcs.glsl"
#include "shared/structures.glsl"

#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : require

coherent layout(std430, binding = 0) buffer info_ {
	PathTracerInfo g_info;
	int g_counts[];
};
layout(binding = 1) uniform config_ { PathTracerConfig u_config; };

layout(binding = 2, rgba8) uniform image2D g_raster_image;

layout(binding = 4) buffer buf04_ { uint g_indices[]; };
layout(binding = 5) buffer buf05_ { float g_vertices[]; };
layout(binding = 6) uniform accelerationStructureEXT g_accelStruct;

layout(binding = 10) uniform sampler2D opaque_texture;
layout(binding = 11) uniform sampler2D transparent_texture;

#include "%shader_debug"
DEBUG_SETUP(1, 12)

shared ivec2 s_bin_pos;

void outputPixel(ivec2 pixel_pos, vec4 color) { imageStore(g_raster_image, pixel_pos, color); }

layout(local_size_x = LSIZE) in;

const float epsilon = 0.0001;
const float infinity = 1.0 / 0.0; // TODO...

vec3 getVertex(uint idx) {
	return vec3(g_vertices[idx * 3 + 0], g_vertices[idx * 3 + 1], g_vertices[idx * 3 + 2]);
}

void getTriangleVertices(uint tri_id, out vec3 tri0, out vec3 tri1, out vec3 tri2) {
	uint idx0 = g_indices[tri_id * 3 + 0], idx1 = g_indices[tri_id * 3 + 1],
		 idx2 = g_indices[tri_id * 3 + 2];
	tri0 = getVertex(idx0);
	tri1 = getVertex(idx1);
	tri2 = getVertex(idx2);
}

void getTriangleVectors(in vec3 tri0, in vec3 tri1, in vec3 tri2, out vec3 tangent, out vec3 normal,
						out vec3 binormal) {
	tangent = normalize(tri1 - tri0);
	normal = normalize(cross(tangent, tri2 - tri0));
	binormal = cross(normal, tangent);
}

struct TraceResult {
	float dist;
	uint num_iters;
	uint tri_id;
};

#define MAX_ISECT_DIST 10000.0
#define INVALID_TRI_ID uint(0xffffffff)

TraceResult rayTraceAS(vec3 origin, vec3 dir) {
	TraceResult result;

	rayQueryEXT rq;
	rayQueryInitializeEXT(rq, g_accelStruct, gl_RayFlagsOpaqueEXT, 0xff, origin, 0.0, dir,
						  MAX_ISECT_DIST);
	result.num_iters = 0;
	while(rayQueryProceedEXT(rq))
		result.num_iters++;
	if(rayQueryGetIntersectionTypeEXT(rq, true) != 0) {
		result.dist = rayQueryGetIntersectionTEXT(rq, true);
		result.tri_id = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
		// rayQueryGetIntersectionBarycentricsEXT
	} else {
		result.dist = MAX_ISECT_DIST;
		result.tri_id = INVALID_TRI_ID;
	}

	return result;
}

vec3 uniformSampleHemisphere(vec2 u) {
	float z = u[0];
	float r = sqrt(max(0.0, 1.0 - z * z));
	float phi = 2 * PI * u[1];
	return vec3(r * cos(phi), r * sin(phi), z);
}

void getScreenRay(ivec2 pixel_pos, out vec3 origin, out vec3 dir) {
	origin = u_config.frustum.ws_origin0.xyz;
	dir = u_config.frustum.ws_dir0.xyz + float(pixel_pos.x) * u_config.frustum.ws_dirx.xyz +
		  float(pixel_pos.y) * u_config.frustum.ws_diry.xyz;
	dir += vec3(0.0000001); // avoiding division by 0
}

float computeAO(uint tri_id, vec3 hit_point) {
	vec3 tri[3];
	getTriangleVertices(tri_id, tri[0], tri[1], tri[2]);
	vec3 tri_vecs[3];
	getTriangleVectors(tri[0], tri[1], tri[2], tri_vecs[0], tri_vecs[1], tri_vecs[2]);

	const int dim_size = 4;
	int hits = 0, total = (dim_size + 1) * (dim_size + 1);

	for(int x = 0; x <= dim_size; x++)
		for(int y = 0; y <= dim_size; y++) {
			vec2 pos = vec2(x, y) * (1.0 / dim_size);
			vec3 hemi = uniformSampleHemisphere(pos);
			vec3 dir = tri_vecs[0] * hemi[0] + tri_vecs[1] * hemi[2] + tri_vecs[2] * hemi[1];
			vec3 origin = hit_point + dir * 0.001;
			TraceResult ao_hit = rayTraceAS(origin, dir);
			if(ao_hit.dist > 0.05 && ao_hit.dist < 5.0 && ao_hit.tri_id != tri_id)
				hits++;
		}

	return max(0.0, (total - hits) / float(total) - 0.1) * (1.0 / 0.9);
}

void traceBin() {
	ivec2 pixel_pos = ivec2(LIX & 31, LIX >> 5) + s_bin_pos;
	vec3 ray_origin, ray_dir;
	getScreenRay(pixel_pos, ray_origin, ray_dir);

	TraceResult result = rayTraceAS(ray_origin, ray_dir);
	float ao_value = 1.0;
	if(result.dist < MAX_ISECT_DIST) {
		vec3 hit_point = ray_origin + ray_dir * result.dist;
		ao_value = computeAO(result.tri_id, hit_point);
	}

	vec3 vcolor = vec3(0.0);
	if(result.dist < MAX_ISECT_DIST) {
		float value = sqrt(result.dist);
		vcolor = vec3(1.0 - value * 0.02, 1.0 - value * 0.05, 1.0 - value * 0.1);
	}
	vcolor *= ao_value;

	outputPixel(pixel_pos, SATURATE(vec4(vcolor, 1.0)));
}

void main() {
	if(LIX == 0) {
		s_bin_pos = ivec2(gl_WorkGroupID.xy) * 32;
	}
	barrier();
	traceBin();
}