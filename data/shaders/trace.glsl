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

layout(binding = 3) buffer buf03_ { uint g_indices[]; };
layout(binding = 4) buffer buf04_ { float g_vertices[]; };
layout(binding = 5) buffer buf05_ { vec2 g_tex_coords[]; };
layout(binding = 6) uniform accelerationStructureEXT g_accelStruct;

layout(binding = 10) uniform sampler2D albedo_tex;
layout(binding = 11) uniform sampler2D normal_tex;
layout(binding = 12) uniform sampler2D pbr_tex;
layout(binding = 13) uniform sampler2D env_map;

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

uvec3 getTriangleIndices(uint tri_id) {
	return uvec3(g_indices[tri_id * 3 + 0], g_indices[tri_id * 3 + 1], g_indices[tri_id * 3 + 2]);
}

void getTriangleVertices(uvec3 tri_indices, out vec3 tri0, out vec3 tri1, out vec3 tri2) {
	tri0 = getVertex(tri_indices[0]);
	tri1 = getVertex(tri_indices[1]);
	tri2 = getVertex(tri_indices[2]);
}

void getTriangleTexCoords(uvec3 tri_indices, out vec2 tri0, out vec2 tri1, out vec2 tri2) {
	tri0 = g_tex_coords[tri_indices[0]];
	tri1 = g_tex_coords[tri_indices[1]];
	tri2 = g_tex_coords[tri_indices[2]];
}

void getTriangleVectors(in vec3 tri0, in vec3 tri1, in vec3 tri2, out vec3 tangent, out vec3 normal,
						out vec3 binormal) {
	tangent = normalize(tri1 - tri0);
	normal = normalize(cross(tangent, tri2 - tri0));
	binormal = cross(normal, tangent);
}

struct TraceResult {
	float dist;
	vec2 barycentric;
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
		result.barycentric = rayQueryGetIntersectionBarycentricsEXT(rq, true);
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
	dir = normalize(dir);
}

float randomFloat(inout uint rngState) {
	// Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
	rngState = rngState * 747796405 + 1;
	uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
	word = (word >> 22) ^ word;
	return float(word) / 4294967295.0f;
}

float computeAO(inout uint random_seed, uint tri_id, vec3 hit_point) {
	uvec3 tri_indices = getTriangleIndices(tri_id);
	vec3 tri[3];
	getTriangleVertices(tri_indices, tri[0], tri[1], tri[2]);
	vec3 tri_vecs[3];
	getTriangleVectors(tri[0], tri[1], tri[2], tri_vecs[0], tri_vecs[1], tri_vecs[2]);

	const int dim_size = 3;
	int hits = 0, total = (dim_size + 1) * (dim_size + 1);

	for(int x = 0; x <= dim_size; x++)
		for(int y = 0; y <= dim_size; y++) {
			vec2 uv = vec2(randomFloat(random_seed), randomFloat(random_seed));
			vec3 hemi = uniformSampleHemisphere(uv);
			vec3 dir = tri_vecs[0] * hemi[0] + tri_vecs[1] * hemi[2] + tri_vecs[2] * hemi[1];
			vec3 origin = hit_point + dir * 0.001;
			TraceResult ao_hit = rayTraceAS(origin, dir);
			if(ao_hit.dist > 0.01 && ao_hit.dist < 10.0 && ao_hit.tri_id != tri_id)
				hits++;
		}

	return max(0.0, (total - hits) / float(total) - 0.1) * (1.0 / 0.9);
}

vec2 longLat(vec3 normal) {
	// convert normal to longitude and latitude
	float latitude = acos(normal.y) / PI;
	float longitude = (atan(normal.x, normal.z) + PI) / (2.0 * PI);
	return vec2(longitude, latitude);
}

void traceBin() {
	ivec2 pixel_pos = ivec2(LIX & 31, LIX >> 5) + s_bin_pos;
	uint random_seed = pixel_pos.x + (pixel_pos.y << 16);

	vec3 ray_origin, ray_dir;
	getScreenRay(pixel_pos, ray_origin, ray_dir);

	TraceResult result = rayTraceAS(ray_origin, ray_dir);

	vec3 vcolor = vec3(0.0);
	if(result.dist < MAX_ISECT_DIST) {
		vec2 uvs[3];
		uvec3 tri_indices = getTriangleIndices(result.tri_id);
		getTriangleTexCoords(tri_indices, uvs[0], uvs[1], uvs[2]);
		vec3 bary = vec3(1.0 - result.barycentric[0] - result.barycentric[1], result.barycentric);
		vec2 uv = uvs[0] * bary.x + uvs[1] * bary.y + uvs[2] * bary.z;
		vcolor = texture(albedo_tex, uv).rgb;

		vec3 hit_point = ray_origin + ray_dir * result.dist;
		float ao = computeAO(random_seed, result.tri_id, hit_point);
		ao = 0.3 + 0.7 * ao;
		vcolor *= ao;
	} else {
		vec2 tex_coord = longLat(-ray_dir) * vec2(1.0, -1.0);
		vcolor = texture(env_map, tex_coord).rgb * 0.5;
	}

	outputPixel(pixel_pos, SATURATE(vec4(vcolor, 1.0)));
}

void main() {
	if(LIX == 0) {
		s_bin_pos = ivec2(gl_WorkGroupID.xy) * 32;
	}
	barrier();
	traceBin();
}