// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

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

void outputPixel(ivec2 pixel_pos, vec4 color) { imageStore(g_raster_image, pixel_pos, color); }

layout(local_size_x = LSIZE) in;

const float isect_epsilon = 0.0000000001;
const float epsilon = 0.0001;
const float infinity = 1.0 / 0.0; // TODO...

void getTriangleVectors(uint idx, out vec3 tangent, out vec3 normal, out vec3 binormal) {
	vec3 tri0 = g_bvh_tris[idx * 3 + 0].xyz;
	vec3 tri1 = g_bvh_tris[idx * 3 + 1].xyz;
	vec3 tri2 = g_bvh_tris[idx * 3 + 2].xyz;
	tangent = normalize(tri1 - tri0);
	normal = normalize(cross(tangent, tri2 - tri0));
	binormal = cross(normal, tangent);
}

void getBox(uint idx, out vec3 box_min, out vec3 box_max) {
	box_min = vec3(g_bvh_boxes[idx * 6 + 0], g_bvh_boxes[idx * 6 + 1], g_bvh_boxes[idx * 6 + 2]);
	box_max = vec3(g_bvh_boxes[idx * 6 + 3], g_bvh_boxes[idx * 6 + 4], g_bvh_boxes[idx * 6 + 5]);
}

float isectTriangle(uint idx, vec3 origin, vec3 dir) {
	vec3 tri0 = g_bvh_tris[idx * 3 + 0].xyz;
	vec3 tri1 = g_bvh_tris[idx * 3 + 1].xyz;
	vec3 tri2 = g_bvh_tris[idx * 3 + 2].xyz;
	vec3 e1 = tri1 - tri0, e2 = tri2 - tri0;

	// Begin calculating determinant - also used to calculate u parameter
	vec3 vp = cross(dir, e2);
	float det = dot(e1, vp);

	// if determinant is near zero, ray lies in plane of triangle
	if(det > -isect_epsilon && det < isect_epsilon) {
		// TODO: fix this...
		return infinity;
	}

	float inv_det = 1.0 / det;

	// calculate distance from V1 to ray origin
	vec3 vt = origin - tri0;

	// Calculate u parameter and test bound
	float tu = dot(vt, vp) * inv_det;
	// The intersection lies outside of the triangle
	if(tu < 0.0f || tu > 1.0)
		return infinity;

	// Prepare to test v parameter
	vec3 vq = cross(vt, e1);

	// Calculate V parameter and test bound
	float tv = dot(dir, vq) * inv_det;
	// The intersection lies outside of the triangle
	if(tv < 0.0 || tu + tv > 1.0)
		return infinity;

	float t = dot(e2, vq) * inv_det;

	if(t > epsilon)
		return t;

	return infinity;
}

float slabTest(uint node_idx, vec3 origin, vec3 inv_dir) {
	vec3 box_min, box_max;
	getBox(node_idx, box_min, box_max);

	float l1 = inv_dir.x * (box_min.x - origin.x);
	float l2 = inv_dir.x * (box_max.x - origin.x);
	float lmin = min(l1, l2);
	float lmax = max(l1, l2);

	l1 = inv_dir.y * (box_min.y - origin.y);
	l2 = inv_dir.y * (box_max.y - origin.y);
	lmin = max(lmin, min(l1, l2));
	lmax = min(lmax, max(l1, l2));

	l1 = inv_dir.z * (box_min.z - origin.z);
	l2 = inv_dir.z * (box_max.z - origin.z);
	lmin = max(lmin, min(l1, l2));
	lmax = min(lmax, max(l1, l2));

	if(lmin > lmax)
		return infinity;
	return lmin;
}

#define MAX_LEVELS 20

struct TraceResult {
	float dist;
	uint num_iters;
};

void rayTraceBVH(out TraceResult result, vec3 origin, vec3 dir, out int hit_tri_id,
				 bool skip_selected) {
	vec3 idir = vec3(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);

	if(slabTest(0, origin, idir) == infinity) {
		result.dist = infinity;
		result.num_iters = 0;
		return;
	}

	uint stack[MAX_LEVELS], ssize = 1;
	stack[0] = 0;

	uint visited[4], num_visited = 0;
	uint num_iters = 0;
	float out_isect = infinity;

	while(ssize > 0 && num_iters < 256) {
		while(ssize > 0 && num_iters < 256) {
			uint nidx = stack[--ssize];
			uint first = g_bvh_nodes[nidx * 2 + 0];
			uint count = g_bvh_nodes[nidx * 2 + 0];
			num_iters++;

			if((first & 0x80000000) == 0) {
				uint last = first + 1;
				float dist0 = slabTest(first, origin, idir);
				float dist1 = slabTest(last, origin, idir);
				if(dist0 < dist1) { // makes no difference...
					swap(dist0, dist1);
					swap(first, last);
				}

				if(dist0 < out_isect)
					stack[ssize++] = first;
				if(dist1 < out_isect)
					stack[ssize++] = last;
			} else {
				first &= 0x7fffffff;
				visited[num_visited++] = nidx;
				if(num_visited == 4)
					break;
			}
		}

		for(int n = 0; n < num_visited; n++) {
			uint idx = visited[n];
			uint first = g_bvh_nodes[idx * 2 + 0];
			uint count = g_bvh_nodes[idx * 2 + 1];
			first &= 0x7fffffff;
			for(uint i = 0; i < count; i++) {
				if(skip_selected && first + i == hit_tri_id)
					continue;
				float dist = isectTriangle(first + i, origin, dir);
				if(dist < out_isect) {
					out_isect = min(out_isect, dist);
					hit_tri_id = int(first + i);
				}
			}
		}
		num_visited = 0;
	}

	result.dist = out_isect;
	result.num_iters = num_iters;
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

//#define COMPUTE_AO

void traceBin() {
	ivec2 pixel_pos = ivec2(LIX & 31, LIX >> 5) + s_bin_pos;
	vec3 ray_origin, ray_dir;
	getScreenRay(pixel_pos, ray_origin, ray_dir);

	int hit_tri_id = -1;
	TraceResult result;
	rayTraceBVH(result, ray_origin, ray_dir, hit_tri_id, false);
	float ao_value = 1.0;

#ifdef COMPUTE_AO
	if(min_isect != infinity) {
		vec3 tri_vecs[3];
		getTriangleVectors(hit_tri_id, tri_vecs[0], tri_vecs[1], tri_vecs[2]);
		vec3 hit_point = ray_origin + ray_dir * min_isect;

		const int dim_size = 3;
		int hits = 0, total = (dim_size + 1) * (dim_size + 1);

		for(int x = 0; x <= dim_size; x++)
			for(int y = 0; y <= dim_size; y++) {
				vec2 pos = vec2(x, y) * (1.0 / dim_size);
				vec3 hemi = uniformSampleHemisphere(pos);
				vec3 dir = tri_vecs[0] * hemi[0] + tri_vecs[1] * hemi[2] + tri_vecs[2] * hemi[1];
				vec3 origin = hit_point + dir * 0.00001;
				float dist = rayTraceBVH(origin, dir, hit_tri_id, true);
				if(dist < 0.05)
					hits++;
			}

		ao_value = (total - hits) / float(total);
	}
#endif

	vec3 vcolor = vec3(0.0);
	if(result.dist < infinity)
		vcolor = vec3(result.dist * 0.1, result.dist * 0.05, result.dist * 0.01);
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