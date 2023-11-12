// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#ifndef _FUNCS_GLSL_
#define _FUNCS_GLSL_

#include "structures.glsl"

#define PI 3.14159265359

#define SATURATE(val) clamp(val, 0.0, 1.0)

// decode/encode source: http://aras-p.info/texts/CompactNormalStorage.html
// TODO: this encoding is wrong for z == -1.0
vec3 decodeNormal(vec2 enc) {
	vec2 fenc = enc * 4.0 - 2.0;
	float f = dot(fenc, fenc);
	float g = sqrt(1.0 - f / 4.0);
	return vec3(fenc * g, 1.0 - f * 0.5);
}

vec2 encodeNormal(vec3 n) {
	float p = sqrt(n.z * 8.0 + 8.0);
	return n.xy / p + 0.5;
}

// TODO: use whole range
uint encodeNormalUint(vec3 n) {
	uint x = uint(512.0 + n.x * 511.0) & 0x3ffu;
	uint y = uint(512.0 + n.y * 511.0) & 0x3ffu;
	uint z = uint(512.0 + n.z * 511.0) & 0x3ffu;
	return x | (y << 10) | (z << 20);
}

uvec2 encodeAABB64(uvec4 aabb) {
	return uvec2(aabb[0] | (aabb[1] << 16), aabb[2] | (aabb[3] << 16));
}

uvec4 decodeAABB64(uvec2 aabb) {
	return uvec4(aabb[0] & 0xffffu, aabb[0] >> 16, aabb[1] & 0xffffu, aabb[1] >> 16);
}

uint encodeAABB32(uvec4 aabb) {
	return ((aabb[0] & 0xffu) << 0) | ((aabb[1] & 0xffu) << 8) | ((aabb[2] & 0xffu) << 16) |
		   ((aabb[3] & 0xffu) << 24);
}

ivec4 decodeAABB32(uint aabb) {
	return ivec4(aabb & 0xffu, (aabb >> 8) & 0xffu, (aabb >> 16) & 0xffu, aabb >> 24);
}

uint encodeAABB28(uvec4 aabb) {
	return ((aabb[0] & 0x7fu) << 0) | ((aabb[1] & 0x7fu) << 7) | ((aabb[2] & 0x7fu) << 14) |
		   ((aabb[3] & 0x7fu) << 21);
}

ivec4 decodeAABB28(uint aabb) {
	return ivec4(aabb & 0x7fu, (aabb >> 7) & 0x7fu, (aabb >> 14) & 0x7fu, (aabb >> 21) & 0x7fu);
}

vec3 decodeNormalUint(uint n) {
	float x = (float((n >> 0) & 0x3ffu) - 512.0) * (1.0 / 511.0);
	float y = (float((n >> 10) & 0x3ffu) - 512.0) * (1.0 / 511.0);
	float z = (float((n >> 20) & 0x3ffu) - 512.0) * (1.0 / 511.0);
	return vec3(x, y, z); // TODO: normalize ?
}

float decodeFloat3(vec3 xyz) { return dot(xyz, vec3(1.0, 1.0 / 255.0, 1.0 / 65025.0)); }

vec3 encodeFloat3(float v) {
	vec4 enc = vec4(1.0, 255.0, 65025.0, 160581375.0) * v;
	enc = fract(enc);
	enc -= enc.yzww * vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0);
	return enc.xyz;
}

float decodeFloat2(vec2 xy) { return dot(xy, vec2(1.0, 1.0 / 255.0)) * (1.0 / 0.99); }

vec2 encodeFloat2(float v) {
	vec2 enc = vec2(1.0, 255.0) * (v * 0.99);
	enc = fract(enc);
	enc.x -= enc.y * 1.0 / 255.0;
	return enc.xy;
}

vec4 encodeInt4(uint v) {
	vec4 enc;
	enc.x = float(v & 0xffu);
	enc.y = float((v & 0xff00u) >> 8);
	enc.z = float((v & 0xff0000u) >> 16);
	enc.w = float((v & 0xff000000u) >> 24);
	return enc * (1.0 / 255.0);
}

uint decodeInt4(vec4 v) {
	v *= 255.0f;
	return uint(v.x) | (uint(v.y) << 8) | (uint(v.z) << 16) | (uint(v.w) << 24);
}

vec4 decodeRGBA8(uint icolor) {
	return vec4(float((icolor >> 0) & 0xffu), float((icolor >> 8) & 0xffu),
				float((icolor >> 16) & 0xffu), float((icolor >> 24) & 0xffu)) *
		   (1.0 / 255.0);
}

vec3 decodeRGB8(uint icolor) {
	return vec3(float(icolor & 0xffu), float((icolor >> 8) & 0xffu),
				float((icolor >> 16) & 0xffu)) *
		   (1.0 / 255.0);
}

vec3 decodeRGB10(uint icolor) {
	return vec3(float(icolor & 0x7ffu) * (1.0 / 2047.0),
				float((icolor >> 11) & 0x7ffu) * (1.0 / 2047.0),
				float((icolor >> 22) & 0x3ffu) * (1.0 / 1023.0));
}

uint encodeRGBA8(vec4 col) {
	return (uint(col.r * 255.0)) | (uint(col.g * 255.0) << 8) | (uint(col.b * 255.0) << 16) |
		   ((uint(col.a * 255.0)) << 24);
}

uint encodeRGB8(vec3 col) {
	return (uint(col.r * 255.0)) | (uint(col.g * 255.0) << 8) | (uint(col.b * 255.0) << 16);
}

uint encodeRGB10(vec3 col) {
	return (uint(col.r * 2047.0)) | (uint(col.g * 2047.0) << 11) | (uint(col.b * 1023.0) << 22);
}

uint tintColor(uint enc_color, vec3 tint, float strength) {
	vec3 color = decodeRGB8(enc_color);
	color = color * (1.0 - strength) + tint * strength;
	return encodeRGB8(SATURATE(color));
}

uvec2 encodeCD(vec4 color, float depth) {
	depth = float(0xffffff) / (1.0 + depth);
	uint enc_col = (uint(clamp(color.r, 0.0, 1.0) * 2047.0)) |
				   (uint(clamp(color.g, 0.0, 1.0) * 2047.0) << 11) |
				   (uint(clamp(color.b, 0.0, 1.0) * 1023.0) << 22);
	uint enc_depth_alpha = (uint(depth) << 8) | uint(color.a * 255.0);
	return uvec2(enc_col, enc_depth_alpha);
}

vec4 decodeCDColor(uvec2 enc) {
	return vec4(float((enc[0] >> 0) & 0x7ffu) * (1.0 / 2047.0),
				float((enc[0] >> 11) & 0x7ffu) * (1.0 / 2047.0),
				float((enc[0] >> 22) & 0x3ffu) * (1.0 / 1023.0),
				float(enc[1] & 0xffu) * (1.0 / 255.0));
}

vec3 linearToSRGB(vec3 color) {
	return vec3(color.r < 0.0031308 ? 12.92 * color.r : 1.055 * pow(color.r, 1.0 / 2.4) - 0.055,
				color.g < 0.0031308 ? 12.92 * color.g : 1.055 * pow(color.g, 1.0 / 2.4) - 0.055,
				color.b < 0.0031308 ? 12.92 * color.b : 1.055 * pow(color.b, 1.0 / 2.4) - 0.055);
}

vec3 SRGBToLinear(vec3 color) {
	return vec3(
		color.r < 0.04045 ? (1.0 / 12.92) * color.r : pow((color.r + 0.055) * (1.0 / 1.055), 2.4),
		color.g < 0.04045 ? (1.0 / 12.92) * color.g : pow((color.g + 0.055) * (1.0 / 1.055), 2.4),
		color.b < 0.04045 ? (1.0 / 12.92) * color.b : pow((color.b + 0.055) * (1.0 / 1.055), 2.4));
}

uint encodeNormalHemiOct(vec3 n) {
	vec2 p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
	return uint((p.x + p.y) * 32767.0 + 32768.0) | (uint((p.x - p.y) * 16383.0 + 16384.0) << 16) |
		   (n.z < 0.0 ? 0x80000000u : 0u);
}

vec3 decodeNormalHemiOct(uint n) {
	vec2 e = vec2((float(n & 0xffffu) - 32768.0) * (0.5 / 32767.0),
				  (float((n >> 16) & 0x7fffu) - 16384.0) * (0.5 / 16383.0));
	vec2 temp = vec2(e.x + e.y, e.x - e.y);
	vec3 v = vec3(temp, 1.0 - abs(temp.x) - abs(temp.y));
	if((n & 0x80000000u) != 0u)
		v.z = -v.z;
	return normalize(v);
}

vec2 signNotZero(vec2 v) { return vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0); }

vec2 float32x3_to_oct(in vec3 v) {
	// Project the sphere onto the octahedron, and then onto the xy plane
	vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
	// Reflect the folds of the lower hemisphere over the diagonals
	vec2 e = (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
	return e * 0.5 + 0.5;
}

vec3 oct_to_float32x3(vec2 e) {
	e = e * 2.0 - 1.0;
	vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
	if(v.z < 0)
		v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
	return normalize(v);
}

uint encodeNormalOct(vec3 n) {
	vec2 e = float32x3_to_oct(n) * 65535.0;
	return uint(e.x) | (uint(e.y) << 16);
}

vec3 decodeNormalOct(uint ei) {
	vec2 e = vec2(ei & 0xffffu, ei >> 16) * (1.0 / 65535.0);
	return oct_to_float32x3(e);
}

float invLerp(float a, float b, float v) { return (v - a) / (b - a); }

void swap(inout float a, inout float b) {
	float temp = a;
	a = b;
	b = temp;
}

void swap(inout uint a, inout uint b) {
	uint temp = a;
	a = b;
	b = temp;
}

void swap(inout int a, inout int b) {
	int temp = a;
	a = b;
	b = temp;
}

vec4 rectPos(Rect rect, vec3 pos) { return vec4(rect.pos + rect.size * pos.xy, 0.0, 1.0); }
vec2 rectTexCoord(Rect rect, vec3 pos) {
	return (rect.pos + rect.size * pos.xy + vec2(1.0, 1.0)) * 0.5;
}

vec3 gradientColor(uint value, uvec4 steps) {
	vec3 color_a, color_b;
	int step_id = 3;

	for(int i = 0; i < 4; i++)
		if(value < steps[i]) {
			step_id = i;
			break;
		}

	vec3 colors[4] = {vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0),
					  vec3(1.0, 1.0, 1.0)};
	float base_step = step_id == 0 ? 0 : steps[step_id - 1];
	vec3 base_color = step_id == 0 ? vec3(0) : colors[step_id - 1];
	float t = float(value - base_step) / float(steps[step_id] - base_step);
	return colors[step_id] * t + base_color * (1.0 - t);
}

// --------------------- Lighting functions -----------------------------------

vec3 skyColor(float vertical_pos) {
	vec3 sky = vec3(5.0, 234.0, 250.0) / 255.0;
	vec3 horizon = vec3(247.0, 214.0, 255.0) / 255.0;
	return mix(sky, horizon, 1.0 - vertical_pos);
}

vec3 finalShading(Lighting lighting, vec3 diffuse, float light_value) {
	// TODO: read more about HDR?
	diffuse = SRGBToLinear(diffuse);
	vec3 amb_light = lighting.ambient_color.rgb * lighting.ambient_power;
	vec3 dif_light = lighting.sun_color.rgb * lighting.sun_power * light_value;

	//return lighting.scene_color;
	//return diffuse;

	return linearToSRGB(diffuse * (amb_light + dif_light));
}

// --------------------- Viewport functions -----------------------------------

// Spaces:
//
//  View: (0, 0, 0) is camera position, z is in range -(near_plane, far_plane)
//        TODO: why is it negated?
//   NDC: (-1, -1, -1) - (1, 1, 1); near plane is at (0, 0, 1), far plane at (0, 0, -1)
//
// World * view_matrix -> View
//  View * proj_matrix -> Clip
//  Clip / w           -> NDC

float zndcToView(Viewport viewport, float zndc) {
	return (zndc * viewport.proj_matrix[3][3] - viewport.proj_matrix[3][2]) /
		   (zndc * viewport.proj_matrix[2][3] - viewport.proj_matrix[2][2]);
}

float decodeZView(Viewport viewport, vec2 xy) { return -decodeFloat2(xy) * viewport.far_plane; }
vec2 encodeZView(Viewport viewport, float z) { return encodeFloat2(-z * viewport.inv_far_plane); }

float depthToZView(Viewport viewport, float depth_value) {
	float zndc = 2.0 * depth_value - 1.0;
	return -zndcToView(viewport, zndc);
}

float zndcToDepth(Viewport viewport, float zndc) { return (-zndc + 1.0) * 0.5; }

#endif