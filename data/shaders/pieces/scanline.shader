// $$include structures

struct ScanlineParams {
	vec3 min, max, step;
};

ScanlineParams loadScanlineParamsRow(uvec4 val0, uvec4 val1, vec2 start) {
	ScanlineParams params;

	bvec3 xneg = bvec3((val1.w & 1) != 0, (val1.w & 2) != 0, (val1.w & 4) != 0);
	vec3 scan = uintBitsToFloat(val0.xyz);
	params.step = uintBitsToFloat(val1.xyz);

	const float inf = 1.0 / 0.0;
	scan += params.step * start.y - vec3(start.x);
	params.min = vec3(xneg[0] ? -inf : scan[0], xneg[1] ? -inf : scan[1], xneg[2] ? -inf : scan[2]);
	params.max = vec3(xneg[0] ? scan[0] : inf, xneg[1] ? scan[1] : inf, xneg[2] ? scan[2] : inf);

	return params;
}

ScanlineParams loadScanlineParamsBin(uvec4 val0, uvec4 val1, out int min_by, out int max_by) {
	ScanlineParams params;

	vec3 scan = uintBitsToFloat(val0.xyz);
	params.step = uintBitsToFloat(val1.xyz);
	min_by = int(val0.w & 0xffff) >> BIN_SHIFT;
	max_by = int(val0.w >> 16) >> BIN_SHIFT;

	bvec3 xneg = bvec3((val1.w & 1) != 0, (val1.w & 2) != 0, (val1.w & 4) != 0);
	bvec3 yneg = bvec3((val1.w & 8) != 0, (val1.w & 16) != 0, (val1.w & 32) != 0);

	// Computing offsets for trivial reject corner
	float offset = BIN_SIZE - 0.989;
	vec3 yoffset = vec3(yneg[0] ? 0.0 : offset, yneg[1] ? 0.0 : offset, yneg[2] ? 0.0 : offset);
	vec3 xoffset = vec3(xneg[0] ? 0.0 : offset, xneg[1] ? 0.0 : offset, xneg[2] ? 0.0 : offset);

	vec2 start = vec2(0.99, min_by * BIN_SIZE - 0.01);
	scan += params.step * (yoffset + vec3(start.y)) - (xoffset + vec3(start.x));
	const float inf = 1.0 / 0.0;
	params.min = vec3(xneg[0] ? -inf : scan[0], xneg[1] ? -inf : scan[1], xneg[2] ? -inf : scan[2]);
	params.max = vec3(xneg[0] ? scan[0] : inf, xneg[1] ? scan[1] : inf, xneg[2] ? scan[2] : inf);
	params.step *= BIN_SIZE;

	return params;
}