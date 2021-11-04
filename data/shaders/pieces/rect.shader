struct Rect {
	vec2 pos, size;
	vec2 min_uv, max_uv;
};

uniform Rect rect;

vec4 rectPos(vec3 pos) {
	return vec4(rect.pos + rect.size * pos.xy, 0.0, 1.0);
}

vec2 rectTexCoord(vec3 pos) {
	return (rect.pos + rect.size * pos.xy + vec2(1.0, 1.0)) * 0.5;
}
