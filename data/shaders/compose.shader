// $$include rect funcs

layout(binding = 0) uniform usampler2D raster_image;

#ifdef VERTEX_SHADER

in vec3 in_pos;
out vec2 v_tex_coord;

void main() {
	gl_Position = rectPos(in_pos);
	v_tex_coord = rectTexCoord(in_pos);
}

#else

in vec2 v_tex_coord;
out vec4 frag_color;

void main() {
	uint col = texelFetch(raster_image, ivec2(v_tex_coord * textureSize(raster_image, 0)), 0).x;
	frag_color = decodeRGBA8(col);
}

#endif
