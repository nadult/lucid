// $$include funcs

layout(std430, binding = 0) readonly buffer buf0_ { uint g_raster_image[]; };

uniform ivec2 bin_counts;
uniform vec2 screen_scale;

#ifdef VERTEX_SHADER

in uint in_pos;

out flat uint v_bin_offset;
out vec2 v_bin_pos;

void main() {
	uint bin_id = in_pos >> 16;
	uint pos_x = in_pos & 0xff, pos_y = (in_pos >> 8) & 0xff;
	uint bin_pos_y = bin_id / bin_counts.x, bin_pos_x = bin_id - bin_pos_y * bin_counts.x;

	v_bin_offset = bin_id << (BIN_SHIFT * 2);
	v_bin_pos = vec2(pos_x, pos_y);
	vec2 screen_pos = (vec2(pos_x, pos_y) + vec2(bin_pos_x, bin_pos_y) * BIN_SIZE) * screen_scale;
	gl_Position = vec4(screen_pos * 2.0 - vec2(1.0), 0.0, 1.0);
}

#else

in vec2 v_bin_pos;
in flat uint v_bin_offset;
out vec4 frag_color;

void main() {
	ivec2 bin_pos = ivec2(v_bin_pos);
	uint col = g_raster_image[v_bin_offset + bin_pos.x + (bin_pos.y << BIN_SHIFT)];
	frag_color.rgb = decodeRGB8(col); // TODO: 11:11:10 RGB bits
	frag_color.a = 1.0;
}

#endif
