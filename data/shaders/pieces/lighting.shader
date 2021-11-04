// $$include funcs

struct Lighting {
	vec3 ambient_color;
	vec3 scene_color;
	vec3 sun_color;
	vec3 sun_dir;
	float scene_power, sun_power, ambient_power;
};

uniform Lighting lighting;

vec3 skyColor(float vertical_pos) {
	vec3 sky = vec3(5.0, 234.0, 250.0) / 255.0;
	vec3 horizon = vec3(247.0, 214.0, 255.0) / 255.0;
	return mix(sky, horizon, 1.0 - vertical_pos);
}

vec3 finalShading(vec3 diffuse, float light_value) {
	// TODO: is this correct? read more about HDR?
	diffuse = SRGBToLinear(mix(diffuse, lighting.scene_color, lighting.scene_power));
	vec3 amb_light = lighting.ambient_color * lighting.ambient_power;
	vec3 dif_light = lighting.sun_color * lighting.sun_power * light_value;

	//return lighting.scene_color;
	//return diffuse;

	return linearToSRGB(diffuse * (amb_light + dif_light));
}
