#include "shading.h"

#include "program.h"
#include <fwk/gfx/gl_program.h>
#include <fwk/gfx/gl_texture.h>
#include <fwk/io/stream.h>

SceneLighting SceneLighting::makeDefault() {
	SceneLighting out;
	out.sun.dir = {0.842121, -0.300567, -0.447763};
	out.sun.color = {0.8, 0.8, 0.8};
	out.sun.power = 2.5;
	out.ambient.color = {0.8, 0.8, 0.6};
	out.ambient.power = 0.4f;
	out.scene.color = {0.7, 0.6, 0.5};
	out.scene.power = 0.3f;
	return out;
}

void SceneLighting::setUniforms(PProgram program) const {
	program["lighting.ambient_color"] = ambient.color;
	program["lighting.ambient_power"] = ambient.power;
	program["lighting.scene_color"] = scene.color;
	program["lighting.scene_power"] = scene.power;
	program["lighting.sun_color"] = sun.color;
	program["lighting.sun_power"] = sun.power;
	program["lighting.sun_dir"] = sun.dir;
}
