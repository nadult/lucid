#include "lucid_base.h"

#include <fwk/enum_map.h>

DEFINE_ENUM(WavefrontMapType, diffuse, bump);

struct WavefrontMap {
	string name;
	vector<string> args;
};

struct WavefrontMaterial {
	static Ex<void> load(ZStr path, vector<WavefrontMaterial> &out);

	using MapType = WavefrontMapType;
	string name;
	float dissolve_factor = 1.0;
	float3 diffuse = float3(1.0f);
	EnumMap<MapType, WavefrontMap> maps = {};
};

struct WavefrontObject {
	static Ex<WavefrontObject> load(ZStr path, i64 file_size_limit = 1400 * 1024 * 1024);

	vector<float3> positions;
	vector<float3> normals;
	vector<float2> tex_coords;
	vector<Array<int, 3>> tris;

	struct MaterialGroup {
		int material_id = 0;
		int first_tri = 0, num_tris = 0;
	};

	string resource_path;
	vector<WavefrontMaterial> materials;
	vector<MaterialGroup> material_groups;
};
