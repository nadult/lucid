// TODO: safety for data types: when loading how can we make sure
// that there are no leftovers from previous processing ?
//
// TODO: when de-serializing enums we have to make sure that they are in valid range,
//      inf not then sanitize them and report error!
//
// W dowolnym momencie coś w streamie może się zepsuć; Jak to naprawić ?
// Czy chcemy dać limit na wielkość stringa ?
// Możemy mieć funkcję loadString która bierze taki argument ?
// OK, jak jest błąd to w jaki sposób go zgłaszamy ?
//
// Na pewno stream może przechowywać błąd wewnątrz. ale jak go wyciągnąć ?
//
// A) przechwycenie błędu w strumieniu ręcznie ?
// B) jak poleci inny błąd, to trzeba też sprawdzić, czy nie zaszedł wcześniej błąd w strunieniu
//    Może zrobić tu podobnie jak z XMLem ?
//    - przy zamknięciu strumienia można wymusić pobranie błędu ?
// C) jakakolwiek operacja na strumieniu ktora generuje blad EX<>
//
// Wszystkie te operacje mają zwracać EX<> ?
// Może żadna nie powinna ? albo raportujemy wewn. Strumienia albo zwracamy Ex<>
// Wtedy
// Chcę wiedzieć kiedy coś złego w strumieniu się zadziało
//
// Mogę zrobić strumienie tak, że błędy przy strumieniowaniu są domyślnie ukrywane, tzn.
// nie mam wszędzie zwracać Ex<>
//
// A nie mogę po prostu założyć, że dane są poprawne ?
//
// Sprecyzujmy wymagania jakie mamy w stosunku do wszystkich podsystemów
//
//
// Strumień:
// - prosty sposób strumieniowania danych; obsługiwane opcje:
//   pliki / pipe-y(potem) / pamięć
// - jak coś się psuje to podczas zapisywania sie nic nie dzieje a podczas odczytu
//   zwraca 0 i tyle
// - zapisuje pierwszy błąd jaki się pojawił; można ten błąd odzyskać
//   ** jak zapewnić, że błąd nie jest ignorowany ? **
//      przy destrukcji strumienia jakoś go wypisywać jak nie został
//      przechwycony ? później można zrobić prosty system warningów... OK!
//   coś mi się tutaj nie podoba...
//   fakt, że można zignorować błąd, czy coś jeszcze ?
//
// - tylko podstawowe typy są domyślnie serializowalne ?
//   ok a co z: vector<>, hash_map<>, string, pair<> ?
//   tu też mogę dać limity ?
//   albo najpierw zbudować  na podstawowe typy i potem w ramach potrzeb!
//
//
// Tak samo z XMLem i z parserem ?
// - parser i plik XML mogą przechowywać informacje o błedach
//   a moze po prostu potrzebuje globalnego systemu do raportowanie bledow ?
//   duzo sie nie zmieni z wyjatkiem lokalizacji bledow
//
//   Chodzi o to, żeby

// Ex< T & > >> dziala jesli nie zawiera bledu, w przypadku bledu ignoruje kolejne wywolania ?
// a moze po prostu zamiast expected zawsze trzymac blad serializacji wewn. strumienia ?
// funkcje serializujace by nie zwracaly ex<> tylko ustawialy to w strumieniu ?
// OK ale blad w strumieniu mozna zignnorowac latwo ?
//stream >> meshes >> materials >> instances;

// TODO: extended support for exConstruct ?
// TODO: emplace_back mogłoby mieć opcję kontruowania z EX<>
//       w przypadku błędu nie dodaje elementu ?
// TODO: może funkcja loadMany w takiej sytuacji się przyda ?
// Tutaj problem jest z Ex a nie z loadem: chodzi o to, żeby to się
// tak wygodnie przekazywało jak wyjątek; Jak taką funkcję nazwać ?
// funkcja która wielokrotnie odpala zestaw funkcj
// generalnie construct<> powinno być wydajne (copy elison)

// Jak bład w strumieniu zamieniam na EX<> ?
// muszę go jakoś wykryć i zwrócić ?
// I tak muszę obsługiwać sytuacje, gdzie dane są błędne, chodzi tylko o to,

// Przydałyby się też opcje serializacji, żeby te funkcje były sensownie rozszerzalne
// a interfejs mnie nie ograniczał...
//
// Strumien ma tylko support dla typów POD ? reszta ręcznie ?
// można też użyć exConstruct zamiast load ?

#include "scene_setup.h"

#include "shading.h"
#include <fwk/enum_map.h>
#include <fwk/gfx/fpp_camera.h>
#include <fwk/gfx/orbiting_camera.h>
#include <fwk/gui/imgui.h>
#include <fwk/gui/widgets.h>
#include <fwk/io/file_stream.h>

FilePath mainPath();

SceneSetup::SceneSetup(string name) : name(move(name)) {}
SceneSetup::~SceneSetup() = default;

BoxesSetup::BoxesSetup() : SceneSetup("#boxes") { render_config.scene_opacity = 0.5; }
PlanesSetup::PlanesSetup() : SceneSetup("#planes") { render_config.scene_opacity = 0.25; }

void BoxesSetup::doMenu(VDeviceRef device) {
	auto scene_dims = m_dims;
	auto &gui = Gui::instance();
	gui.text("Dimensions:");
	ImGui::SameLine();
	if(ImGui::InputInt3("##dims", &scene_dims.x, ImGuiInputTextFlags_EnterReturnsTrue)) {
		scene_dims = vclamp(scene_dims, int3(1), int3(16));
		if(scene_dims != m_dims) {
			m_dims = scene_dims;
			if(scene)
				updateScene(device).check();
		}
	}
}

void PlanesSetup::doMenu(VDeviceRef device) {
	auto &gui = Gui::instance();
	int label_size = (int)ImGui::CalcTextSize("Num planes").x;
	ImGui::SetNextItemWidth(220 * gui.dpiScale() - label_size);
	int num_planes = m_num_planes;
	ImGui::SliderInt("Num planes", &num_planes, 1, 256);
	if(scene && m_num_planes != num_planes) {
		m_num_planes = num_planes;
		updateScene(device).check();
	}
}

static void addBox(Scene &scene, SceneMesh &out, IColor color, float size, float3 pos) {
	auto corners = (FBox(float3(size)) + pos).corners();
	array<int, 3> tris[12] = {{0, 2, 3}, {0, 3, 1}, {1, 3, 7}, {1, 7, 5}, {2, 6, 7}, {2, 7, 3},
							  {0, 6, 2}, {0, 4, 6}, {0, 5, 4}, {0, 1, 5}, {4, 7, 6}, {4, 5, 7}};

	int off = scene.positions.size();
	for(auto &tri : tris)
		out.tris.emplace_back(tri[0] + off, tri[1] + off, tri[2] + off);
	insertBack(scene.positions, corners);
	scene.colors.resize(scene.colors.size() + corners.size(), color);
}

static void addQuad(Scene &scene, SceneMesh &out, IColor color, float size, float3 pos) {
	auto corners = (FRect(float2(size)) + pos.xy()).corners();
	array<int, 3> tris[2] = {{0, 1, 2}, {0, 2, 3}};

	int off = scene.positions.size();
	for(auto &tri : tris)
		out.tris.emplace_back(tri[0] + off, tri[1] + off, tri[2] + off);
	for(auto &corner : corners)
		scene.positions.emplace_back(float3(corner, pos.z));
	scene.colors.resize(scene.colors.size() + corners.size(), color);
}

Ex<> BoxesSetup::updateScene(VDeviceRef device) {
	if(m_current_dims == m_dims && scene)
		return {};

	m_current_dims = m_dims;
	float3 offset = -float3(m_dims) * (m_box_size + m_box_dist) * 0.5f;
	float3 col_scale = vinv(float3(m_dims) - float3(1));
	Random rand;

	scene = Scene{};
	SceneMesh mesh;
	for(int x = 0; x < m_dims.x; x++) {
		for(int y = 0; y < m_dims.y; y++) {
			for(int z = 0; z < m_dims.z; z++) {
				float3 pos = offset + float3(x, y, z) * (m_box_size + m_box_dist);
				FColor color(float3(x, y, z) * col_scale, 1.0f);
				pos += rand.sampleBox(float3(-0.1f), float3(0.1f));
				addBox(*scene, mesh, IColor(color), m_box_size, pos);
			}
		}
	}
	mesh.bounding_box = enclose(scene->positions);
	mesh.colors_opaque = true;

	scene->materials.emplace_back("default");
	scene->meshes.emplace_back(std::move(mesh));
	scene->generateQuads(4.0f);

	views = {OrbitingCamera({}, 10.0f, 0.5f, 0.8f)};
	if(!camera)
		camera = views.front();
	return scene->updateRenderingData(*device);
}

Ex<> PlanesSetup::updateScene(VDeviceRef device) {
	if(m_current_planes == m_num_planes && scene)
		return {};
	m_current_planes = m_num_planes;

	scene = Scene{};
	SceneMesh mesh;
	for(int z = 0; z < m_num_planes; z++) {
		float size = m_plane_size * (1.0 + float(z) * 0.05);
		float t = float(z) / (m_num_planes - 1);
		float3 pos = float3(-0.5, -0.5, 0.0) * size + float3(0.0, 0, z * m_plane_dist);
		FColor color(hsvToRgb(float3(t, 1.0, 1.0)), 1.0f);
		addQuad(*scene, mesh, IColor(color), size, pos);
	}
	mesh.bounding_box = enclose(scene->positions);
	mesh.colors_opaque = true;

	scene->materials.emplace_back("default");
	scene->meshes.emplace_back(std::move(mesh));
	scene->generateQuads(4.0f);

	// TODO: rasterization bugs visible in this scene when camera forward
	// vector is equal to -plane.normal
	views = {FppCamera({0, 0, -5.0}, normalize(float2(0.001, 1.0)), 0.001)};
	if(!camera)
		camera = views.front();
	return scene->updateRenderingData(*device);
}

LoadedSetup::LoadedSetup(string name) : SceneSetup(move(name)) {}

Ex<> LoadedSetup::updateScene(VDeviceRef device) {
	if(scene)
		return {};
	auto path = format("%/scenes/%.scene", mainPath(), name);
	scene = EX_PASS(Scene::load(path));

	auto name = this->name;
	if(name.rfind("_old") != string::npos)
		name.resize(name.size() - 4);

	if(isOneOf(name, "bunny", "hairball", "teapot"))
		render_config.scene_opacity = 0.5;
	else
		render_config.scene_opacity = 0.8;

	EXPECT(scene->updateRenderingData(*device));

	auto box = scene->bounding_box;
	auto max_size = max(box.width(), box.height(), box.depth());
	OrbitingCamera default_cam(box.center(), max_size, 0.5f, 0.8f);

	views.clear();
	if(name == "powerplant") {
		insertBack(views,
				   {FppCamera{{6.479178, 15.869515, -5.917777}, {-0.349199, 0.602876}, 0.847362},
					FppCamera{{-5.031062, 14.030015, 8.243547}, {0.491592, -0.493698}, 0.405695},
					FppCamera{{-5.077578, 2.493423, 10.024123}, {0.534752, 0.446587}, 0.162463}});
	} else if(name == "gallery") {
		insertBack(views,
				   {FppCamera{{-0.518197, 11.031467, -28.708052}, {-0.034821, 0.695836}, 0.180695},
					FppCamera{{-10.443496, 9.609625, -6.724856}, {0.483258, 0.501858}, 0.364028}});
	} else if(name == "conference") {
		insertBack(
			views,
			{FppCamera{{46.586071, 4.31637, -15.807777}, {-0.775055, 0.554252}, 0.106798},
			 FppCamera{{50.89043, 4.515099, 19.323435}, {-0.863081, -0.403725}, 0.031798},
			 FppCamera{{57.927486, 19.688606, -36.936211}, {-0.834299, 0.460266}, 0.273465}});
	} else if(name == "dragon") {
		views.emplace_back(OrbitingCamera{{-1.61925, 6.953201, 2.7753}, 100, -0.933333, 0.516666});
	} else if(name == "sponza") {
		insertBack(views,
				   {FppCamera{{-4.045459, 12.368692, -4.060153}, {-0.075328, 0.69262}, 0.639028},
					FppCamera{{28.343906, 16.751987, -5.52777}, {-0.638277, 0.279287}, 0.147361},
					FppCamera{{30.492384, 5.880484, -1.414973}, {-0.69641, 0.020292}, 0.014028},
					FppCamera{{21.594707, 4.578131, 5.349687}, {-0.69641, 0.020292}, 0.014028}});
	} else if(name == "san-miguel") {
		insertBack(views,
				   {FppCamera{{34.412136, 26.08173, 19.988665}, {-0.487458, -0.497777}, 0.814029},
					FppCamera{{18.670315, 1.785693, 8.138441}, {-0.461969, -0.521516}, 0.672362}});
	} else if(name == "white_oak") {
		insertBack(views,
				   {OrbitingCamera{{-1.35474, 44.001759, -1.40296}, 32.840042, 0.483333, 0.425},
					default_cam});
	} else if(name == "hairball") {
		views.emplace_back(default_cam);
		views.emplace_back(
			FppCamera{{0.133944, 0.298333, -0.290384}, {-0.334019, 0.611418}, 0.830695});
	} else if(name == "buddha") {
		views.emplace_back(
			OrbitingCamera{{-2.746651, 75.126877, -3.378395}, 103.999908, 3.108339, 0.266667});
	} else if(name == "dragon2") {
		views.emplace_back(
			OrbitingCamera{{1.100576, -3.339322, 4.876184}, 73.448845, 0.741666, 0.283334});
	} else {
		views.emplace_back(default_cam);
	}

	if(isOneOf(name, "powerplant", "conference", "bunny", "dragon"))
		render_config.backface_culling = true;

	if(!camera)
		camera = views.front();
	return {};
}

vector<string> LoadedSetup::findAll() {
	auto out = findFiles("scenes/", ".scene");
	makeSorted(out);
	return out;
}
