# Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
# This file is part of LucidRaster. See license.txt for details.

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_GENERATOR MATCHES "Visual Studio")
	set(CMAKE_GENERATOR_TOOLSET "ClangCL")
endif()

project(lucid VERSION 0.1 LANGUAGES CXX)

set(FWK_BUILD_TESTS OFF CACHE BOOL "")
set(FWK_BUILD_TOOLS OFF CACHE BOOL "")
set(FWK_UNITY_BUILD ON CACHE BOOL "")
set(FWK_DEPENDENCIES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies" CACHE PATH "")
add_subdirectory(libfwk EXCLUDE_FROM_ALL)

set(HEADERS_lucid
    lucid_app.h
    lucid_base.h
    lucid_pch.h
    lucid_renderer.h
    meshlet.h
    path_tracer.h
    pbr_renderer.h
    quad_generator.h
    scene_convert.h
    scene_setup.h
    scene.h
    shader_structs.h
    shading.h
    simple_renderer.h
    texture_atlas.h
    wavefront_obj.h
)

set(SOURCES_lucid
    extern_impl.cpp
    lucid_app.cpp
    lucid_renderer.cpp
    lucid.cpp
    meshlet.cpp
    path_tracer.cpp
    pbr_renderer.cpp
    quad_generator.cpp
    scene_convert.cpp
    scene_setup.cpp
    scene.cpp
    shading.cpp
    simple_renderer.cpp
    texture_atlas.cpp
    tri_optimizer.cpp
    wavefront_obj.cpp
)

list(TRANSFORM HEADERS_lucid PREPEND "src/")
list(TRANSFORM SOURCES_lucid PREPEND "src/")

add_executable(lucid)
target_sources(lucid PRIVATE ${SOURCES_lucid})
target_sources(lucid PUBLIC FILE_SET HEADERS FILES ${HEADERS_lucid})
target_link_libraries(lucid PRIVATE libfwk)

set(BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET lucid PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${BASE_DIR}")
set_target_properties(lucid PROPERTIES
    DEBUG_POSTFIX "_dbg" DEVELOP_POSTFIX "_dev"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BASE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEVELOP "${BASE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BASE_DIR}"
)
