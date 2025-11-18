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

# Components to collect sources from (directories or groups of sources)
set(SRC_lucid
    lucid lucid_app
    lucid_renderer simple_renderer pbr_renderer path_tracer
    scene scene_setup scene_convert
    shading
    texture_atlas
    wavefront_obj
    meshlet quad_generator tri_optimizer
    extern_impl)

set(SRC_FILES "")
foreach(SOURCE ${SRC_lucid})
    list(APPEND SRC_FILES "src/${SOURCE}.cpp")
endforeach()

fwk_add_module(lucid)
add_executable(lucid ${SRC_FILES})
target_link_libraries(lucid PRIVATE libfwk)

set(BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET lucid PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${BASE_DIR}")
set_target_properties(lucid PROPERTIES
    DEBUG_POSTFIX "_d" DEVELOP_POSTFIX "_dev"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BASE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEVELOP "${BASE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BASE_DIR}"
)
