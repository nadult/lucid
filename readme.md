# Lucid: Real-time GPU software transparency rasterizer

Lucid is a fully software-based rasterizer designed specifically for efficient rendering of transparent scenes running on Vulkan's compute shaders. 
Core shaders are only available in compiled Spir-v format. If you're interested in full shader code, please contact the author.  
  
[Project page (more details + scene files)](https://nadult.github.io/lucid/)  
[Author's Linkedin profile](https://www.linkedin.com/in/nadult/)  

![](https://nadult.github.io/images/lucid/lucid1.jpg)

## Building

Lucid is using [libfwk](https://github.com/nadult/libfwk) framework and can be built for Windows and Linux. To build you have to:
- properly initialize and update all submodules (libfwk and imgui in libfwk/extern/)
- make sure that all libfwk dependencies are available
- under linux build with make
- under windows Visual Studio 2022 with clang compiler is required; use windows/lucid.sln and make sure that all libfwk dependencies are reachable; libfwk uses windows/shared_libraries.props file to define paths to dependencies; The easiest way is to simply fix include & library paths in this file

