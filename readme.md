# LucidRaster: real-time GPU software rasterizer for exact order independent transparency

Lucid is a software rasterizer running on a gpu designed specifically to support efficient order independent transparency. Most of Lucid logic is implemented in Vulkan compute shaders, the rest in C++. 
  
[Project page (more details + scene files)](https://nadult.github.io/lucid/)  
[Author's Linkedin profile](https://www.linkedin.com/in/nadult/)  

![](https://nadult.github.io/images/lucid/lucid1.jpg)

## Building

Lucid is using [libfwk](https://github.com/nadult/libfwk) framework and can be built for Windows and Linux. To build you have to:
- properly initialize and update all submodules (libfwk and imgui in libfwk/extern/)
- make sure that all libfwk dependencies are available
- under linux build with make
- under windows Visual Studio 2022 with clang compiler is required; use windows/lucid.sln and make sure that all libfwk dependencies are reachable; libfwk uses windows/shared_libraries.props file to define paths to dependencies; The easiest way is to simply fix include & library paths in this file
