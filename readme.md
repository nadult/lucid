
# LucidRaster: real-time GPU software rasterizer for exact order independent transparency [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Build status](https://github.com/nadult/lucid/workflows/build/badge.svg?branch=main)](https://github.com/nadult/lucid/actions)

LucidRaster is quite efficient: it's faster than high-quality approximations like Moment-based OIT and it's exact. On average it takes only 3x as much time as (unordered) hardware alpha-blending. For scenes with lots of tiny triangles it can even be faster than hardware. 

Most of LucidRaster's logic is implemented in Vulkan compute shaders, the rest of the code is mainly in C++.

[Paper draft](https://nadult.github.io/lucid/paper_draft.html)  
[Project page (more details + scene files)](https://nadult.github.io/lucid/)  
[Author's Linkedin profile](https://www.linkedin.com/in/nadult/)  

This work is licensed under a [GNU GPL v3 license](https://www.gnu.org/licenses/gpl-3.0.html).

![](https://nadult.github.io/images/lucid/lucid1.jpg)



## Building

The easiest way to build LucidRaster is by using github [build action](https://github.com/nadult/lucid/actions/workflows/build.yml), which builds Lucid for windows and prepares an artifact, which can be downloaded.  

LucidRaster is using [libfwk](https://github.com/nadult/libfwk) framework and can be built for Windows and Linux. To build it:
- properly initialize and recursively update all submodules (libfwk and imgui in libfwk/extern/)
- make sure that all libfwk dependencies are available (see libfwk's readme for that)
- under linux build with make
- under windows simply build with Visual Studio