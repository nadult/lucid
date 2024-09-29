
# LucidRaster: GPU Software Rasterizer for Exact Order-Independent Transparency [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Build status](https://github.com/nadult/lucid/workflows/build/badge.svg?branch=main)](https://github.com/nadult/lucid/actions)

LucidRaster is a software rasterizer running on a GPU which allows for efficient exact rendering of complex transparent scenes. It uses a new two-stage sorting technique and sample accumulation method. On average it's faster than high-quality OIT approximations and only about 3x slower than hardware alpha blending. It can be very efficient especially when rendering scenes with high triangle density or high depth complexity.

Most of LucidRaster's logic is implemented in Vulkan compute shaders, the rest of the code is mainly in C++.

[Paper](https://arxiv.org/abs/2405.13364)  
[Windows build + scene files](https://github.com/nadult/lucid/releases)  
[Project page (more details)](https://nadult.github.io/lucid/)  
[Author's Linkedin profile](https://www.linkedin.com/in/nadult/)  

This work is licensed under a [GNU GPL v3 license](https://www.gnu.org/licenses/gpl-3.0.html).

![](https://nadult.github.io/images/lucid/lucid1.jpg)



## Building

The easiest way to build LucidRaster is by using github [build action](https://github.com/nadult/lucid/actions/workflows/test.yml), which builds Lucid for windows and prepares an artifact, which can be downloaded.  

LucidRaster is using [libfwk](https://github.com/nadult/libfwk) framework and can be built for Windows and Linux. To build it:
- properly initialize and recursively update all submodules (libfwk and imgui in libfwk/extern/)
- make sure that all libfwk dependencies are available (see libfwk's readme for that)
- under linux build with make
- under windows simply build with Visual Studio