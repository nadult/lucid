# LucidRaster: GPU Software Rasterizer for Exact Order-Independent Transparency
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build status](https://github.com/nadult/lucid/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/nadult/lucid/actions)

LucidRaster is a software rasterizer running on a GPU which allows for efficient exact rendering of
complex transparent scenes. It uses a new two-stage sorting technique and sample accumulation
method. On average it's faster than high-quality OIT approximations and only about 3x slower than
hardware alpha blending. It can be very efficient especially when rendering scenes with high
triangle density or high depth complexity.

Most of LucidRaster's logic is implemented in Vulkan compute shaders, the rest of the code is mainly
C++.

[Paper](https://arxiv.org/abs/2405.13364)  
[Windows build + scene files](https://github.com/nadult/lucid/releases)  
[Project page (more details)](https://nadult.github.io/lucid/)  
[Author's Linkedin profile](https://www.linkedin.com/in/nadult/)  

This work is licensed under a [GNU GPL v3 license](https://www.gnu.org/licenses/gpl-3.0.html).

![](https://nadult.github.io/images/lucid/lucid1.jpg)



## Building

LucidRaster is based on CMake and [libfwk](https://github.com/nadult/libfwk) framework. Please take
a look at libfwk's readme to learn what tools / compilers are required. LucidRaster can be easily
built under Windows and Linux by running the following commands:
```
cd lucid/
git submodule update --init --recursive
libfwk/tools/configure.py download-deps
libfwk/tools/configure.py
cmake --build build --parallel
```

There are also github workflows available, which build LucidRaster for Windows & Linux.