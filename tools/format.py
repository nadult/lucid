#!/usr/bin/env python

# Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
# This file is part of LucidRaster. See license.txt for details.

import argparse, sys, os, re


def lucid_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


sys.path.insert(0, os.path.join(lucid_dir(), "libfwk", "tools"))
from format import CodeFormatter, find_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="Tool for code formatting and format verification",
    )
    parser.add_argument("-c", "--check", action="store_true")
    args = parser.parse_args()

    formatter = CodeFormatter()
    os.chdir(lucid_dir())
    files = find_files(["src", os.path.join("data", "shaders")], re.compile(".*[.](h|cpp|glsl)$"))
    formatter.format_cpp(files, args.check)
