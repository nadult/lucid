#!/usr/bin/env python

import argparse, os, re, subprocess, shutil


# TODO: import this form libfwk/tools/format.py
class CodeFormatter:
    def __init__(self, expected_version=17):
        self.expected_version = expected_version
        self.clang_format_cmd = self._find_clang_format_cmd()
        self._verify_clang_format()

    def _find_clang_format_cmd(self):
        names = [f"clang-format-{self.expected_version}", "clang-format"]
        for name in names:
            if shutil.which(name) is not None:
                return name
        raise Exception(f"{names[0]} is missing")

    def _verify_clang_format(self):
        result = subprocess.run([self.clang_format_cmd, "--version"], stdout=subprocess.PIPE)
        tokens = result.stdout.decode("utf-8").split()
        while len(tokens) > 0 and tokens[0] != "clang-format":
            tokens.pop(0)
        if (
            result.returncode != 0
            or len(tokens) < 3
            or tokens[0] != "clang-format"
            or tokens[1] != "version"
        ):
            raise Exception(f"error while checking clang-format version (version string: {tokens})")
        version = tokens[2].split(".", 2)
        print(f"clang-format version: {version[0]}.{version[1]}.{version[2]}")

        if int(version[0]) < self.expected_version:
            raise Exception(
                "clang-format is too old; At least version {self.expected_version} is required"
            )

    def format_cpp(self, files, check: bool):
        print("Checking code formatting..." if check else "Formatting code...")
        full_command = [self.clang_format_cmd, "-i"]
        if check:
            full_command += ["--dry-run", "-Werror"]
        full_command += files
        result = subprocess.run(full_command)

        if check:
            if result.returncode != 0:
                exit(1)
            print("All OK")


def find_files(root_dirs, regex):
    out = []
    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if regex.match(file):
                    out.append(os.path.join(root, file))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="Tool for code formatting and format verification",
    )
    parser.add_argument("-c", "--check", action="store_true")
    args = parser.parse_args()

    formatter = CodeFormatter()
    lucid_path = os.path.abspath(os.path.join(__file__, "../.."))
    os.chdir(lucid_path)
    files = find_files(["src", os.path.join("data", "shaders")], re.compile(".*[.](h|cpp|glsl)$"))
    formatter.format_cpp(files, args.check)
