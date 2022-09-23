import os
import ycm_core

flags = [
'-x', 'c++', '-std=c++2a', '-fno-exceptions',
'-Wall', '-Wextra', '-Werror', '-Woverloaded-virtual', '-Wnon-virtual-dtor', '-Werror=return-type', '-Werror=switch',
'-Wimplicit-fallthrough', '-Wno-reorder', '-Wuninitialized', '-Wno-unused-function', '-Wno-unused-variable',
'-Wno-unused-parameter', '-Wparentheses', '-Wno-overloaded-virtual', '-Wno-undefined-inline', '-Wno-unknown-attributes',
'-DFWK_PARANOID',
'-Ilibfwk/include', '-Isrc/',
'-include', 'src/lucid_pch.h'
]

def appendSystemPaths():
    for idx in range(20, 6, -1):
        cpp_path = "/usr/include/c++/" + str(idx)
        if os.path.isdir(cpp_path):
            break
    if not os.path.isdir(cpp_path):
        raise Exception("Cannot find libstdc++ headers")

    flags.extend([
        '-isystem', cpp_path,
        '-isystem', '/usr/local/include',
        '-isystem', '/usr/include/SDL2',
        '-isystem', '/usr/include/freetype2',
        '-isystem', '/usr/include' ])

def MakePathsAbsolute(flags, working_directory):
  if not working_directory:
    return list(flags)
  new_flags = []
  make_next_absolute = False
  path_flags = [ '-isystem', '-I', '-iquote', '--sysroot=' ]
  for flag in flags:
    new_flag = flag
 
    if make_next_absolute:
      make_next_absolute = False
      if not flag.startswith( '/' ):
        new_flag = os.path.join( working_directory, flag )
 
    for path_flag in path_flags:
      if flag == path_flag:
        make_next_absolute = True
        break
 
      if flag.startswith( path_flag ):
        path = flag[ len( path_flag ): ]
        new_flag = path_flag + os.path.join( working_directory, path )
        break
 
    if new_flag:
      new_flags.append( new_flag )
  return new_flags

def Settings( **kwargs ):
    global flags
    relative_to = os.path.dirname(os.path.abspath(__file__))
    flags = MakePathsAbsolute(flags, relative_to)
    appendSystemPaths()
    return { 'flags': flags }
