import ctypes
# libc = ctypes.cdll.msvcrt # Windows
# libc = ctypes.CDLL('libc.dylib') # Mac
libc = ctypes.CDLL('libc.so') # Linux and most other *nix
libc.printf(b'hi there, %s\n', b'world')
17
