"""SPIKE (spike/wheels-mac-win): make libint2 2.7.2's C++ headers usable on Windows.

Applied in CI only, never committed to the override itself (its .patch file
pins it byte for byte). What it does, mirroring upstream libint2 >= 2.8:

1. writes an `unistd.h` stub into <prefix>/include: 2.7.2's `libint2/basis.h:39`
   includes <unistd.h> unconditionally (upstream later wrapped it in
   `#ifdef _MSC_VER` with <io.h> and an S_ISDIR macro);
2. ports 2.13.1's `_MSC_VER` posix_memalign/_aligned_malloc block into ferric's
   2.7.2 boys.h override, frees the aligned buffers with the matching call, and
   replaces `std::max(sizeof(Real), 32ul)` (no deduction when unsigned long is
   32-bit, as on Windows) with `(size_t)32`.

    python wheel_spike_win_libint272.py <libint2-prefix> <boys.h override path>
"""

import pathlib
import sys

prefix, boys = map(pathlib.Path, sys.argv[1:3])

(prefix / "include" / "unistd.h").write_text(
    "#pragma once\n"
    "/* SPIKE stub: libint2 2.7.2 basis.h includes <unistd.h> but uses only stat(). */\n"
    "#include <io.h>\n"
    "#include <sys/stat.h>\n"
    "#ifndef S_ISDIR\n"
    "#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)\n"
    "#endif\n"
)

s = boys.read_text()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, f"expected {count} of {old!r}, found {n}"
    s = s.replace(old, new)


rep(
    "#include <memory>\n",
    "#include <memory>\n"
    "#ifdef _MSC_VER\n"
    "#include <malloc.h>\n"
    "#define posix_memalign(p, a, s) \\\n"
    "  (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)\n"
    "#define posix_memfree(p) ((_aligned_free((p))))\n"
    "#else\n"
    "#define posix_memfree(p) ((free((p))))\n"
    "#endif\n",
)
rep("free(c);", "posix_memfree(c);")
rep("free(c_);", "posix_memfree(c_);")
rep("std::max(sizeof(Real), 32ul)", "std::max(sizeof(Real), (size_t)32)")
boys.write_text(s)
print("patched", boys, "and wrote", prefix / "include" / "unistd.h")
