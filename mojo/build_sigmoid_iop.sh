#!/bin/bash
# Build libsigmoid.so from the darktable source tree.
# This mimics what CMake does (run introspection, then compile + link).

set -e

SRC=/home/mc/code/darktable
BUILD=/tmp/sigmoid_build
DEST=$BUILD/libsigmoid.so

mkdir -p "$BUILD"

# Step 1: Run introspection (generates introspection_sigmoid.c)
echo "=== Step 1: Introspection ==="
perl "$SRC/tools/introspection/parser.pl" \
    "$SRC/src/" \
    "$SRC/src/iop/sigmoid.c" \
    "$BUILD/introspection_sigmoid.c"
echo "  -> $BUILD/introspection_sigmoid.c ($(wc -l < "$BUILD/introspection_sigmoid.c") lines)"

# Step 2: Collect flags
GTK_FLAGS=$(pkg-config --cflags gtk+-3.0 glib-2.0 librsvg-2.0 lcms2 json-glib-1.0)
DT_INCLUDE="-I$SRC/src -I$SRC/src/iop -I$SRC/build/bin -I/usr/lib/darktable"
DEFINES="-DHAVE_CONFIG_H -DHAVE_OPENCL -D_GNU_SOURCE -include common/module_api.h -include iop/iop_api.h"
CFLAGS="-O3 -march=native -fPIC -fvisibility=hidden -fopenmp $GTK_FLAGS $DT_INCLUDE $DEFINES"
LDFLAGS="-L/usr/lib/darktable -ldarktable -lm -lgomp -Wl,-rpath,/usr/lib/darktable"

echo ""
echo "=== Step 2: Compile ==="
gcc $CFLAGS \
    -Wno-unused-function \
    -Wno-deprecated-declarations \
    -c "$BUILD/introspection_sigmoid.c" \
    -o "$BUILD/introspection_sigmoid.o" 2>&1
echo "  -> $BUILD/introspection_sigmoid.o"

echo ""
echo "=== Step 3: Link ==="
gcc -shared -fPIC -fopenmp \
    "$BUILD/introspection_sigmoid.o" \
    $LDFLAGS \
    -o "$DEST"
echo "  -> $DEST"

echo ""
echo "=== Step 4: Verify ==="
nm -D "$DEST" | grep " T " | awk '{print $3}'

echo ""
echo "Build complete: $DEST"
echo "Size: $(du -sh "$DEST" | cut -f1)"
