#!/bin/bash
MOJO_LIB_DIR="/home/mc/code/darktable/mojo/.pixi/envs/default/lib"
# We need both the Mojo libs and the directory where libsigmoid_mojo.so lives (usually /usr/lib)
export LD_LIBRARY_PATH="$MOJO_LIB_DIR:/usr/lib:$LD_LIBRARY_PATH"

echo "--- Launching Darktable with Mojo GPU Support ---"
/usr/bin/darktable -d opencl "$@"
