#!/bin/bash
set -e

echo "--- Building Mojo Library ---"
pixi run mojo build lib_fault.mojo --emit shared-lib -o lib_fault.so

echo "--- Building C Driver ---"
clang main.c -Wl,-rpath=. -ldl -o reproduce_crash

./reproduce_crash || echo "Exited with code $?"
