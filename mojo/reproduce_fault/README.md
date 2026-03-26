## The Problem
When a Mojo function is exported to C and launches a GPU kernel via `elementwise`, any variables "captured" by the kernel's closure are kept on the **Host (CPU) Stack**. 

- **Manual Loop (CPU):** Works fine. The Mojo function can iterate and access its own stack variables locally.
- **`elementwise` (CPU or GPU):** Fails via FFI. The utility seems to assume a Mojo-managed stack layout for its closure handling, which is violated when the entry point is a C function.
- **GPU Failure:** Specifically, the GPU attempts to reach back to the CPU's stack address, which is not mapped in the GPU's page tables, causing a `SIGSEGV`.

## Evidence
- **Host Stack Address:** `0x7fff266bb2a4`
- **GPU Fault Address:** `0x7fff266ba000`
The GPU is attempting to read memory at the exact location of the C program's stack.

## Files
- `lib_fault.mojo`: Mojo library capturing a `BigParams` struct.
- `main.c`: C program that `dlopen`s the Mojo library.
- `build.sh`: Script to compile and run the reproduction.

## Running the reproduction
```bash
./build.sh
```
Expected output:
```text
C: Host Stack Address around 0x7fff...
C: Calling Mojo launch_gpu_kernel()...
Mojo: Launching GPU kernel capturing a BigParams struct...
Memory access fault by GPU node-1 ... on address 0x7fff...
```
