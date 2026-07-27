# Blurs Kernel Benchmarks

This directory contains benchmarks for the darktable `blurs` module kernels, comparing the **OpenCL (C)** implementation against the **Mojo GPU (Tiled)** implementation.

## Running Benchmarks

### 1. OpenCL Baseline (C)
```bash
make run
```

### 2. Mojo GPU Benchmark
Run from the `benchmark/blurs` directory. Use `-I` to point to the `mojo/` source root.

```bash
mojo -I /path/to/darktable/mojo blurs_benchmark_gpu.mojo
```

Or if using pixi:
```bash
pixi run mojo -I /path/to/darktable/mojo blurs_benchmark_gpu.mojo
```

From the darktable project root:
```bash
cd benchmark/blurs && mojo -I ../../mojo blurs_benchmark_gpu.mojo
```

## Performance Notes
The Mojo implementation uses a tiled approach with SIMD vectorization to ensure coalesced memory access on the GPU. This is designed to be significantly faster than a naive implementation.
